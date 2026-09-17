#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify backend speech recognition selection in the local voice application."""

import asyncio
import gzip
import inspect
import io
import json
import os
import runpy
import socket
import struct
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import serve

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.volcengine.stt import VolcengineSTTService
from pipecat.services.volcengine.tts import VolcengineTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_transport import BaseTransport

APP = Path(__file__).resolve().parents[1] / "local-voice-app" / "bot.py"
FAKE_WIKI_API = Path(__file__).resolve().parent / "assets" / "fake_wiki_api.py"

# The application module, loaded for the names every test needs. A test that
# patches a setting loads its own copy of it.
_APP = runpy.run_path(str(APP))
KB_TOOL = _APP["KNOWLEDGE_TOOL_NAME"]

# A question the fake knowledge base answers from a page's text alone: its
# snippet is cut before the line that carries the level.
KB_QUESTION = "how severe is the axle counter section unknown alarm"
KB_ANSWER = "| Axle counter section unknown | 4 |"

_fake_wiki_api = runpy.run_path(str(FAKE_WIKI_API))
FakeWikiAPI = _fake_wiki_api["FakeWikiAPI"]

WIKI_TOKEN = "test-wiki-token"
WRITE_TOOL = "rebuild_index"

BACKEND_SETTINGS = (
    "VOLCENGINE_API_KEY",
    "VOLCENGINE_STT_OPTIONS",
    "VOLCENGINE_TTS_SPEAKER",
    "VOLCENGINE_TTS_OPTIONS",
    "WIKI_TOKEN",
    "WIKI_API_BASE_URL",
)

# Stands in for a machine where the optional MCP dependency was never installed:
# every import of it fails, so the application only runs if it reaches for
# nothing MCP provides. The path to the application module is passed as the
# first argument.
WITHOUT_MCP_PROGRAM = """
import runpy
import sys


class WithoutMcp:
    def find_spec(self, name, path=None, target=None):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, WithoutMcp())
app = runpy.run_path(sys.argv[1])
assert app["create_knowledge_base"]({}) is None
assert app["create_knowledge_base"]({"WIKI_TOKEN": "test-wiki-token"}) is not None
"""


class StandInTransport(BaseTransport):
    """A transport whose two ends are plain processors."""

    def __init__(self):
        super().__init__()
        self.input_processor = FrameProcessor()
        self.output_processor = FrameProcessor()

    def input(self):
        return self.input_processor

    def output(self):
        return self.output_processor


def llm_wiki(app, api, token=WIKI_TOKEN):
    """The session's knowledge base, pointed at a fake knowledge base API."""
    return app["LLMWikiKnowledgeBase"](token=token, base_url=api.base_url)


async def call_tool(llm, context, arguments, name=KB_TOOL):
    """Call one of a session's tools the way the service calls it.

    Args:
        llm: The service the tool is registered on.
        context: The session's conversation context.
        arguments: The arguments the model sent with the call.
        name: The tool to call.

    Returns:
        The results the handler settled the call with.
    """
    results = []

    async def result_callback(result, *, properties=None):
        results.append(result)

    await llm._functions[name].handler(
        FunctionCallParams(
            function_name=name,
            tool_call_id="call-1",
            arguments=arguments,
            llm=llm,
            pipeline_worker=SimpleNamespace(),
            context=context,
            result_callback=result_callback,
        )
    )
    return results


def use_backend(monkeypatch, **settings):
    """Set backend configuration to exactly the given settings."""
    for name in BACKEND_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)


def llm_of(pipeline):
    """Return the pipeline's LLM service."""
    return next(
        processor for processor in pipeline.processors if isinstance(processor, DeepSeekLLMService)
    )


def application_namespace(app):
    """Return the namespace the application's own functions read their globals from.

    ``runpy.run_path`` hands back a copy of the namespace it ran the application
    in, so replacing a module-level name has to happen here for the application
    to see it.
    """
    return app["create_voice_pipeline"].__globals__


def unused_local_port():
    """Return a port on the loopback interface that nothing is listening on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# Audio a stand-in synthesis service returns, and how long it takes: long enough
# that an interruption can land in the middle of the line it is speaking.
STAND_IN_TTS_SAMPLE_RATE = 16000
STAND_IN_TTS_CHUNK = b"\x00\x01" * 320
STAND_IN_TTS_CHUNKS = 10
STAND_IN_TTS_CHUNK_INTERVAL = 0.03


class StandInTTSService(TTSService):
    """A synthesis service that speaks a fixed number of audio chunks.

    Carries a line to the conversation the way the session does — through a real
    synthesis service and a real context aggregator — without a provider. Audio
    arrives from a background task, as it does from a streaming provider, so an
    interruption lands while the line is still being spoken.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            pause_frame_processing=False,
            sample_rate=STAND_IN_TTS_SAMPLE_RATE,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def speak():
            for _ in range(STAND_IN_TTS_CHUNKS):
                await asyncio.sleep(STAND_IN_TTS_CHUNK_INTERVAL)
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(
                        audio=STAND_IN_TTS_CHUNK,
                        sample_rate=STAND_IN_TTS_SAMPLE_RATE,
                        num_channels=1,
                        context_id=context_id,
                    ),
                )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(speak(), name=f"stand_in_tts_{context_id}")
        # An async generator that yields nothing, which is the shape of a service
        # delivering its audio from its own task.
        if False:
            yield


class ModelCallDriver(FrameProcessor):
    """Makes the calls a model would make, once the services that handle them run.

    A session's service has somewhere to run a call only after the pipeline
    starts it, so the calls go out on the ``StartFrame``. Each call is its own
    inference's worth of calls, the way a model makes them one turn at a time.
    """

    def __init__(self, llm, context, tool_names):
        super().__init__()
        self._llm = llm
        self._context = context
        self._tool_names = tool_names

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            for index, name in enumerate(self._tool_names):
                await self._llm.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            function_name=name,
                            tool_call_id=f"call-{index}",
                            arguments={"query": KB_QUESTION},
                            context=self._context,
                        )
                    ]
                )
        await self.push_frame(frame, direction)


async def other_tool_handler(params):
    """The handler of a tool a session has besides the knowledge base one."""
    await params.result_callback("done")


async def run_model_calls(
    app, session, knowledge_base, tool_names, pushed, *, extra=(), sleep=0.3, settle=False
):
    """Run a session through the calls a model would make.

    Args:
        app: The application module.
        session: HTTP session used by the HTTP synthesis provider.
        knowledge_base: The session's knowledge base, ``None`` for a session
            without one.
        tool_names: The tools the model calls, one call each.
        pushed: List every frame the session's LLM service pushes is appended to.
        extra: Names of tools the session also exposes, for the sessions whose
            model reaches for more than the knowledge base.
        sleep: How long the run stays open after the calls go out, so a call
            that outlives the session's timeout times out inside the run.
        settle: Whether the run ends the way a real session does, with the
            assembler pair's assistant side settling the calls in the context.

    Returns:
        The pipeline and its conversation context, for the caller to clean up.
    """
    pipeline, context = app["create_voice_pipeline"](StandInTransport(), session, knowledge_base)
    llm = llm_of(pipeline)
    for name in extra:
        llm.register_function(name, other_tool_handler)

    @llm.event_handler("on_after_push_frame")
    async def on_after_push_frame(service, frame):
        pushed.append(frame)

    # The session's own service, in a pipeline of its own: the assembly's other
    # processors want providers a test does not have.
    settling = [LLMAssistantAggregator(context)] if settle else []
    try:
        await run_test(
            Pipeline([ModelCallDriver(llm, context, tool_names), llm, *settling]),
            frames_to_send=[SleepFrame(sleep=sleep)],
            start_timeout=5,
        )
    finally:
        # The run is over, so the knowledge base it answered through holds
        # nothing open for whoever comes next.
        if knowledge_base is not None:
            await knowledge_base.aclose()
    return pipeline, context


def fillers_of(pushed):
    """The retrieval fillers among the frames a session pushed."""
    return [frame for frame in pushed if isinstance(frame, TTSSpeakFrame)]


def audio_of(frames):
    """The audio of a spoken line."""
    return [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]


def without_audio(frames):
    """A spoken line without its audio, which nothing downstream of the
    synthesis service acts on."""
    return [frame for frame in frames if not isinstance(frame, TTSAudioRawFrame)]


async def write_history(context, spoken):
    """Write a spoken line into a conversation, through a real assistant
    aggregator, and return what the conversation holds."""
    await run_test(LLMAssistantAggregator(context), frames_to_send=without_audio(spoken))
    return context.messages


@pytest.mark.parametrize("entrypoint", [APP, APP.parent / "bot" / "bot.py"])
def test_default_provider_remains_elevenlabs(entrypoint):
    app = runpy.run_path(str(entrypoint))
    service = app["create_stt_service"]({"ELEVENLABS_API_KEY": "test-key"})
    assert isinstance(service, ElevenLabsRealtimeSTTService)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", [APP, APP.parent / "bot" / "bot.py"])
async def test_default_synthesis_remains_elevenlabs(entrypoint):
    app = runpy.run_path(str(entrypoint))
    async with aiohttp.ClientSession() as session:
        service = app["create_tts_service"](
            session, {"ELEVENLABS_API_KEY": "test-key", "ELEVENLABS_VOICE_ID": "test-voice"}
        )
        assert isinstance(service, ElevenLabsHttpTTSService)
        assert service._settings.voice == "test-voice"
        assert service._settings.model == "eleven_multilingual_v2"


@pytest.mark.asyncio
async def test_volcengine_backend_options_reach_recognition_peer(monkeypatch):
    requests = []
    headers = []

    async def handler(websocket):
        headers.append(websocket.request.headers)
        requests.append(json.loads(gzip.decompress((await websocket.recv())[12:])))
        async for message in websocket:
            if struct.unpack(">i", message[4:8])[0] < 0:
                payload = b'{"result":{}}'
                await websocket.send(
                    b"\x11\x93\x10\x00" + struct.pack(">iI", -1, len(payload)) + payload
                )

    async with serve(handler, "127.0.0.1", 0) as server:

        async def connect(url, **kwargs):
            assert url == "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
            return await websocket_connect(
                f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", **kwargs
            )

        monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", connect)
        app = runpy.run_path(str(APP))
        service = app["create_stt_service"](
            {
                "VOLCENGINE_API_KEY": "volc-test-key",
                "VOLCENGINE_STT_OPTIONS": json.dumps(
                    {
                        "enable_itn": False,
                        "enable_punc": True,
                        "corpus_context": {"hotwords": [{"word": "Pipecat"}]},
                    }
                ),
            }
        )
        assert isinstance(service, STTService)
        await run_test(service, frames_to_send=[], start_timeout=5)

    assert headers[0]["X-Api-Key"] == "volc-test-key"
    assert headers[0]["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
    assert requests[0]["request"] == {
        "model_name": "bigmodel",
        "result_type": "single",
        "show_utterances": True,
        "enable_nonstream": True,
        "enable_itn": False,
        "enable_punc": True,
        "corpus": {"context": '{"hotwords": [{"word": "Pipecat"}]}'},
    }


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, "ELEVENLABS_API_KEY"),
        ({"ELEVENLABS_API_KEY": "  "}, "ELEVENLABS_API_KEY"),
        *[
            (
                {
                    "VOLCENGINE_API_KEY": "secret-test-key",
                    "VOLCENGINE_STT_OPTIONS": options,
                },
                "VOLCENGINE_STT_OPTIONS",
            )
            for options in [
                "secret-invalid-json",
                '{"enable_itn":"secret-invalid-value"}',
                '{"enable_nonstream":true}',
                '{"unknown":"secret-invalid-value"}',
            ]
        ],
    ],
)
def test_invalid_backend_configuration_fails_without_disclosing_values(config, expected):
    app = runpy.run_path(str(APP))
    with pytest.raises(ValueError, match=expected) as error:
        app["create_stt_service"](config)
    assert "secret-" not in str(error.value)


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"VOLCENGINE_STT_OPTIONS": "{}"}, "VOLCENGINE_API_KEY"),
        (
            {"VOLCENGINE_API_KEY": " ", "VOLCENGINE_STT_OPTIONS": "{}"},
            "VOLCENGINE_API_KEY",
        ),
    ],
)
def test_incomplete_volcengine_configuration_fails_clearly(config, expected):
    app = runpy.run_path(str(APP))
    with pytest.raises(ValueError, match=expected) as error:
        app["create_stt_service"]({"ELEVENLABS_API_KEY": "test-key", **config})
    assert "volc-test-key" not in str(error.value)


def test_volcengine_recognition_pins_the_model_version():
    app = runpy.run_path(str(APP))
    service = app["create_stt_service"]({"VOLCENGINE_API_KEY": "volc-test-key"})
    assert isinstance(service, VolcengineSTTService)
    assert service._resource_id == "volc.seedasr.sauc.duration"


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"VOLCENGINE_RESOURCE_ID": "secret-resource"}, "VOLCENGINE_RESOURCE_ID"),
        ({"VOLCENGINE_TTS_API_KEY": "secret-key"}, "VOLCENGINE_TTS_API_KEY"),
        ({"VOLCENGINE_TTS_RESOURCE_ID": "secret-resource"}, "VOLCENGINE_TTS_RESOURCE_ID"),
    ],
)
def test_retired_volcengine_settings_fail_without_disclosing_values(config, expected):
    app = runpy.run_path(str(APP))
    with pytest.raises(ValueError, match=expected) as error:
        app["create_stt_service"]({"ELEVENLABS_API_KEY": "test-key", **config})
    assert "secret-" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["elevenlabs", "volcengine"])
async def test_selected_provider_uses_existing_voice_pipeline(provider, monkeypatch):
    from pipecat.pipeline.pipeline import PipelineSink, PipelineSource
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregator,
        LLMUserAggregator,
    )
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.services.deepseek.llm import DeepSeekLLMService
    from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
    from pipecat.services.volcengine.stt import VolcengineSTTService
    from pipecat.transports.base_transport import BaseTransport

    class Transport(BaseTransport):
        def __init__(self):
            super().__init__()
            self.input_processor = FrameProcessor()
            self.output_processor = FrameProcessor()

        def input(self):
            return self.input_processor

        def output(self):
            return self.output_processor

    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    for name in (
        "VOLCENGINE_API_KEY",
        "VOLCENGINE_RESOURCE_ID",
        "VOLCENGINE_STT_OPTIONS",
        "VOLCENGINE_TTS_API_KEY",
        "VOLCENGINE_TTS_RESOURCE_ID",
        "VOLCENGINE_TTS_SPEAKER",
        "VOLCENGINE_TTS_OPTIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    if provider == "volcengine":
        monkeypatch.setenv("VOLCENGINE_API_KEY", "volc-test-key")
    transport = Transport()
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        pipeline, context = app["create_voice_pipeline"](transport, session)
        processors = [
            processor
            for processor in pipeline.processors
            if not isinstance(processor, (PipelineSource, PipelineSink))
        ]
        assert [type(processor) for processor in processors] == [
            FrameProcessor,
            ElevenLabsRealtimeSTTService if provider == "elevenlabs" else VolcengineSTTService,
            LLMUserAggregator,
            DeepSeekLLMService,
            ElevenLabsHttpTTSService,
            FrameProcessor,
            LLMAssistantAggregator,
        ]
        assert processors[0] is transport.input()
        assert processors[5] is transport.output()
        assert context.messages == []
        await pipeline.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, {}),
        ({"VOLCENGINE_TTS_OPTIONS": "{}"}, {}),
        (
            {"VOLCENGINE_TTS_OPTIONS": '{"audio_format":"mp3","sample_rate":24000}'},
            {"audio_format": "mp3", "sample_rate": 24000},
        ),
        (
            {
                "VOLCENGINE_TTS_OPTIONS": '{"sample_rate":48000,"additions":"{\\"disable_markdown_filter\\":true}"}'
            },
            {"sample_rate": 48000, "additions": '{"disable_markdown_filter":true}'},
        ),
    ],
)
async def test_volcengine_synthesis_configuration_reaches_service(config, expected):
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        service = app["create_tts_service"](
            session,
            {
                "VOLCENGINE_API_KEY": "volc-test-key",
                "VOLCENGINE_TTS_SPEAKER": "test-speaker",
                **config,
            },
        )
        assert isinstance(service, VolcengineTTSService)
        assert service._params.speaker == "test-speaker"
        assert service._resource_id == "seed-tts-2.0"
        assert service._api_key == "volc-test-key"
        params = {"audio_format": "pcm", "sample_rate": None, "additions": None}
        params.update(expected)
        assert service._params.audio_format == params["audio_format"]
        assert service._params.sample_rate == params["sample_rate"]
        assert service._params.additions == params["additions"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, expected",
    [
        ({"VOLCENGINE_TTS_OPTIONS": "{}"}, "VOLCENGINE_API_KEY"),
        ({"VOLCENGINE_TTS_SPEAKER": "test-speaker"}, "VOLCENGINE_API_KEY"),
        (
            {"VOLCENGINE_API_KEY": "volc-test-key", "VOLCENGINE_TTS_SPEAKER": " "},
            "VOLCENGINE_TTS_SPEAKER",
        ),
        (
            {"VOLCENGINE_API_KEY": " ", "VOLCENGINE_TTS_OPTIONS": "{}"},
            "VOLCENGINE_API_KEY",
        ),
    ],
)
async def test_incomplete_volcengine_synthesis_configuration_fails_clearly(config, expected):
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match=expected) as error:
            app["create_tts_service"](session, {"ELEVENLABS_API_KEY": "test-key", **config})
        assert "volc-test-key" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        "secret-invalid-json",
        '{"sample_rate":"secret-invalid-value"}',
        '{"audio_format":["secret-invalid-value"]}',
        '{"unknown":"secret-invalid-value"}',
        '{"speaker":"secret-invalid-speaker"}',
        "[1, 2]",
    ],
)
async def test_invalid_volcengine_synthesis_options_fail_without_disclosing_values(options):
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="VOLCENGINE_TTS_OPTIONS") as error:
            app["create_tts_service"](
                session,
                {
                    "VOLCENGINE_API_KEY": "volc-test-key",
                    "VOLCENGINE_TTS_SPEAKER": "test-speaker",
                    "VOLCENGINE_TTS_OPTIONS": options,
                },
            )
        assert "secret-" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, tts_expected, stt_expected",
    [
        # The shared key alone selects Volcengine recognition; synthesis stays on
        # ElevenLabs until a synthesis setting appears.
        (
            {"VOLCENGINE_API_KEY": "volc-test-key"},
            ElevenLabsHttpTTSService,
            VolcengineSTTService,
        ),
        # A lone recognition option also selects Volcengine recognition, which then
        # requires the shared key.
        (
            {"VOLCENGINE_STT_OPTIONS": "{}"},
            ElevenLabsHttpTTSService,
            None,
        ),
        (
            {"VOLCENGINE_API_KEY": "volc-test-key", "VOLCENGINE_TTS_SPEAKER": "test-speaker"},
            VolcengineTTSService,
            VolcengineSTTService,
        ),
        # A lone synthesis option selects Volcengine synthesis, which then requires
        # the shared key and the voice.
        ({"VOLCENGINE_TTS_OPTIONS": "{}"}, None, ElevenLabsRealtimeSTTService),
    ],
)
async def test_synthesis_and_recognition_selection_are_independent(
    config, tts_expected, stt_expected
):
    app = runpy.run_path(str(APP))
    config = {"ELEVENLABS_API_KEY": "test-key", **config}
    async with aiohttp.ClientSession() as session:
        if tts_expected is None:
            with pytest.raises(ValueError, match="VOLCENGINE_API_KEY"):
                app["create_tts_service"](session, config)
        else:
            assert isinstance(app["create_tts_service"](session, config), tts_expected)
        if stt_expected is None:
            with pytest.raises(ValueError, match="VOLCENGINE_API_KEY"):
                app["create_stt_service"](config)
        else:
            assert isinstance(app["create_stt_service"](config), stt_expected)


@pytest.mark.asyncio
async def test_volcengine_synthesis_does_not_require_elevenlabs_key(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", "volc-test-key")
    monkeypatch.setenv("VOLCENGINE_TTS_SPEAKER", "test-speaker")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        assert isinstance(app["create_tts_service"](session), VolcengineTTSService)


@pytest.mark.asyncio
async def test_volcengine_session_still_requires_elevenlabs_tts_key(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", "volc-test-key")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            app["create_voice_pipeline"](None, session)


@pytest.mark.parametrize(
    "config",
    [
        # No token at all.
        {},
        # An address on its own configures nothing: the token is what turns the
        # knowledge base on.
        {"WIKI_API_BASE_URL": "http://127.0.0.1:19828"},
        # A token left blank is one left to be filled in later, so it turns the
        # knowledge base off the way an absent one does rather than failing the
        # session over a feature nobody asked for.
        {"WIKI_TOKEN": ""},
        {"WIKI_TOKEN": " "},
    ],
)
def test_a_wiki_token_without_a_value_disables_the_knowledge_base(config):
    app = runpy.run_path(str(APP))
    assert app["create_knowledge_base"](config) is None


def test_wiki_token_enables_the_knowledge_base():
    app = runpy.run_path(str(APP))
    configured = app["create_knowledge_base"]({"WIKI_TOKEN": WIKI_TOKEN})
    assert isinstance(configured, app["LLMWikiKnowledgeBase"])
    # The desktop application's API is what a session reaches by default.
    assert configured._base_url == app["DEFAULT_WIKI_API_BASE_URL"] == "http://127.0.0.1:19828"


@pytest.mark.parametrize("declared", ["http://127.0.0.1:9999", "http://127.0.0.1:9999/"])
def test_wiki_api_base_url_overrides_the_default(declared):
    app = runpy.run_path(str(APP))
    configured = app["create_knowledge_base"](
        {"WIKI_TOKEN": WIKI_TOKEN, "WIKI_API_BASE_URL": declared}
    )
    assert configured._base_url == "http://127.0.0.1:9999"


@pytest.mark.parametrize(
    "config, expected",
    [
        # An address a lookup could not be sent to.
        (
            {"WIKI_TOKEN": WIKI_TOKEN, "WIKI_API_BASE_URL": "secret-wiki-value"},
            "WIKI_API_BASE_URL",
        ),
        ({"WIKI_TOKEN": WIKI_TOKEN, "WIKI_API_BASE_URL": "ftp://wiki"}, "WIKI_API_BASE_URL"),
        ({"WIKI_TOKEN": WIKI_TOKEN, "WIKI_API_BASE_URL": "http://"}, "WIKI_API_BASE_URL"),
    ],
)
def test_invalid_wiki_configuration_fails_without_disclosing_values(config, expected):
    app = runpy.run_path(str(APP))
    with pytest.raises(ValueError, match=expected) as error:
        app["create_knowledge_base"](config)
    assert "secret-wiki-value" not in str(error.value)
    assert WIKI_TOKEN not in str(error.value)


@pytest.mark.parametrize("entrypoint", [APP, APP.parent / "bot" / "bot.py"])
def test_wiki_configuration_is_available_through_both_entrypoints(entrypoint):
    app = runpy.run_path(str(entrypoint))
    configured = app["create_knowledge_base"]({"WIKI_TOKEN": WIKI_TOKEN})
    assert isinstance(configured, app["KnowledgeBase"])


def test_wiki_configuration_is_synchronous_and_connects_to_nothing():
    app = runpy.run_path(str(APP))
    create = app["create_knowledge_base"]
    assert not inspect.iscoroutinefunction(create)
    # An address nothing listens on is fine here: reading the configuration
    # opens nothing, so a session whose knowledge base is down is created like
    # any other and fails a lookup at a time.
    down = create({"WIKI_TOKEN": WIKI_TOKEN, "WIKI_API_BASE_URL": "http://127.0.0.1:1"})
    assert down._session is None


def test_the_application_runs_without_the_mcp_dependency():
    # The knowledge base is reached over HTTP, so the application runs on a
    # machine that never installed the optional MCP dependency, whether or not a
    # knowledge base is configured.
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_MCP_PROGRAM, str(APP)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_invalid_wiki_configuration_fails_session_creation(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("WIKI_TOKEN", WIKI_TOKEN)
    monkeypatch.setenv("WIKI_API_BASE_URL", "secret-wiki-value")
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        # The session fails before it reaches the transport or the runner, so
        # neither is needed to reach the failure.
        with pytest.raises(ValueError, match="WIKI_API_BASE_URL") as error:
            await app["run_bot_session"](None, None, session)
        assert "secret-wiki-value" not in str(error.value)


@pytest.mark.asyncio
async def test_only_the_knowledge_tool_is_registered(monkeypatch):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, context = app["create_voice_pipeline"](
            StandInTransport(), session, llm_wiki(app, api)
        )
        try:
            llm = llm_of(pipeline)
            assert set(llm._functions) == {KB_TOOL}
            # The registration's deadline must fall after the handler's own, so
            # that a slow lookup is settled by the handler as one that found
            # nothing rather than cancelled by the service, which reads as the
            # user having interrupted it.
            assert llm._functions[KB_TOOL].timeout_secs == app["KNOWLEDGE_BACKSTOP_TIMEOUT_SECS"]
            assert llm._functions[KB_TOOL].timeout_secs > app["KNOWLEDGE_LOOKUP_TIMEOUT_SECS"]
            assert [schema.name for schema in context.tools.standard_tools] == [KB_TOOL]
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_knowledge_base_leaves_the_pipeline_and_its_context_unchanged(monkeypatch):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        baseline, baseline_context = app["create_voice_pipeline"](StandInTransport(), session)
        exposed, _ = app["create_voice_pipeline"](StandInTransport(), session, llm_wiki(app, api))
        assert [type(processor) for processor in exposed.processors] == [
            type(processor) for processor in baseline.processors
        ]
        # Without the argument, the session is the one it was before the
        # knowledge base existed: no tools, and the base instruction alone.
        assert set(llm_of(baseline)._functions) == set()
        assert llm_of(baseline)._settings.system_instruction == app["SYSTEM_INSTRUCTION"]
        assert not baseline_context.tools
        await baseline.cleanup()
        await exposed.cleanup()


@pytest.mark.asyncio
async def test_knowledge_instruction_is_appended_only_when_the_knowledge_base_is_present(
    monkeypatch,
):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, _ = app["create_voice_pipeline"](StandInTransport(), session, llm_wiki(app, api))
        try:
            instruction = llm_of(pipeline)._settings.system_instruction
            assert instruction == (
                f"{app['SYSTEM_INSTRUCTION']} {app['KNOWLEDGE_BASE_INSTRUCTION']}"
            )
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_a_lookup_answers_from_the_pages_the_search_ranks_highest(monkeypatch):
    # What the model reads is page text, not a snippet: the fake API's snippet is
    # cut before the line that carries the answer, which is the whole reason a
    # lookup reads pages. The hits that rank below the pages it reads are left
    # out, so a lookup cannot fill the context with everything the search found.
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI(token=WIKI_TOKEN) as api, aiohttp.ClientSession() as session:
        knowledge_base = llm_wiki(app, api)
        pipeline, context = app["create_voice_pipeline"](
            StandInTransport(), session, knowledge_base
        )
        try:
            results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
        finally:
            await pipeline.cleanup()
            await knowledge_base.aclose()
    assert len(results) == 1
    assert KB_ANSWER in results[0]
    assert "wiki/sources/bdms-alarms.md" in results[0]
    assert "wiki/index.md" not in results[0]
    # The request carries the token and the parameters the API takes, with a
    # size it accepts: it rejects anything outside 1 to 50.
    assert 1 <= app["WIKI_SEARCH_SIZE"] <= 50
    assert api.requests == [
        {
            "method": "POST",
            "path": "/api/v1/projects/current/search",
            "authorization": f"Bearer {WIKI_TOKEN}",
            "query": {},
            "body": {
                "query": KB_QUESTION,
                "topK": app["WIKI_SEARCH_SIZE"],
                "includeContent": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_a_page_the_search_does_not_carry_is_read_by_path(monkeypatch):
    # An API that ignores includeContent answers with paths alone, and a lookup
    # still has to answer from page text rather than from snippets.
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI(include_content=False) as api, aiohttp.ClientSession() as session:
        knowledge_base = llm_wiki(app, api)
        pipeline, context = app["create_voice_pipeline"](
            StandInTransport(), session, knowledge_base
        )
        try:
            results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
        finally:
            await pipeline.cleanup()
            await knowledge_base.aclose()
    assert KB_ANSWER in results[0]
    assert [request["method"] for request in api.requests] == ["POST", "GET", "GET"]
    assert [request["query"] for request in api.requests[1:]] == [
        {"path": "wiki/entities/axle-counter.md"},
        {"path": "wiki/sources/bdms-alarms.md"},
    ]


@pytest.mark.asyncio
async def test_a_page_that_cannot_be_read_leaves_the_pages_already_read(monkeypatch):
    # The hits are ranked, so the page that answers the question is one of the
    # first ones read. A read that fails leaves that page out of the lookup
    # rather than abandoning it: abandoning it would leave a lookup that found
    # something looking like one that found nothing, which is what the read
    # exists to prevent.
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    hits = [
        {
            "path": "wiki/sources/bdms-alarms.md",
            "title": "BDMS alarms",
            "snippet": KB_ANSWER,
            "score": 30.0,
            "content": f"# BDMS alarms\n\n## Axle counter\n\n{KB_ANSWER}\n",
        },
        {
            "path": "wiki/entities/axle-counter.md",
            "title": "Axle counter",
            "snippet": "## Alarms in the source",
            "score": 16.0,
            # A page that carries no text of the search's own and cannot be read,
            # the way one deleted between the search and the read cannot.
            "content_status": 404,
        },
    ]
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        async with FakeWikiAPI(token=WIKI_TOKEN, results=hits) as api:
            async with aiohttp.ClientSession() as session:
                knowledge_base = llm_wiki(app, api)
                pipeline, context = app["create_voice_pipeline"](
                    StandInTransport(), session, knowledge_base
                )
                try:
                    results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
                finally:
                    await pipeline.cleanup()
                    await knowledge_base.aclose()
    finally:
        logger.remove(handler_id)
    # The page that was read is what the lookup answers from, rather than the
    # whole lookup coming back empty.
    assert len(results) == 1
    assert KB_ANSWER in results[0]
    assert "wiki/sources/bdms-alarms.md" in results[0]
    assert "wiki/entities/axle-counter.md" not in results[0]
    # Logged as the degraded lookup it is, without the token.
    logged = sink.getvalue()
    assert "wiki/entities/axle-counter.md" in logged
    assert "could not be read" in logged
    assert WIKI_TOKEN not in logged


@pytest.mark.asyncio
async def test_a_lookup_that_finds_nothing_is_reported_as_finding_nothing(monkeypatch):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        async with FakeWikiAPI(results=[]) as api, aiohttp.ClientSession() as session:
            knowledge_base = llm_wiki(app, api)
            pipeline, context = app["create_voice_pipeline"](
                StandInTransport(), session, knowledge_base
            )
            try:
                results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
            finally:
                await pipeline.cleanup()
                await knowledge_base.aclose()
    finally:
        logger.remove(handler_id)
    # The model reads a lookup that found nothing rather than an empty result,
    # which the knowledge base instruction turns into "I could not find it".
    assert results == [app["NO_LOOKUP_RESULT"]]
    # Logged apart from a lookup that could not be reached.
    assert "found nothing" in sink.getvalue()
    assert "failed" not in sink.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unreachable, expected_error",
    [
        # A knowledge base whose API rejects the configured token.
        ("unauthorized", "ClientResponseError"),
        # A knowledge base that is not running at all.
        ("unreachable", "ClientConnectorError"),
    ],
)
async def test_a_lookup_that_cannot_reach_the_knowledge_base_reports_finding_nothing(
    monkeypatch, unreachable, expected_error
):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    api = FakeWikiAPI(
        token="a-token-the-application-does-not-have" if unreachable == "unauthorized" else None
    )
    base_url = (
        await api.start()
        if unreachable == "unauthorized"
        else f"http://127.0.0.1:{unused_local_port()}"
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        async with aiohttp.ClientSession() as session:
            knowledge_base = app["LLMWikiKnowledgeBase"](token=WIKI_TOKEN, base_url=base_url)
            pipeline, context = app["create_voice_pipeline"](
                StandInTransport(), session, knowledge_base
            )
            try:
                results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
            finally:
                await pipeline.cleanup()
                await knowledge_base.aclose()
    finally:
        logger.remove(handler_id)
        await api.aclose()
    # The user is told the answer could not be found, and the session carries on:
    # a knowledge base that cannot be reached is not a failed conversation.
    assert results == [app["NO_LOOKUP_RESULT"]]
    logged = sink.getvalue()
    # Logged apart from a lookup that found nothing, with the kind of failure but
    # never the address or the token the lookup was reaching for.
    assert f"Knowledge base lookup failed ({expected_error})" in logged
    assert "found nothing" not in logged
    assert WIKI_TOKEN not in logged
    assert base_url not in logged


@pytest.mark.asyncio
async def test_a_lookup_that_never_returns_times_out_and_asks_for_an_answer(monkeypatch):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    # The timeout is a constant, so a session that has to be seen timing out is
    # given a shorter one.
    application_namespace(app)["KNOWLEDGE_LOOKUP_TIMEOUT_SECS"] = 0.5
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    pushed = []
    try:
        async with FakeWikiAPI(delay_secs=60) as api, aiohttp.ClientSession() as session:
            pipeline, context = await run_model_calls(
                app, session, llm_wiki(app, api), [KB_TOOL], pushed, sleep=1.5, settle=True
            )
            try:
                # The lookup reached the knowledge base, so the deadline was
                # measured against a real one rather than a handler that never ran.
                assert [request["method"] for request in api.requests] == ["POST"]
                # The lookup's own deadline is what fired, and it fired at the
                # call: the model is asked for the answer the abandoned lookup
                # can no longer provide. Nothing cancels the call out from under
                # it, which is what an interruption looks like.
                logged = sink.getvalue()
                assert "Knowledge base lookup timed out after 0.5 seconds" in logged
                assert "is being cancelled" not in logged
                assert not [frame for frame in pushed if isinstance(frame, FunctionCallCancelFrame)]
                # The call settles in the conversation as one that found nothing,
                # which is what the model reads and answers from: the session
                # carries on instead of waiting on a result that can never arrive.
                assert {
                    frame.result for frame in pushed if isinstance(frame, FunctionCallResultFrame)
                } == {app["NO_LOOKUP_RESULT"]}
                assert [message["role"] for message in context.messages] == ["assistant", "tool"]
                assert context.messages[0]["tool_calls"][0]["function"]["name"] == KB_TOOL
                assert context.messages[1]["content"] == json.dumps(app["NO_LOOKUP_RESULT"])
            finally:
                await pipeline.cleanup()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_a_knowledge_base_that_is_down_does_not_fail_session_creation(monkeypatch):
    # Reading the configuration opens nothing, so a session whose knowledge base
    # is down is created as usual: it carries on past the knowledge base and
    # fails on the provider configuration the session also needs.
    app = runpy.run_path(str(APP))
    use_backend(
        monkeypatch,
        ELEVENLABS_API_KEY="test-key",
        WIKI_TOKEN=WIKI_TOKEN,
        WIKI_API_BASE_URL=f"http://127.0.0.1:{unused_local_port()}",
    )
    async with aiohttp.ClientSession() as session:
        with pytest.raises(KeyError, match="DEEPSEEK_API_KEY"):
            await app["run_bot_session"](None, None, session)


@pytest.mark.asyncio
async def test_a_session_that_fails_releases_its_knowledge_base(monkeypatch):
    # A session that stops before it ever listens still releases what the
    # knowledge base holds open.
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", WIKI_TOKEN=WIKI_TOKEN)
    released = []

    class RecordingKnowledgeBase(app["KnowledgeBase"]):
        async def retrieve(self, query: str) -> str:
            return ""

        async def aclose(self) -> None:
            released.append(True)

    application_namespace(app)["create_knowledge_base"] = lambda environ=None: (
        RecordingKnowledgeBase()
    )
    async with aiohttp.ClientSession() as session:
        with pytest.raises(KeyError, match="DEEPSEEK_API_KEY"):
            await app["run_bot_session"](None, None, session)
    assert released == [True]


@pytest.mark.asyncio
async def test_a_lookup_does_not_go_through_a_proxy_from_the_environment(monkeypatch):
    # The knowledge base answers on loopback, so a proxy read from the
    # environment has no business carrying the lookup: it would take the request
    # to an address that does not answer for the knowledge base.
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{unused_local_port()}")
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{unused_local_port()}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        knowledge_base = llm_wiki(app, api)
        pipeline, context = app["create_voice_pipeline"](
            StandInTransport(), session, knowledge_base
        )
        try:
            results = await call_tool(llm_of(pipeline), context, {"query": KB_QUESTION})
        finally:
            await pipeline.cleanup()
            await knowledge_base.aclose()
    assert KB_ANSWER in results[0]


@pytest.mark.asyncio
async def test_a_lookup_releases_the_connection_it_opened(monkeypatch):
    app = runpy.run_path(str(APP))
    async with FakeWikiAPI() as api:
        knowledge_base = llm_wiki(app, api)
        # A knowledge base nothing has looked anything up in holds nothing open.
        await knowledge_base.aclose()
        assert knowledge_base._session is None
        await knowledge_base.retrieve(KB_QUESTION)
        client = knowledge_base._session
        assert client is not None and not client.closed
        await knowledge_base.aclose()
        assert client.closed


@pytest.mark.asyncio
async def test_a_cancel_that_overtakes_the_call_in_progress_is_dropped():
    # A deadline's cancel is a system frame, so it overtakes the call's own
    # in-progress frame whenever the pipeline is busy ahead of it — the filler
    # being spoken, say. The assembler settles a call it has already seen start
    # and drops a cancel for one it has not, which is what leaves the user
    # waiting on an answer that never comes when the timeout is shorter than
    # that wait. Pinned here so a change in either half is noticed.
    overtaken = LLMContext()
    _, upstream = await run_test(
        LLMAssistantAggregator(overtaken),
        frames_to_send=[
            FunctionCallCancelFrame(function_name=KB_TOOL, tool_call_id="call-0", run_llm=True),
            FunctionCallInProgressFrame(
                function_name=KB_TOOL,
                tool_call_id="call-0",
                arguments={"query": "what is pipecat"},
                cancel_on_interruption=True,
            ),
        ],
    )
    assert [message["content"] for message in overtaken.messages if message["role"] == "tool"] == [
        "IN_PROGRESS"
    ]
    assert not [frame for frame in upstream if isinstance(frame, LLMContextFrame)]

    # The same cancel arriving after the call started settles it and asks the
    # model for the answer.
    settled = LLMContext()
    _, upstream = await run_test(
        LLMAssistantAggregator(settled),
        frames_to_send=[
            FunctionCallInProgressFrame(
                function_name=KB_TOOL,
                tool_call_id="call-0",
                arguments={"query": "what is pipecat"},
                cancel_on_interruption=True,
            ),
            # Frames sent in one batch reach the aggregator cancel-first, since
            # a system frame jumps the queue. The pause is what delivers them in
            # the other order, which is the order a call that runs takes.
            SleepFrame(sleep=0.1),
            FunctionCallCancelFrame(function_name=KB_TOOL, tool_call_id="call-0", run_llm=True),
        ],
    )
    assert [message["content"] for message in settled.messages if message["role"] == "tool"] == [
        "CANCELLED"
    ]
    assert [frame for frame in upstream if isinstance(frame, LLMContextFrame)]


@pytest.mark.asyncio
async def test_the_call_in_progress_frame_goes_out_before_the_filler(monkeypatch):
    # The order the two leave the service in is what keeps the call's own frames
    # out from behind the spoken filler: the synthesis service emits everything
    # else in order relative to the audio it is playing, so an in-progress frame
    # queued behind the filler reaches the assembler only once the filler has
    # played. A deadline that expires before then leaves the assembler dropping
    # the cancel as one for a call it never saw start, and the question goes
    # unanswered.
    pushed = []
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, _ = await run_model_calls(app, session, llm_wiki(app, api), [KB_TOOL], pushed)
        try:
            first_in_progress = next(
                index
                for index, frame in enumerate(pushed)
                if isinstance(frame, FunctionCallInProgressFrame)
            )
            first_filler = next(
                index for index, frame in enumerate(pushed) if isinstance(frame, TTSSpeakFrame)
            )
            assert first_in_progress < first_filler
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_retrieval_filler_is_spoken_while_the_lookup_runs(monkeypatch):
    pushed = []
    in_flight = []

    class SamplingKnowledgeBase:
        """A knowledge base that reports what the session pushed before it
        answered, which is what the user has heard by the time it does."""

        def __init__(self, knowledge_base):
            self._knowledge_base = knowledge_base

        async def retrieve(self, query: str) -> str:
            content = await self._knowledge_base.retrieve(query)
            in_flight.extend(fillers_of(pushed))
            return content

        async def aclose(self) -> None:
            await self._knowledge_base.aclose()

    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, _ = await run_model_calls(
            app, session, SamplingKnowledgeBase(llm_wiki(app, api)), [KB_TOOL], pushed
        )
        try:
            fillers = fillers_of(pushed)
            assert [frame.text for frame in fillers] == [app["RETRIEVAL_FILLER"]]
            # Spoken to the user rather than said to the model.
            assert fillers[0].append_to_context is False
            assert [frame.text for frame in in_flight] == [app["RETRIEVAL_FILLER"]]
            # The lookup the filler announced still ran to a result.
            assert any(isinstance(frame, FunctionCallResultFrame) for frame in pushed)
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_retrieval_filler_is_spoken_only_for_the_knowledge_tool(monkeypatch):
    pushed = []
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    # A session whose model also reaches for a tool that writes to the knowledge
    # base. Only the knowledge base tool's calls are announced.
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, _ = await run_model_calls(
            app,
            session,
            llm_wiki(app, api),
            [WRITE_TOOL, KB_TOOL],
            pushed,
            extra=[WRITE_TOOL],
        )
        try:
            assert {
                frame.function_name
                for frame in pushed
                if isinstance(frame, FunctionCallResultFrame)
            } == {WRITE_TOOL, KB_TOOL}
            assert [frame.text for frame in fillers_of(pushed)] == [app["RETRIEVAL_FILLER"]]
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_retrieval_filler_does_not_enter_the_conversation_history(monkeypatch):
    pushed = []
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, context = await run_model_calls(
            app, session, llm_wiki(app, api), [KB_TOOL], pushed
        )
        try:
            filler = fillers_of(pushed)[0]
            # Spoken through a real synthesis service, and written into the
            # session's own conversation by a real assistant aggregator.
            spoken, _ = await run_test(
                StandInTTSService(), frames_to_send=[filler, SleepFrame(sleep=0.5)]
            )
            assert await write_history(context, spoken) == []

            # The flag is what holds it back: the same line, asked to be kept.
            kept, _ = await run_test(
                StandInTTSService(),
                frames_to_send=[TTSSpeakFrame(text=filler.text), SleepFrame(sleep=0.5)],
            )
            history = await write_history(LLMContext(), kept)
            assert [message["content"] for message in history] == [filler.text]
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_interrupting_the_filler_stops_it_and_leaves_no_message(monkeypatch):
    pushed = []
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    async with FakeWikiAPI() as api, aiohttp.ClientSession() as session:
        pipeline, context = await run_model_calls(
            app, session, llm_wiki(app, api), [KB_TOOL], pushed
        )
        try:
            filler = fillers_of(pushed)[0]
            whole, _ = await run_test(
                StandInTTSService(), frames_to_send=[filler, SleepFrame(sleep=0.5)]
            )
            # The user speaks again partway through the filler.
            interrupted, _ = await run_test(
                StandInTTSService(),
                frames_to_send=[
                    filler,
                    SleepFrame(sleep=0.06),
                    InterruptionFrame(),
                    SleepFrame(sleep=0.2),
                ],
            )
            assert len(audio_of(interrupted)) < len(audio_of(whole))
            # An interrupted line is not something the assistant said.
            assert await write_history(context, interrupted) == []
        finally:
            await pipeline.cleanup()


@pytest.mark.asyncio
async def test_a_session_without_a_knowledge_base_speaks_no_filler(monkeypatch):
    app = runpy.run_path(str(APP))
    use_backend(monkeypatch, ELEVENLABS_API_KEY="test-key", DEEPSEEK_API_KEY="test-key")
    # A session with nothing configured answers as it did before the knowledge
    # base existed: the tool is not registered, so no call can be announced.
    async with aiohttp.ClientSession() as session:
        pushed = []
        pipeline, _ = await run_model_calls(app, session, None, [KB_TOOL], pushed)
        try:
            assert fillers_of(pushed) == []
        finally:
            await pipeline.cleanup()
