#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify backend speech recognition selection in the local voice application."""

import gzip
import json
import runpy
import struct
from pathlib import Path

import aiohttp
import pytest
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import serve

from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.services.stt_service import STTService
from pipecat.services.volcengine.stt import VolcengineSTTService
from pipecat.services.volcengine.tts import VolcengineTTSService
from pipecat.tests.utils import run_test

APP = Path(__file__).resolve().parents[1] / "local-voice-app" / "bot.py"


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
                "VOLCENGINE_RESOURCE_ID": "test-resource",
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
    assert headers[0]["X-Api-Resource-Id"] == "test-resource"
    assert requests[0]["request"] == {
        "model_name": "bigmodel",
        "result_type": "single",
        "show_utterances": True,
        "enable_nonstream": False,
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
                    "VOLCENGINE_RESOURCE_ID": "test-resource",
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
        ({"VOLCENGINE_API_KEY": "volc-test-key"}, "VOLCENGINE_RESOURCE_ID"),
        ({"VOLCENGINE_RESOURCE_ID": "test-resource"}, "VOLCENGINE_API_KEY"),
        (
            {"VOLCENGINE_API_KEY": "volc-test-key", "VOLCENGINE_RESOURCE_ID": " "},
            "VOLCENGINE_RESOURCE_ID",
        ),
        ({"VOLCENGINE_STT_OPTIONS": "{}"}, "VOLCENGINE_API_KEY"),
    ],
)
def test_incomplete_volcengine_configuration_fails_clearly(config, expected):
    app = runpy.run_path(str(APP))
    with pytest.raises(ValueError, match=expected) as error:
        app["create_stt_service"]({"ELEVENLABS_API_KEY": "test-key", **config})
    assert "volc-test-key" not in str(error.value)


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
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_RESOURCE_ID", raising=False)
    if provider == "volcengine":
        monkeypatch.setenv("VOLCENGINE_API_KEY", "volc-test-key")
        monkeypatch.setenv("VOLCENGINE_RESOURCE_ID", "test-resource")
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
                "VOLCENGINE_TTS_API_KEY": "volc-test-key",
                "VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0",
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
        ({"VOLCENGINE_TTS_API_KEY": "volc-test-key"}, "VOLCENGINE_TTS_RESOURCE_ID"),
        ({"VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0"}, "VOLCENGINE_TTS_API_KEY"),
        (
            {"VOLCENGINE_TTS_API_KEY": "volc-test-key", "VOLCENGINE_TTS_RESOURCE_ID": " "},
            "VOLCENGINE_TTS_RESOURCE_ID",
        ),
        (
            {"VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0", "VOLCENGINE_TTS_OPTIONS": "{}"},
            "VOLCENGINE_TTS_API_KEY",
        ),
        ({"VOLCENGINE_TTS_OPTIONS": "{}"}, "VOLCENGINE_TTS_API_KEY"),
        (
            {
                "VOLCENGINE_TTS_API_KEY": "volc-test-key",
                "VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0",
            },
            "VOLCENGINE_TTS_SPEAKER",
        ),
        (
            {
                "VOLCENGINE_TTS_API_KEY": "volc-test-key",
                "VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0",
                "VOLCENGINE_TTS_SPEAKER": " ",
            },
            "VOLCENGINE_TTS_SPEAKER",
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
                    "VOLCENGINE_TTS_API_KEY": "volc-test-key",
                    "VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0",
                    "VOLCENGINE_TTS_SPEAKER": "test-speaker",
                    "VOLCENGINE_TTS_OPTIONS": options,
                },
            )
        assert "secret-" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, tts_expected, stt_expected",
    [
        (
            {"VOLCENGINE_API_KEY": "volc-test-key", "VOLCENGINE_RESOURCE_ID": "test-resource"},
            ElevenLabsHttpTTSService,
            VolcengineSTTService,
        ),
        (
            {"VOLCENGINE_STT_OPTIONS": "{}"},
            ElevenLabsHttpTTSService,
            # A lone recognition option still selects Volcengine recognition, which
            # then requires the full credential pair.
            None,
        ),
        (
            {
                "VOLCENGINE_TTS_API_KEY": "volc-test-key",
                "VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-2.0",
                "VOLCENGINE_TTS_SPEAKER": "test-speaker",
            },
            VolcengineTTSService,
            ElevenLabsRealtimeSTTService,
        ),
        # A lone synthesis option still selects Volcengine synthesis, which then
        # requires the full credential set.
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
            with pytest.raises(ValueError, match="VOLCENGINE_TTS_API_KEY"):
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
    monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "volc-test-key")
    monkeypatch.setenv("VOLCENGINE_TTS_RESOURCE_ID", "seed-tts-2.0")
    monkeypatch.setenv("VOLCENGINE_TTS_SPEAKER", "test-speaker")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        assert isinstance(app["create_tts_service"](session), VolcengineTTSService)


@pytest.mark.asyncio
async def test_volcengine_session_still_requires_elevenlabs_tts_key(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", "volc-test-key")
    monkeypatch.setenv("VOLCENGINE_RESOURCE_ID", "test-resource")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = runpy.run_path(str(APP))
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            app["create_voice_pipeline"](None, session)
