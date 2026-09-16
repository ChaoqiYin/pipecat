#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""Browser voice application using DeepSeek, ElevenLabs, and Volcengine."""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.volcengine.stt import VolcengineSTTService
from pipecat.services.volcengine.tts import VolcengineTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

VOLCENGINE_STT_SETTINGS = (
    "VOLCENGINE_API_KEY",
    "VOLCENGINE_RESOURCE_ID",
    "VOLCENGINE_STT_OPTIONS",
)

VOLCENGINE_TTS_SETTINGS = (
    "VOLCENGINE_TTS_API_KEY",
    "VOLCENGINE_TTS_RESOURCE_ID",
    "VOLCENGINE_TTS_SPEAKER",
    "VOLCENGINE_TTS_OPTIONS",
)

ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"


def _required_setting(config: Mapping[str, str], name: str) -> str:
    value = config.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required in backend configuration")
    return value


def create_stt_service(environ: Mapping[str, str] | None = None) -> STTService:
    """Create the session's recognition service from backend configuration.

    Args:
        environ: Environment variables, defaulting to the process environment.

    Returns:
        The configured recognition service.

    Raises:
        ValueError: A recognition option is invalid.
    """
    config = os.environ if environ is None else environ
    if not any(name in config for name in VOLCENGINE_STT_SETTINGS):
        return ElevenLabsRealtimeSTTService(api_key=_required_setting(config, "ELEVENLABS_API_KEY"))
    api_key = _required_setting(config, "VOLCENGINE_API_KEY")
    resource_id = _required_setting(config, "VOLCENGINE_RESOURCE_ID")
    try:
        params = VolcengineSTTService.InputParams.model_validate_json(
            config.get("VOLCENGINE_STT_OPTIONS", "{}")
        )
    except ValidationError:
        raise ValueError(
            "VOLCENGINE_STT_OPTIONS must be a JSON object with supported recognition options; "
            "second-pass recognition is not supported"
        ) from None
    return VolcengineSTTService(api_key=api_key, resource_id=resource_id, params=params)


def create_tts_service(
    session: aiohttp.ClientSession, environ: Mapping[str, str] | None = None
) -> TTSService:
    """Create the session's synthesis service from backend configuration.

    Args:
        session: HTTP session used by the HTTP synthesis provider.
        environ: Environment variables, defaulting to the process environment.

    Returns:
        The configured synthesis service.

    Raises:
        ValueError: A synthesis setting is missing or an option is invalid.
    """
    config = os.environ if environ is None else environ
    if not any(name in config for name in VOLCENGINE_TTS_SETTINGS):
        return ElevenLabsHttpTTSService(
            aiohttp_session=session,
            api_key=_required_setting(config, "ELEVENLABS_API_KEY"),
            settings=ElevenLabsHttpTTSService.Settings(
                voice=config.get("ELEVENLABS_VOICE_ID", ""),
                model=ELEVENLABS_TTS_MODEL,
            ),
        )
    api_key = _required_setting(config, "VOLCENGINE_TTS_API_KEY")
    resource_id = _required_setting(config, "VOLCENGINE_TTS_RESOURCE_ID")
    speaker = _required_setting(config, "VOLCENGINE_TTS_SPEAKER")
    options = config.get("VOLCENGINE_TTS_OPTIONS", "{}")
    try:
        # The voice is configured on its own, so it is supplied here and excluded
        # from the options object.
        params = VolcengineTTSService.InputParams.model_validate_json(
            json.dumps({"speaker": speaker, **json.loads(options)})
        )
    except (ValidationError, json.JSONDecodeError, TypeError):
        raise ValueError(
            "VOLCENGINE_TTS_OPTIONS must be a JSON object with supported synthesis options "
            "and must not set the voice"
        ) from None
    return VolcengineTTSService(api_key=api_key, resource_id=resource_id, params=params)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the voice pipeline with a managed HTTP session."""
    async with aiohttp.ClientSession(trust_env=True) as session:
        await run_bot_session(transport, runner_args, session)


def create_voice_pipeline(
    transport: BaseTransport, session: aiohttp.ClientSession
) -> tuple[Pipeline, LLMContext]:
    """Assemble the voice pipeline using backend recognition and synthesis configuration.

    Args:
        transport: Session audio transport.
        session: HTTP session used by the HTTP synthesis provider.

    Returns:
        The pipeline and its conversation context.
    """
    stt = create_stt_service()
    tts = create_tts_service(session)

    llm = DeepSeekLLMService(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        settings=DeepSeekLLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    return pipeline, context


async def run_bot_session(
    transport: BaseTransport, runner_args: RunnerArguments, session: aiohttp.ClientSession
) -> None:
    """Build and run the voice pipeline with session-scoped recognition and synthesis.

    Args:
        transport: Session audio transport.
        runner_args: Runner settings for the worker lifecycle.
        session: HTTP session used by the HTTP synthesis provider.
    """
    logger.info("Starting bot")
    pipeline, context = create_voice_pipeline(transport, session)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Start the conversation when a browser connects."""
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Stop the pipeline when the browser disconnects."""
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
