#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""Browser voice application using DeepSeek, ElevenLabs, and Volcengine."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import FunctionCallFromLLM, LLMRunFrame, TTSSpeakFrame
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
from pipecat.services.llm_service import FunctionCallHandler, FunctionCallParams, LLMService
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
    "VOLCENGINE_STT_OPTIONS",
)

VOLCENGINE_TTS_SETTINGS = (
    "VOLCENGINE_TTS_SPEAKER",
    "VOLCENGINE_TTS_OPTIONS",
)

ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"

# The provider selects the model version with its X-Api-Resource-Id header. The
# application targets one version per modality, so the values are fixed here
# rather than read from configuration.
VOLCENGINE_STT_RESOURCE_ID = "volc.seedasr.sauc.duration"
VOLCENGINE_TTS_RESOURCE_ID = "seed-tts-2.0"

# Settings that the shared API key or the fixed model versions replaced.
RETIRED_VOLCENGINE_SETTINGS = {
    "VOLCENGINE_TTS_API_KEY": "use VOLCENGINE_API_KEY",
    "VOLCENGINE_RESOURCE_ID": "the recognition model version is fixed",
    "VOLCENGINE_TTS_RESOURCE_ID": "the synthesis model version is fixed",
}


def _required_setting(config: Mapping[str, str], name: str) -> str:
    value = config.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required in backend configuration")
    return value


def _reject_retired_settings(config: Mapping[str, str]) -> None:
    for name, guidance in RETIRED_VOLCENGINE_SETTINGS.items():
        if name in config:
            raise ValueError(f"{name} is no longer used in backend configuration; {guidance}")


def create_stt_service(environ: Mapping[str, str] | None = None) -> STTService:
    """Create the session's recognition service from backend configuration.

    Args:
        environ: Environment variables, defaulting to the process environment.

    Returns:
        The configured recognition service.

    Raises:
        ValueError: A recognition setting is retired, missing, or invalid.
    """
    config = os.environ if environ is None else environ
    _reject_retired_settings(config)
    if not any(name in config for name in VOLCENGINE_STT_SETTINGS):
        return ElevenLabsRealtimeSTTService(api_key=_required_setting(config, "ELEVENLABS_API_KEY"))
    api_key = _required_setting(config, "VOLCENGINE_API_KEY")
    try:
        params = VolcengineSTTService.InputParams.model_validate_json(
            config.get("VOLCENGINE_STT_OPTIONS", "{}")
        )
    except ValidationError:
        raise ValueError(
            "VOLCENGINE_STT_OPTIONS must be a JSON object with supported recognition options; "
            "the second-pass setting is fixed"
        ) from None
    return VolcengineSTTService(
        api_key=api_key, resource_id=VOLCENGINE_STT_RESOURCE_ID, params=params
    )


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
        ValueError: A synthesis setting is retired, missing, or an option is invalid.
    """
    config = os.environ if environ is None else environ
    _reject_retired_settings(config)
    if not any(name in config for name in VOLCENGINE_TTS_SETTINGS):
        return ElevenLabsHttpTTSService(
            aiohttp_session=session,
            api_key=_required_setting(config, "ELEVENLABS_API_KEY"),
            settings=ElevenLabsHttpTTSService.Settings(
                voice=config.get("ELEVENLABS_VOICE_ID", ""),
                model=ELEVENLABS_TTS_MODEL,
            ),
        )
    api_key = _required_setting(config, "VOLCENGINE_API_KEY")
    speaker = _required_setting(config, "VOLCENGINE_TTS_SPEAKER")
    options = config.get("VOLCENGINE_TTS_OPTIONS", "{}")
    try:
        parsed = json.loads(options)
        # The voice has its own setting, so an options-supplied one is rejected
        # rather than silently overriding it.
        if not isinstance(parsed, dict) or "speaker" in parsed:
            raise TypeError
        params = VolcengineTTSService.InputParams.model_validate({"speaker": speaker, **parsed})
    except (ValidationError, json.JSONDecodeError, TypeError):
        raise ValueError(
            "VOLCENGINE_TTS_OPTIONS must be a JSON object with supported synthesis options "
            "and must not set the voice"
        ) from None
    return VolcengineTTSService(
        api_key=api_key, resource_id=VOLCENGINE_TTS_RESOURCE_ID, params=params
    )


WIKI_TOKEN_SETTING = "WIKI_TOKEN"
WIKI_API_BASE_URL_SETTING = "WIKI_API_BASE_URL"
DEFAULT_WIKI_API_BASE_URL = "http://127.0.0.1:19828"

# How a lookup reads the knowledge base: the search returns its hits ranked and
# the model answers from the full text of the top few. A hit's snippet is cut
# mid-line, so the line that answers the question is often missing from it. Each
# page is capped as it goes in, so a hit that is a whole transcript cannot fill
# the context on its own.
WIKI_SEARCH_SIZE = 8
WIKI_PAGES_PER_LOOKUP = 2
WIKI_PAGE_LIMIT = 4000


class KnowledgeBase(ABC):
    """The content a question is answered from.

    One implementation per knowledge base. The session exposes whatever it
    returns through a single tool, so adding a knowledge base means adding an
    implementation and nothing else.
    """

    @abstractmethod
    async def retrieve(self, query: str) -> str:
        """Fetch the content that answers a question.

        Args:
            query: The question to look up.

        Returns:
            The content the knowledge base holds for the query, or the empty
            string when it holds nothing.

        Raises:
            Exception: The knowledge base could not be reached. The caller
                reports that to the model as finding nothing, so an
                implementation can let its client's errors through as they are.
        """

    async def aclose(self) -> None:
        """Release what the knowledge base is holding open."""


class LLMWikiKnowledgeBase(KnowledgeBase):
    """An ``llm_wiki`` desktop application, reached over its local HTTP API."""

    def __init__(self, token: str, base_url: str = DEFAULT_WIKI_API_BASE_URL) -> None:
        """Initialize the knowledge base.

        Args:
            token: The token the application's API accepts.
            base_url: Root of the application's API.
        """
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def retrieve(self, query: str) -> str:
        """Search the wiki and return the text of the pages it ranks highest."""
        hits = await self._search(query)
        if not hits:
            # Logged apart from a failed lookup: this is what a lookup that
            # found nothing looks like, and the two leave the user in the same
            # place.
            logger.warning(
                "Knowledge base lookup found nothing; the user is told the answer could not "
                "be found"
            )
            return ""
        pages = []
        for hit in hits[:WIKI_PAGES_PER_LOOKUP]:
            path = hit.get("path")
            # The search carries each page's text, and reads one by path when it
            # does not: a hit without text would otherwise leave a lookup that
            # found something looking like one that found nothing.
            content = hit.get("content")
            if not content and path:
                try:
                    content = await self._read_page(path)
                except Exception as error:
                    # A page that cannot be read is left out rather than taking
                    # the pages already read down with it: the hits are ranked, so
                    # the ones ahead of it are the ones holding the answer.
                    logger.warning(
                        f"Knowledge base page {path} could not be read "
                        f"({type(error).__name__}); the page is left out of the lookup"
                    )
                    continue
            if isinstance(content, str) and content.strip():
                title = hit.get("title") or path
                pages.append(f"# {title} ({path})\n\n{content.strip()[:WIKI_PAGE_LIMIT]}")
        return "\n\n".join(pages)

    async def aclose(self) -> None:
        """Close the HTTP client, if a lookup opened one."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Return the ranked hits a query produces, with the text of each page."""
        payload = await self._call(
            "POST",
            "search",
            json={"query": query, "topK": WIKI_SEARCH_SIZE, "includeContent": True},
        )
        hits = payload.get("results")
        if not isinstance(hits, list):
            raise ValueError("the search response carries no results")
        return [hit for hit in hits if isinstance(hit, dict)]

    async def _read_page(self, path: str) -> str:
        """Return one page's text."""
        payload = await self._call("GET", "files/content", params={"path": path})
        content = payload.get("content")
        return content if isinstance(content, str) else ""

    async def _call(self, method: str, route: str, **kwargs) -> dict[str, Any]:
        """Send one request to the API and return its object body.

        Args:
            method: HTTP method to send.
            route: Path under the project's API, without a leading slash.
            **kwargs: Request arguments, such as a JSON body or query parameters.

        Returns:
            The body, which the API sends as an object.

        Raises:
            aiohttp.ClientError: The request failed or the API refused it.
            ValueError: The API answered with a body that is not an object.
        """
        url = f"{self._base_url}/api/v1/projects/current/{route}"
        async with self._client().request(
            method, url, headers={"Authorization": f"Bearer {self._token}"}, **kwargs
        ) as response:
            response.raise_for_status()
            payload = await response.json(content_type=None)
        if not isinstance(payload, dict):
            raise ValueError("the knowledge base answered with a body that is not an object")
        return payload

    def _client(self) -> aiohttp.ClientSession:
        """Return the HTTP client, creating it on the first lookup.

        Built without ``trust_env``: the knowledge base is reached at a loopback
        address of its own, and a proxy read from the environment would take the
        request away from it.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(trust_env=False)
        return self._session


def create_knowledge_base(environ: Mapping[str, str] | None = None) -> KnowledgeBase | None:
    """Read the session's knowledge base from backend configuration.

    Synchronous and free of I/O by design: an address a lookup could not be
    sent to fails session creation rather than a live conversation, and nothing
    here contacts the knowledge base — a session whose knowledge base is down
    starts as usual and answers as not found when it is asked to look something
    up.

    Args:
        environ: Environment variables, defaulting to the process environment.

    Returns:
        The configured knowledge base, or ``None`` when the token carries no
        value, in which case the session runs without one.

    Raises:
        ValueError: A setting the configuration declares carries no usable
            value. The message names the setting and never repeats its value.
    """
    config = os.environ if environ is None else environ
    # A token left blank is a setting left to be filled in later, so it turns
    # the knowledge base off the way an absent one does rather than failing the
    # session over a feature that is not in use.
    token = config.get(WIKI_TOKEN_SETTING, "").strip()
    if not token:
        return None
    base_url = config.get(WIKI_API_BASE_URL_SETTING, "").strip() or DEFAULT_WIKI_API_BASE_URL
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{WIKI_API_BASE_URL_SETTING} must be an absolute http or https URL")
    return LLMWikiKnowledgeBase(token=token, base_url=base_url)


# The knowledge base's only entry point, and the name a retrieval filler is
# spoken for. A constant rather than a configuration entry: with one tool it is
# what the application calls its own lookup, not a property of the knowledge base.
KNOWLEDGE_TOOL_NAME = "search_knowledge_base"

# How long one lookup may take before it is abandoned and the model is asked for
# an answer without it.
KNOWLEDGE_LOOKUP_TIMEOUT_SECS = 8.0

# The deadline the tool is registered with, which has to fall after the one
# above. Both deadlines are the same length of time, so whichever is armed first
# wins; the registration's is armed first and its cancellation is
# indistinguishable from the user interrupting, which would settle a slow lookup
# as an interrupted one rather than as finding nothing. The margin keeps the
# handler's own deadline the one that fires, without depending on the order the
# service happens to arm its timer in.
KNOWLEDGE_BACKSTOP_TIMEOUT_SECS = KNOWLEDGE_LOOKUP_TIMEOUT_SECS + 2.0

# Advertised without a handler on purpose: a handler carried here is registered
# by the service's automatic registration, which knows no timeout, so the call
# would run on the service-level default. The handler goes on with
# ``register_function`` instead, where the timeout is set.
KNOWLEDGE_TOOL = FunctionSchema(
    name=KNOWLEDGE_TOOL_NAME,
    description=(
        "Look up the answer to a question in the knowledge base. Returns the text of the most "
        "relevant pages."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The question to look up, in the user's own words.",
        }
    },
    required=["query"],
)

# Spoken while a knowledge base lookup runs, so the user hears that the assistant
# is working rather than waiting through silence. A constant rather than a
# configuration entry: it is how the assistant sounds, not which knowledge base
# it consults. Chinese, like the interface and the configured voice, even though
# the system instruction is English.
RETRIEVAL_FILLER = "我查一下。"

# What the model reads when a lookup has nothing to answer from, whether it found
# nothing or never reached the knowledge base. The instruction turns it into the
# same "could not find it", and the log tells the two apart.
NO_LOOKUP_RESULT = "No matching content was found in the knowledge base."


def _knowledge_lookup(knowledge_base: KnowledgeBase) -> FunctionCallHandler:
    """Build the handler the knowledge base tool runs.

    A lookup that fails is settled as one that found nothing: a knowledge base
    that is down and one that holds no answer leave the user in the same place,
    and the model is told the same thing either way. A lookup that runs too long
    settles the same way, from the deadline the handler keeps itself — the tool's
    own deadline cancels the call, which reads as the user interrupting it.

    Args:
        knowledge_base: Where the handler takes a lookup.

    Returns:
        The handler for the knowledge base tool.
    """

    async def lookup(params: FunctionCallParams) -> None:
        query = str(params.arguments.get("query", "")).strip()
        if not query:
            logger.warning(
                "Knowledge base lookup arrived without a question; the user is told the answer "
                "could not be found"
            )
            await params.result_callback(NO_LOOKUP_RESULT)
            return
        try:
            # The deadline is kept here rather than left to ``timeout_secs`` on
            # the registration, whose cancellation is indistinguishable from the
            # user interrupting the lookup. Cancellation from outside still
            # reaches the call as one: ``wait_for`` reports its own deadline as
            # ``TimeoutError``, and an interruption as ``CancelledError``, which
            # the ``except Exception`` below does not catch.
            content = await asyncio.wait_for(
                knowledge_base.retrieve(query), timeout=KNOWLEDGE_LOOKUP_TIMEOUT_SECS
            )
        except TimeoutError:
            logger.warning(
                f"Knowledge base lookup timed out after {KNOWLEDGE_LOOKUP_TIMEOUT_SECS} seconds; "
                "the user is told the answer could not be found"
            )
            await params.result_callback(NO_LOOKUP_RESULT)
            return
        except Exception as error:
            # The error's kind, not its text: a client can name the address it
            # was reaching for, and this line goes to the log.
            logger.warning(
                f"Knowledge base lookup failed ({type(error).__name__}); the user is told the "
                "answer could not be found"
            )
            await params.result_callback(NO_LOOKUP_RESULT)
            return
        await params.result_callback(content or NO_LOOKUP_RESULT)

    return lookup


def register_knowledge_tool(llm: LLMService, knowledge_base: KnowledgeBase) -> None:
    """Expose the knowledge base to the model as one tool.

    Args:
        llm: The service the model reaches the tool through.
        knowledge_base: Where a lookup goes.
    """
    # The registration's deadline is the backstop: the handler keeps its own
    # deadline and settles the call by it, so what is left for this one to cancel
    # is a handler that outlived its own.
    llm.register_function(
        KNOWLEDGE_TOOL_NAME,
        _knowledge_lookup(knowledge_base),
        timeout_secs=KNOWLEDGE_BACKSTOP_TIMEOUT_SECS,
    )


def announce_retrieval(llm: LLMService) -> None:
    """Speak a short line while a knowledge base lookup is in flight.

    The line goes out when the model's calls arrive and before any of them runs,
    so it covers the lookup it announces. It is spoken rather than said: the
    model never reads it back, so it cannot be taken for a turn of its own
    between a call and that call's result. Clients still show it as bot speech,
    since ``append_to_context`` governs the context alone.

    Args:
        llm: The service whose tool calls the filler is spoken for.
    """

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls: Sequence[FunctionCallFromLLM]):
        # The event fires for every tool call, so a session that grows a second
        # tool would otherwise have it announced as a lookup.
        if not any(call.function_name == KNOWLEDGE_TOOL_NAME for call in function_calls):
            return
        # Spoken from a task, so the filler does not go into the pipeline ahead
        # of the frames the service pushes for the call itself. A filler in
        # front of them carries the call's in-progress frame downstream behind
        # its audio, and a deadline that expires before it surfaces leaves the
        # assembler dropping the cancel as one for a call it never saw start,
        # which the user hears as a lookup that never answers.
        llm.create_task(
            llm.push_frame(TTSSpeakFrame(text=RETRIEVAL_FILLER, append_to_context=False))
        )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the voice pipeline with a managed HTTP session."""
    async with aiohttp.ClientSession(trust_env=True) as session:
        await run_bot_session(transport, runner_args, session)


# Appended to the base instruction, and only for sessions whose configuration
# declares a knowledge base. It states that an answer the knowledge base cannot
# supply is reported as not found — never as a failure and never from the
# model's own recall — since a lookup that fails and a lookup that comes back
# empty leave the user in the same place.
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, "
    "so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what "
    "the user said in a creative, helpful, and brief way."
)

KNOWLEDGE_BASE_INSTRUCTION = (
    "You have a knowledge base tool. Do not call it for greetings, thanks, or small talk; "
    "answer those directly. For anything else, call the tool before answering rather than "
    "relying on what you already know. Answer only from what the tool returns, in one or two "
    "spoken sentences, and say you could not find it when the result does not cover the "
    "question — never invent details the result does not contain. A lookup that fails counts as "
    "finding nothing: say you could not find it, without reporting the failure and without "
    "falling back on what you already know."
)


def create_voice_pipeline(
    transport: BaseTransport,
    session: aiohttp.ClientSession,
    knowledge_base: KnowledgeBase | None = None,
) -> tuple[Pipeline, LLMContext]:
    """Assemble the voice pipeline using backend recognition and synthesis configuration.

    Args:
        transport: Session audio transport.
        session: HTTP session used by the HTTP synthesis provider.
        knowledge_base: The knowledge base to expose to the model, ``None`` for a
            session without one.

    Returns:
        The pipeline and its conversation context.

    Raises:
        ValueError: Backend configuration is retired, missing, or invalid.
    """
    stt = create_stt_service()
    tts = create_tts_service(session)

    if knowledge_base is None:
        instruction = SYSTEM_INSTRUCTION
        tools = None
    else:
        instruction = f"{SYSTEM_INSTRUCTION} {KNOWLEDGE_BASE_INSTRUCTION}"
        tools = ToolsSchema(standard_tools=[KNOWLEDGE_TOOL])

    llm = DeepSeekLLMService(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        settings=DeepSeekLLMService.Settings(system_instruction=instruction),
    )

    if knowledge_base is not None:
        register_knowledge_tool(llm, knowledge_base)
        announce_retrieval(llm)

    # Tools hang off the context rather than taking a place in the pipeline, so
    # a session with a knowledge base has the same processors as one without.
    context = LLMContext(tools=tools) if tools is not None else LLMContext()
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
    # Read outside the pipeline, so a setting that carries no usable value fails
    # session creation rather than a live conversation. Reading opens nothing:
    # whether the knowledge base is up is settled at the first lookup.
    knowledge_base = create_knowledge_base()
    try:
        pipeline, context = create_voice_pipeline(transport, session, knowledge_base)

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
    finally:
        if knowledge_base is not None:
            await knowledge_base.aclose()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
