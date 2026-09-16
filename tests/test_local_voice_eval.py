#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise the local application's repeatable recognition evaluation."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "local-voice-app" / "eval_stt.py"


def test_report_measures_character_accuracy_and_latency_from_speech_stop():
    evaluate = runpy.run_path(str(SCRIPT))["evaluate_turn"]
    result = evaluate(
        "Hello world",
        [
            {"type": "user-transcription", "at": 1.0, "data": {"text": "Hello", "final": False}},
            {"type": "vad-user-stopped-speaking", "at": 2.0},
            {
                "type": "user-transcription",
                "at": 2.25,
                "data": {"text": "Hello word!", "final": True},
            },
        ],
    )
    assert result["character_error_rate"] == pytest.approx(0.1)
    assert result["character_accuracy"] == pytest.approx(0.9)
    assert result["final_latency_ms"] == 250
    assert result["final_segment_count"] == 1
    assert result["interim_count"] == 1
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "user-transcription", "at": 1, "data": {"text": "Hello", "final": True}}],
        [
            {"type": "user-transcription", "at": 1, "data": {"text": "Hello", "final": True}},
            {"type": "vad-user-stopped-speaking", "at": 2},
        ],
    ],
)
def test_missing_or_negative_latency_is_not_reported_as_zero(events):
    result = runpy.run_path(str(SCRIPT))["evaluate_turn"]("Hello", events)
    assert result["final_latency_ms"] is None
    assert result["latency_status"] == "unavailable"
    assert result["ttfb_ms"] is None
    if not events:
        assert result["character_accuracy"] is None


def test_ttfb_measures_llm_end_to_bot_speech():
    evaluate = runpy.run_path(str(SCRIPT))["evaluate_turn"]
    result = evaluate(
        "Hello",
        [
            {"type": "user-transcription", "at": 1, "data": {"text": "Hello", "final": True}},
            {"type": "bot-llm-stopped", "at": 2.0},
            {"type": "bot-started-speaking", "at": 2.25},
        ],
    )
    assert result["ttfb_ms"] == 250


def test_ttfb_measures_first_speech_against_the_llm_end_before_it():
    evaluate = runpy.run_path(str(SCRIPT))["evaluate_turn"]
    # Two replies in one replay: the first speech pairs with the LLM end it
    # follows, not the later one, so the measurement stays on the first reply.
    result = evaluate(
        "Hello",
        [
            {"type": "bot-llm-stopped", "at": 1.0},
            {"type": "bot-started-speaking", "at": 1.3},
            {"type": "bot-llm-stopped", "at": 5.0},
            {"type": "bot-started-speaking", "at": 5.4},
        ],
    )
    assert result["ttfb_ms"] == 300
    # Speech that precedes every LLM end is unmeasured, not clamped to zero.
    earlier = evaluate(
        "Hello",
        [
            {"type": "bot-started-speaking", "at": 1.0},
            {"type": "bot-llm-stopped", "at": 2.0},
        ],
    )
    assert earlier["ttfb_ms"] is None


def test_repeated_finals_remain_visible_in_accuracy_without_text_deduplication():
    events = [
        {"type": "user-transcription", "at": 1, "data": {"text": "Hi", "final": False}},
        {"type": "user-transcription", "at": 2, "data": {"text": "Hi", "final": True}},
        {"type": "user-transcription", "at": 3, "data": {"text": "Hi", "final": True}},
    ]
    result = runpy.run_path(str(SCRIPT))["evaluate_turn"]("Hi", events)
    assert result["exact_match"] is False
    assert result["final_segment_count"] == 2
    assert result["character_error_rate"] == 1


def test_application_supports_eval_audio_transport():
    from pipecat.evals.transport import EvalTransportParams

    app = runpy.run_path(str(ROOT / "local-voice-app" / "bot.py"))
    params = app["transport_params"]["eval"]()
    assert isinstance(params, EvalTransportParams)
    assert params.audio_in_enabled and params.audio_out_enabled


def test_provider_environment_isolated_without_mutating_backend_configuration():
    build_environment = runpy.run_path(str(SCRIPT))["_provider_environment"]
    config = {
        "ELEVENLABS_API_KEY": "eleven-key",
        "VOLCENGINE_API_KEY": "volc-key",
        "VOLCENGINE_STT_OPTIONS": '{"enable_punc":true}',
        "VOLCENGINE_TTS_SPEAKER": "volc-speaker",
        # The application rejects retired settings, so no replayed process may
        # inherit one.
        "VOLCENGINE_TTS_API_KEY": "retired-key",
    }

    elevenlabs_environment = build_environment("elevenlabs", config)
    volcengine_environment = build_environment("volcengine", config)

    assert "VOLCENGINE_API_KEY" not in elevenlabs_environment
    assert "VOLCENGINE_STT_OPTIONS" not in elevenlabs_environment
    assert "VOLCENGINE_TTS_SPEAKER" not in elevenlabs_environment
    assert "VOLCENGINE_TTS_API_KEY" not in volcengine_environment
    assert volcengine_environment == {
        "ELEVENLABS_API_KEY": "eleven-key",
        "VOLCENGINE_API_KEY": "volc-key",
        "VOLCENGINE_STT_OPTIONS": '{"enable_punc":true}',
        "VOLCENGINE_TTS_SPEAKER": "volc-speaker",
    }
    assert config["VOLCENGINE_API_KEY"] == "volc-key"


def _audio_only_server(audio: bytes, *, final: bool = False) -> bytes:
    """Build a Volcengine-style AudioOnlyServer frame carrying raw PCM.

    A final frame carries a negative sequence field and the final flag, which the
    service reads as the end of a synthesized sentence.
    """
    import struct

    flags = 0x2 if final else 0x0
    sequence = struct.pack(">i", -1) if final else b""
    return (
        bytes((0x11, 0xB0 | flags, 0x00, 0x00)) + sequence + struct.pack(">I", len(audio)) + audio
    )


def _decode_client_event(message: bytes) -> tuple[int, str, dict]:
    """Decode a full Volcengine client event into ``(event, session_id, payload)``.

    Session-scoped events carry a length-prefixed session identifier after the
    event identifier; connection-scoped ones do not.
    """
    import json
    import struct

    assert message[:4] == b"\x11\x14\x10\x00"
    offset = 4
    event = struct.unpack_from(">i", message, offset)[0]
    offset += 4
    session_id = ""
    if event not in (1, 2):
        length = struct.unpack_from(">I", message, offset)[0]
        offset += 4
        session_id = message[offset : offset + length].decode()
        offset += length
    size = struct.unpack_from(">I", message, offset)[0]
    offset += 4
    assert len(message) == offset + size
    return event, session_id, json.loads(message[offset:])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, tts_provider",
    [
        ("elevenlabs", "elevenlabs"),
        # Volcengine synthesis needs the shared API key, and that key also selects
        # Volcengine recognition.
        ("volcengine", "elevenlabs"),
        ("volcengine", "volcengine"),
    ],
)
async def test_shared_recording_captions_barge_in_and_cancel_over_real_eval_transport(
    provider, tts_provider, monkeypatch, aiohttp_server, unused_tcp_port
):
    import asyncio
    import base64
    import gzip
    import json
    import struct

    import aiohttp
    import numpy as np
    from aiohttp import web
    from openai.types.chat import ChatCompletionChunk
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from pipecat.evals.serializer import EvalSerializer
    from pipecat.evals.transport import EvalTransport, EvalTransportParams
    from pipecat.runner.types import RunnerArguments

    reference = "What is the capital of Germany?"
    commits = []
    options = []
    peer_closed = asyncio.Event()
    tts_requests = []
    tts_events = []
    tts_connected = asyncio.Event()

    async def tts_peer(websocket):
        # The service writes its protocol as a stream of sends, one frame at a
        # time: StartConnection, StartSession, then TaskRequest per sentence, and
        # FinishSession at the end of a turn. Each request is answered with final
        # audio so the bot actually speaks; the events are recorded so the test can
        # assert the session lifecycle the pipeline drove.
        try:
            await websocket.recv()  # StartConnection
            tts_connected.set()
            while True:
                event, _, payload = _decode_client_event(await websocket.recv())
                tts_events.append(event)
                if event == 200:
                    tts_requests.append(payload["req_params"]["text"])
                    await websocket.send(_audio_only_server(b"\x01\x00" * 4800, final=True))
        finally:
            tts_connected.set()

    async def recognition_peer(websocket):
        active = False
        quiet_samples = 0
        sample_count = 0
        start_sample = 0
        if provider == "volcengine":
            options.append(json.loads(gzip.decompress((await websocket.recv())[12:])))
        else:
            await websocket.send('{"message_type":"session_started"}')

        async def transcript(final):
            text = reference if final else "What is the capital"
            if final and provider == "elevenlabs":
                text = ["What is the capital", "of Germany?"][len(commits) % 2]
            if provider == "elevenlabs":
                paired = (
                    "include_language_detection=true" in websocket.request.path
                    or "include_timestamps=true" in websocket.request.path
                )
                if final and paired:
                    # The API sends these as a pair when language detection is enabled.
                    await websocket.send(
                        json.dumps({"message_type": "committed_transcript", "text": text})
                    )
                await websocket.send(
                    json.dumps(
                        {
                            "message_type": (
                                "committed_transcript_with_timestamps"
                                if paired
                                else "committed_transcript"
                            )
                            if final
                            else "partial_transcript",
                            "text": text,
                            "language_code": "en",
                        }
                    )
                )
            else:
                body = json.dumps(
                    {
                        "result": {
                            "utterances": [
                                {
                                    "text": text,
                                    "definite": final,
                                    "start_time": start_sample // 16,
                                    "end_time": sample_count // 16,
                                }
                            ]
                        }
                    }
                ).encode()
                packet = b"\x11\x91\x10\x00" + struct.pack(">iI", 1, len(body)) + body
                await websocket.send(packet)
                if final:
                    await websocket.send(packet)
            if final:
                commits.append(text)

        try:
            async for message in websocket:
                if provider == "elevenlabs":
                    data = json.loads(message)
                    if data.get("commit"):
                        if active:
                            await transcript(True)
                            active = False
                        continue
                    audio = base64.b64decode(data["audio_base_64"])
                else:
                    if struct.unpack(">i", message[4:8])[0] < 0:
                        body = b'{"result":{}}'
                        await websocket.send(
                            b"\x11\x93\x10\x00" + struct.pack(">iI", -1, len(body)) + body
                        )
                        continue
                    audio = gzip.decompress(message[12:])
                samples = np.frombuffer(audio, dtype=np.int16)
                sample_count += len(samples)
                speaking = bool(len(samples) and np.max(np.abs(samples.astype(np.int32))) > 300)
                if speaking:
                    quiet_samples = 0
                    if not active:
                        active = True
                        start_sample = sample_count - len(samples)
                        await transcript(False)
                elif active:
                    quiet_samples += len(samples)
                    if provider == "volcengine" and quiet_samples >= 16000 * 0.5:
                        await transcript(True)
                        active = False
        finally:
            peer_closed.set()

    async def llm_response(client, **kwargs):
        async def chunks():
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "test-completion",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Here is a spoken response."},
                            "finish_reason": None,
                        }
                    ],
                }
            )

        return chunks()

    async def tts_response(request):
        payload = await request.json()
        assert payload["text"]
        chunk = {"audio_base64": base64.b64encode(b"\x01\x00" * 4800).decode()}
        return web.Response(text=(json.dumps(chunk) + "\n") * 20)

    http_app = web.Application()
    http_app.router.add_post("/tts", tts_response)
    http_server = await aiohttp_server(http_app)
    original_post = aiohttp.ClientSession.post

    def post(session, url, **kwargs):
        assert str(url).startswith("https://api.elevenlabs.io/")
        return original_post(session, http_server.make_url("/tts"), **kwargs)

    monkeypatch.setattr("openai.resources.chat.completions.AsyncCompletions.create", llm_response)
    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    for key in ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "DEEPSEEK_API_KEY"]:
        monkeypatch.setenv(key, "test-value")
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
        monkeypatch.setenv("VOLCENGINE_API_KEY", "test-value")
    if tts_provider == "volcengine":
        monkeypatch.setenv("VOLCENGINE_TTS_SPEAKER", "test-speaker")
    runner_args = RunnerArguments()
    runner_args.handle_sigint = False
    app = runpy.run_path(str(ROOT / "local-voice-app" / "bot.py"))
    replay = runpy.run_path(str(SCRIPT))["replay_recording"]
    transport = EvalTransport(
        host="127.0.0.1",
        port=unused_tcp_port,
        params=EvalTransportParams(
            audio_in_enabled=True, audio_out_enabled=True, serializer=EvalSerializer()
        ),
    )

    async def provider_connect(url, **kwargs):
        # Both providers open their realtime socket here, so the peer is chosen by
        # the endpoint: synthesis for the TTS host, recognition otherwise.
        endpoint = str(url)
        assert endpoint.startswith(("wss://openspeech.bytedance.com/", "wss://api.elevenlabs.io/"))
        peer = tts_peer_server if "tts/bidirection" in endpoint else recognition_peer_server
        return await connect(f"ws://127.0.0.1:{peer.sockets[0].getsockname()[1]}", **kwargs)

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", provider_connect)
    async with (
        serve(recognition_peer, "127.0.0.1", 0) as recognition_peer_server,
        serve(tts_peer, "127.0.0.1", 0) as tts_peer_server,
    ):
        async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as group:
            bot = group.create_task(app["run_bot_session"](transport, runner_args, session))
            try:
                for _ in range(100):
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
                        writer.close()
                        await writer.wait_closed()
                        break
                    except OSError:
                        await asyncio.sleep(0.05)
                result = await replay(
                    f"ws://127.0.0.1:{unused_tcp_port}",
                    ROOT / "scripts/release-evals/assets/capital_question.wav",
                    reference,
                    timeout=10,
                    silence=2,
                )
                await asyncio.wait_for(bot, 5)
            finally:
                bot.cancel()
    assert result["status"] == "passed", result
    assert [turn["transcript"] for turn in result["turns"]] == [reference, reference]
    assert [turn["final_segment_count"] for turn in result["turns"]] == (
        [2, 2] if provider == "elevenlabs" else [1, 1]
    )
    assert commits == (
        ["What is the capital", "of Germany?"] * 2
        if provider == "elevenlabs"
        else [reference, reference]
    )
    assert peer_closed.is_set()
    if provider == "volcengine":
        assert options[0]["request"]["enable_nonstream"] is True
    # The barge-in interrupted speech the bot had actually started, and the bot
    # resumed for the second replay and tore down cleanly.
    assert result["checks"]["barge_in_started_during_bot_speech"] is True
    assert result["checks"]["bot_interrupted"] is True
    if tts_provider == "volcengine":
        assert tts_connected.is_set()
        # The bot synthesized each turn's reply through Volcengine: the peer saw
        # the LLM's text on every TaskRequest it was sent.
        assert set(tts_requests) == {"Here is a spoken response."}
        # Every session the service opened it also finished, so no session is left
        # dangling at teardown.
        assert tts_events.count(100) > 0  # StartSession
        assert tts_events.count(102) == tts_events.count(100)  # FinishSession


def test_final_segments_are_combined_and_complete_text_uses_last_arrival():
    events = [
        {"type": "user-transcription", "at": 1, "data": {"text": "Hello", "final": False}},
        {"type": "user-transcription", "at": 1.5, "data": {"text": "Hello", "final": True}},
        {"type": "vad-user-stopped-speaking", "at": 2},
        {"type": "user-transcription", "at": 2.5, "data": {"text": "world", "final": True}},
    ]
    result = runpy.run_path(str(SCRIPT))["evaluate_turn"]("Hello world", events)
    assert result["exact_match"] is True
    assert result["final_segment_count"] == 2
    assert result["character_accuracy"] == 1
    assert result["final_latency_ms"] == 500


@pytest.mark.asyncio
async def test_missing_credentials_skip_live_execution_without_exposing_values():
    import argparse

    run_provider = runpy.run_path(str(SCRIPT))["run_provider"]
    result = await run_provider(
        "volcengine",
        argparse.Namespace(),
        {
            "ELEVENLABS_API_KEY": "secret-key",
            "ELEVENLABS_VOICE_ID": "test-voice",
            "DEEPSEEK_API_KEY": "secret-key",
        },
    )
    assert result == {
        "provider": "volcengine",
        "status": "skipped",
        "reason": "Missing VOLCENGINE_API_KEY, VOLCENGINE_TTS_SPEAKER",
    }


@pytest.mark.asyncio
async def test_missing_bot_speech_fails_instead_of_claiming_interruption():
    import json

    from websockets.asyncio.server import serve

    async def silent_bot(websocket):
        async for payload in websocket:
            message = json.loads(payload)
            if message["type"] == "client-ready":
                await websocket.send('{"label":"rtvi-ai","type":"bot-ready","data":{}}')
            if message.get("data", {}).get("t") == "eval-cancel":
                await websocket.close()

    replay = runpy.run_path(str(SCRIPT))["replay_recording"]
    async with serve(silent_bot, "127.0.0.1", 0) as server:
        result = await replay(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            ROOT / "scripts/release-evals/assets/capital_question.wav",
            "What is the capital of Germany?",
            timeout=0.1,
        )
    assert result["status"] == "failed"
    assert result["reason"] == "Timed out waiting for bot-started-speaking"
    assert result["checks"]["required_events_within_timeout"] is False
    assert result["checks"]["eval_cancel_closed_connection"] is True
    assert result["turns"] == []


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1"])
def test_cli_rejects_unbounded_timeouts(timeout, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--timeout", timeout])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT))["main"]()
    assert error.value.code == 2
    assert "finite and positive" in capsys.readouterr().err
