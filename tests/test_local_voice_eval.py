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
    if not events:
        assert result["character_accuracy"] is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["elevenlabs", "volcengine"])
async def test_shared_recording_captions_barge_in_and_cancel_over_real_eval_transport(
    provider, monkeypatch, aiohttp_server, unused_tcp_port
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
    for key in [
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "DEEPSEEK_API_KEY",
        "VOLCENGINE_API_KEY",
    ]:
        monkeypatch.setenv(key, "test-value")
    monkeypatch.setenv("STT_PROVIDER", provider)
    monkeypatch.delenv("VOLCENGINE_STT_OPTIONS", raising=False)
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
    async with serve(recognition_peer, "127.0.0.1", 0) as peer:

        async def provider_connect(url, **kwargs):
            return await connect(f"ws://127.0.0.1:{peer.sockets[0].getsockname()[1]}", **kwargs)

        monkeypatch.setattr(
            "pipecat.services.websocket_service.websocket_connect", provider_connect
        )
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
        assert options[0]["request"]["enable_nonstream"] is False


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
        "reason": "Missing VOLCENGINE_API_KEY",
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
