#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#

"""Exercise Volcengine TTS against a local binary WebSocket peer."""

import asyncio
import base64
import json
import struct
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.volcengine.tts import VolcengineTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.runner import WorkerRunner


def _decode_client_event(message: bytes) -> tuple[int, str, dict]:
    """Decode a full client event frame into ``(event, session_id, payload)``.

    Session-scoped events carry a length-prefixed session identifier in the frame
    header after the event identifier; connection-scoped events do not.
    """
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


def _audio_only_server(audio: bytes, *, final: bool = False) -> bytes:
    """Build an AudioOnlyServer frame carrying raw PCM.

    A final frame carries a negative sequence field and the final flag, matching
    the provider's server-frame parser.
    """
    flags = 0x2 if final else 0x0
    sequence = struct.pack(">i", -1) if final else b""
    return (
        bytes((0x11, 0xB0 | flags, 0x00, 0x00)) + sequence + struct.pack(">I", len(audio)) + audio
    )


def _response(data: bytes, *, final: bool = False) -> bytes:
    """Build a Response (event 352) frame with base64-encoded JSON audio.

    The frame carries the event identifier, a session identifier, and, when
    final, a negative sequence field before the length-prefixed payload.
    """
    flags = 0x4 | (0x2 if final else 0x0)
    sequence = struct.pack(">i", -1) if final else b""
    session_id = b"session-1"
    payload = json.dumps({"data": base64.b64encode(data).decode()}).encode()
    return (
        bytes((0x11, 0x90 | flags, 0x10, 0x00))
        + sequence
        + struct.pack(">i", 352)
        + struct.pack(">I", len(session_id))
        + session_id
        + struct.pack(">I", len(payload))
        + payload
    )


def _error(code: int, reason: str) -> bytes:
    """Build an Error frame."""
    body = reason.encode()
    return bytes((0x11, 0xF0, 0x00, 0x00)) + struct.pack(">II", code, len(body)) + body


class _ResultObserver(BaseObserver):
    def __init__(self, service, predicate):
        super().__init__()
        self.service = service
        self.predicate = predicate
        self.received = asyncio.Event()
        self.frames = []

    async def on_push_frame(self, data):
        if data.source is self.service and self.predicate(data.frame):
            self.frames.append(data.frame)
            self.received.set()


async def _speak_and_observe(service, predicate, text="Hello there."):
    """Run one TTS turn and return every frame satisfying ``predicate``."""
    observer = _ResultObserver(service, predicate)
    worker = PipelineWorker(Pipeline([service]), enable_rtvi=False, observers=[observer])
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await worker.queue_frame(TTSSpeakFrame(text))

    async def finish():
        try:
            await asyncio.wait_for(observer.received.wait(), 5)
        finally:
            await worker.cancel()

    await asyncio.gather(runner.run(), finish())
    return observer.frames


@pytest.mark.asyncio
async def test_stream_authenticates_and_sends_handshake_events():
    sessions = []
    handshake_done = asyncio.Event()

    async def handler(websocket):
        headers = websocket.request.headers
        start_conn = _decode_client_event(await websocket.recv())
        # The session is now lazy: the first synthesis request opens it.
        start_session = _decode_client_event(await websocket.recv())
        await websocket.recv()  # TaskRequest
        sessions.append((headers, start_conn, start_session))
        handshake_done.set()
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineTTSService(
            api_key="test-secret",
            resource_id="seed-tts-2.0",
            ws_url=f"ws://127.0.0.1:{port}",
            speaker="speaker-1",
            sample_rate=24000,
        )
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(TTSSpeakFrame("Hello there."))

        async def finish():
            try:
                await asyncio.wait_for(handshake_done.wait(), 5)
            finally:
                await worker.cancel()

        await asyncio.gather(runner.run(), finish())

    headers, start_conn, start_session = sessions[0]
    assert headers["X-Api-Key"] == "test-secret"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"

    conn_event, conn_session_id, conn_payload = start_conn
    assert conn_event == 1
    assert conn_session_id == ""
    assert conn_payload == {"namespace": "BidirectionalTTS"}

    session_event, session_id, session_payload = start_session
    assert session_event == 100
    assert len(session_id) == 12
    assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in session_id)
    assert session_payload["event"] == 100
    req_params = session_payload["req_params"]
    assert req_params["speaker"] == "speaker-1"
    assert req_params["audio_params"] == {"format": "pcm", "sample_rate": 24000}


@pytest.mark.asyncio
async def test_task_request_carries_text():
    requests = []
    request_done = asyncio.Event()

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        requests.append(_decode_client_event(await websocket.recv()))
        request_done.set()
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineTTSService(
            api_key="test-key",
            resource_id="seed-tts-2.0",
            ws_url=f"ws://127.0.0.1:{port}",
            speaker="speaker-1",
        )
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(TTSSpeakFrame("Hello there."))

        async def finish():
            try:
                await asyncio.wait_for(request_done.wait(), 5)
            finally:
                await worker.cancel()

        await asyncio.gather(runner.run(), finish())

    event, session_id, payload = requests[0]
    assert event == 200
    assert session_id != ""
    assert payload["event"] == 200
    assert payload["req_params"]["text"] == "Hello there."


@pytest.mark.asyncio
async def test_audio_only_server_yields_audio_and_stopped_frames():
    audio = b"\x01\x00\x02\x00"

    async def handler(websocket):
        await websocket.recv()  # StartConnection
        await websocket.recv()  # StartSession
        await websocket.recv()  # TaskRequest
        await websocket.send(_audio_only_server(audio, final=True))
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineTTSService(
            api_key="test-key",
            resource_id="seed-tts-2.0",
            ws_url=f"ws://127.0.0.1:{port}",
            speaker="speaker-1",
            sample_rate=24000,
        )
        frames = await _speak_and_observe(
            service, lambda f: isinstance(f, (TTSAudioRawFrame, TTSStoppedFrame))
        )

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    stopped_frames = [f for f in frames if isinstance(f, TTSStoppedFrame)]
    assert len(audio_frames) == 1
    assert audio_frames[0].audio == audio
    assert audio_frames[0].sample_rate == 24000
    assert audio_frames[0].num_channels == 1
    assert len(stopped_frames) == 1


@pytest.mark.asyncio
async def test_response_event_decodes_base64_audio():
    audio = b"\x03\x00\x04\x00"

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.recv()
        await websocket.send(_response(audio, final=True))
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineTTSService(
            api_key="test-key",
            resource_id="seed-tts-2.0",
            ws_url=f"ws://127.0.0.1:{port}",
            speaker="speaker-1",
            sample_rate=24000,
        )
        frames = await _speak_and_observe(service, lambda f: isinstance(f, TTSAudioRawFrame))

    assert len(frames) == 1
    assert frames[0].audio == audio


@pytest.mark.asyncio
async def test_error_frame_emits_nonfatal_error():
    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.send(_error(45000001, "synthesis failed"))
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineTTSService(
            api_key="test-key",
            resource_id="seed-tts-2.0",
            ws_url=f"ws://127.0.0.1:{port}",
            speaker="speaker-1",
            reconnect_on_error=False,
        )
        frames = await _speak_and_observe(service, lambda f: isinstance(f, ErrorFrame))

    assert len(frames) == 1
    assert "45000001" in frames[0].error
    assert frames[0].fatal is False


def _make_service(port, **kwargs):
    return VolcengineTTSService(
        api_key="test-key",
        resource_id="seed-tts-2.0",
        ws_url=f"ws://127.0.0.1:{port}",
        speaker="speaker-1",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_interruption_sends_cancel_session_and_reuses_connection():
    """Interrupting cancels the active session in place without closing the socket."""
    service = VolcengineTTSService(
        api_key="test-key", resource_id="seed-tts-2.0", speaker="speaker-1"
    )
    websocket = AsyncMock()
    websocket.state = State.OPEN
    service._websocket = websocket
    service._session_id = "abcd1234efgh"
    service._session_context_id = "turn-17"

    await service.on_audio_context_interrupted("turn-17")

    assert websocket.send.call_count == 1
    event, session_id, payload = _decode_client_event(websocket.send.call_args.args[0])
    assert event == 101
    assert session_id == "abcd1234efgh"
    assert payload == {}
    assert not websocket.close.called


@pytest.mark.asyncio
async def test_flush_audio_sends_finish_session():
    """Finishing a turn sends FinishSession with an empty payload."""
    service = VolcengineTTSService(
        api_key="test-key", resource_id="seed-tts-2.0", speaker="speaker-1"
    )
    websocket = AsyncMock()
    websocket.state = State.OPEN
    service._websocket = websocket
    service._session_id = "abcd1234efgh"
    service._session_context_id = "turn-17"

    await service.flush_audio("turn-17")

    event, session_id, payload = _decode_client_event(websocket.send.call_args.args[0])
    assert event == 102
    assert session_id == "abcd1234efgh"
    assert payload == {}
    assert service._session_id == ""


async def _collect_events_until_close(websocket, events):
    """Read every client event until the connection closes, ignoring close frames."""
    try:
        while True:
            events.append(_decode_client_event(await websocket.recv()))
    except Exception:
        return


@pytest.mark.asyncio
async def test_stop_finishes_session_and_connection():
    events = []

    async def handler(websocket):
        await _collect_events_until_close(websocket, events)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = _make_service(port)
        await run_test(
            service,
            frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(sleep=0.2)],
        )

    event_ids = [e[0] for e in events]
    assert 102 in event_ids  # FinishSession
    assert 2 in event_ids  # FinishConnection


@pytest.mark.asyncio
async def test_stop_times_out_gracefully_without_session_finished():
    async def handler(websocket):
        await _collect_events_until_close(websocket, [])
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = _make_service(port, flush_timeout=0.2)
        observer = _ResultObserver(service, lambda f: isinstance(f, ErrorFrame))
        await run_test(
            service,
            frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(sleep=0.2)],
            observers=[observer],
        )

    assert any("timed out waiting for SessionFinished" in f.error for f in observer.frames)


@pytest.mark.asyncio
async def test_new_turn_starts_fresh_session():
    events = []

    async def handler(websocket):
        await _collect_events_until_close(websocket, events)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = _make_service(port)
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(TTSSpeakFrame("First."))
        await worker.queue_frame(TTSSpeakFrame("Second."))

        async def finish():
            await asyncio.sleep(0.3)
            await worker.cancel()

        await asyncio.gather(runner.run(), finish())

    start_sessions = [e for e in events if e[0] == 100]
    assert len(start_sessions) == 2
    assert start_sessions[0][1] != start_sessions[1][1]
    # FinishSession follows each turn's synthesis.
    assert events[0][0] == 1  # StartConnection first
    assert any(e[0] == 102 for e in events)


@pytest.mark.asyncio
async def test_cancel_releases_websocket_and_receive_task():
    async def handler(websocket):
        await websocket.recv()  # StartConnection
        await websocket.recv()  # StartSession
        await websocket.recv()  # TaskRequest
        await websocket.send(_audio_only_server(b"\x01\x00\x02\x00", final=True))
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = _make_service(port)
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(TTSSpeakFrame("Hello there."))

        async def finish():
            await asyncio.sleep(0.3)
            await worker.cancel()

        await asyncio.gather(runner.run(), finish())

    assert service._websocket is None
    assert service._receive_task is None
