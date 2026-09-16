#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise Volcengine STT against a local binary WebSocket peer."""

import asyncio
import gzip
import json
import struct
from uuid import UUID

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.volcengine.stt import VolcengineSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.runner import WorkerRunner


def _response(payload, *, sequence=1, compressed=True):
    body = json.dumps(payload).encode()
    if compressed:
        body = gzip.compress(body)
    header = bytes((0x11, 0x93 if sequence < 0 else 0x91, 0x11 if compressed else 0x10, 0))
    return header + struct.pack(">iI", sequence, len(body)) + body


async def _acknowledge_end(websocket):
    async for message in websocket:
        if struct.unpack(">i", message[4:8])[0] < 0:
            await websocket.send(_response({"result": {}}, sequence=-1))


class _ResultObserver(BaseObserver):
    def __init__(self, service, count):
        super().__init__()
        self.service = service
        self.count = count
        self.received = asyncio.Event()
        self.frames = []

    async def on_push_frame(self, data):
        if data.source is self.service and isinstance(
            data.frame, (InterimTranscriptionFrame, TranscriptionFrame, ErrorFrame)
        ):
            self.frames.append(data.frame)
            self.count -= 1
            if self.count == 0:
                self.received.set()


async def _run_until_results(service, count):
    observer = _ResultObserver(service, count)
    worker = PipelineWorker(Pipeline([service]), enable_rtvi=False, observers=[observer])
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await worker.queue_frame(InputAudioRawFrame(b"\0\0" * 160, 16000, 1))

    async def finish():
        try:
            await asyncio.wait_for(observer.received.wait(), 5)
        finally:
            await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), finish())
    return observer.frames


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [True, False])
async def test_partial_and_definite_results_become_standard_frames(compressed):
    utterance = {
        "text": "Hello there.",
        "definite": True,
        "start_time": 0,
        "end_time": 900,
        "speaker": "speaker-1",
    }

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.send(_response({"result": {"text": "Hello"}}, compressed=compressed))
        await websocket.send(
            _response(
                {"result": {"text": "Hello there.", "utterances": [utterance]}},
                sequence=2,
                compressed=compressed,
            )
        )
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{port}",
            audio_passthrough=False,
        )
        transcripts = await _run_until_results(service, 2)

    assert [(type(f), f.text) for f in transcripts] == [
        (InterimTranscriptionFrame, "Hello"),
        (TranscriptionFrame, "Hello there."),
    ]
    assert transcripts[1].result["result"]["utterances"] == [utterance]
    assert transcripts[1].finalized is False


@pytest.mark.asyncio
async def test_request_options_are_serialized_by_protocol_model():
    requests = []

    async def handler(websocket):
        requests.append(json.loads(gzip.decompress((await websocket.recv())[12:])))
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            params=VolcengineSTTService.InputParams(
                enable_itn=True,
                enable_punc=False,
                corpus_context={"hotwords": ["Pipecat"]},
            ),
        )
        await run_test(service, frames_to_send=[], start_timeout=5)

    assert len(requests) == 1
    request = requests[0]
    UUID(request["user"]["uid"])
    assert request["audio"] == {
        "format": "pcm",
        "codec": "raw",
        "rate": 16000,
        "bits": 16,
        "channel": 1,
    }
    assert request["request"] == {
        "model_name": "bigmodel",
        "result_type": "single",
        "show_utterances": True,
        "enable_nonstream": True,
        "enable_itn": True,
        "enable_punc": False,
        "corpus": {"context": '{"hotwords": ["Pipecat"]}'},
    }


@pytest.mark.asyncio
async def test_stream_authenticates_and_sends_sequenced_pcm_with_end_marker():
    sessions = []

    async def handler(websocket):
        messages = []
        sessions.append((websocket.request.headers, messages))
        async for message in websocket:
            messages.append(message)
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {}}, sequence=-1))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        for _ in range(2):
            service = VolcengineSTTService(
                api_key="test-secret",
                resource_id="test-resource",
                ws_url=f"ws://127.0.0.1:{port}",
                sample_rate=16000,
                audio_passthrough=False,
            )
            await run_test(
                service,
                frames_to_send=[
                    InputAudioRawFrame(b"\x01\x00" * 160, 16000, 1),
                    InputAudioRawFrame(b"\x02\x00" * 160, 16000, 1),
                ],
                expected_down_frames=[STTMetadataFrame],
                start_timeout=5,
            )

    request_ids = []
    for headers, messages in sessions:
        assert headers["X-Api-Key"] == "test-secret"
        assert headers["X-Api-Resource-Id"] == "test-resource"
        request_ids.append(str(UUID(headers["X-Api-Request-Id"])))
        assert len(messages) == 4
        assert [message[:4] for message in messages] == [
            b"\x11\x11\x11\x00",
            b"\x11\x21\x01\x00",
            b"\x11\x21\x01\x00",
            b"\x11\x23\x01\x00",
        ]
        assert [struct.unpack(">i", message[4:8])[0] for message in messages] == [1, 2, 3, -4]
        for message in messages:
            assert struct.unpack(">I", message[8:12])[0] == len(message[12:])
        request = json.loads(gzip.decompress(messages[0][12:]))
        assert request["audio"] == {
            "format": "pcm",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
        }
        assert request["request"]["result_type"] == "single"
        assert request["request"]["show_utterances"] is True
        assert request["request"]["enable_nonstream"] is True
        assert [gzip.decompress(message[12:]) for message in messages[1:]] == [
            b"\x01\x00" * 160,
            b"\x02\x00" * 160,
            b"",
        ]
    assert len(set(request_ids)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, expected",
    [
        (b"\x11\xf0\x10\x00" + struct.pack(">II", 45000001, 13) + b"invalid audio", "45000001"),
        (b"\x11\x91", "truncated"),
        (b"\x11\x90\x10\x00\x00\x00\x00\x08{}", "payload size"),
        (b"\x11\x90\x11\x00\x00\x00\x00\x02{}", "gzip"),
        (
            b"\x11\x90\x11\x00\x00\x00\x00\x13"
            + bytes.fromhex("1f8b0800000000000003070000000000000000"),
            "gzip",
        ),
        (b"\x11\x90\x10\x00\x00\x00\x00\x01{", "JSON"),
        ("unexpected text", "binary"),
        (None, "1011"),
    ],
)
async def test_protocol_failures_emit_nonfatal_error_with_request_id(message, expected):
    request_ids = []

    async def handler(websocket):
        request_ids.append(websocket.request.headers["X-Api-Request-Id"])
        await websocket.recv()
        await websocket.recv()
        if message is None:
            await websocket.close(code=1011, reason="test failure")
        else:
            await websocket.send(message)
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{port}",
            reconnect_on_error=False,
        )
        errors = await _run_until_results(service, 1)

    assert len(errors) == 1
    assert expected in errors[0].error
    assert request_ids[0] in errors[0].error
    assert errors[0].fatal is False


@pytest.mark.asyncio
async def test_large_audio_frame_is_sent_as_bounded_pcm_packets():
    packets = []

    async def handler(websocket):
        await websocket.recv()
        async for message in websocket:
            packets.append(gzip.decompress(message[12:]))
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {}}, sequence=-1))

    audio = b"\x01\x00" * 16000
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{port}",
            sample_rate=16000,
        )
        await run_test(
            service,
            frames_to_send=[InputAudioRawFrame(audio, 16000, 1)],
            start_timeout=5,
        )

    assert b"".join(packets) == audio
    assert all(len(packet) <= 3200 for packet in packets)
    assert packets[-1] == b""


@pytest.mark.asyncio
async def test_definite_segments_deduplicate_by_position_and_preserve_repeated_text():
    first = {"text": "Okay.", "definite": True, "start_time": 0, "end_time": 500}
    second = {"text": "Okay.", "definite": True, "start_time": 800, "end_time": 1300}

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        for utterances in ([first], [first, second], [first, second]):
            await websocket.send(_response({"result": {"utterances": utterances}}))
        await websocket.send(_response({"result": {"text": "Done"}}))
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        frames = await _run_until_results(service, 3)

    assert [(type(frame), frame.text) for frame in frames] == [
        (TranscriptionFrame, "Okay."),
        (TranscriptionFrame, "Okay."),
        (InterimTranscriptionFrame, "Done"),
    ]
    assert all(not frame.finalized for frame in frames if isinstance(frame, TranscriptionFrame))


@pytest.mark.asyncio
async def test_normal_end_waits_for_final_marker_and_delivers_tail_transcript():
    async def handler(websocket):
        async for message in websocket:
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {"text": "Last"}}))
                await asyncio.sleep(0.05)
                await websocket.send(
                    _response(
                        {
                            "result": {
                                "utterances": [
                                    {
                                        "text": "Last words.",
                                        "definite": True,
                                        "start_time": 0,
                                        "end_time": 900,
                                    }
                                ]
                            }
                        },
                        sequence=-1,
                    )
                )

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            audio_passthrough=False,
        )
        down, _ = await run_test(
            service,
            frames_to_send=[InputAudioRawFrame(b"\0\0" * 160, 16000, 1)],
            expected_down_frames=[STTMetadataFrame, InterimTranscriptionFrame, TranscriptionFrame],
            start_timeout=5,
        )
    assert down[-1].text == "Last words."
    assert down[-1].finalized is False


@pytest.mark.asyncio
async def test_normal_end_times_out_when_peer_never_marks_response_final():
    closed = asyncio.Event()

    async def handler(websocket):
        async for message in websocket:
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {}}))
        closed.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            flush_timeout=0.05,
        )
        async with asyncio.timeout(5):
            _, errors = await run_test(
                service,
                frames_to_send=[],
                expected_up_frames=[STTMetadataFrame, ErrorFrame],
                start_timeout=5,
            )
            await closed.wait()
    assert "final response timed out" in errors[-1].error
    assert errors[-1].fatal is False


@pytest.mark.asyncio
async def test_reconnect_resets_stream_and_discards_audio_received_during_outage():
    sessions = []
    reconnecting = asyncio.Event()
    allow_reconnect = asyncio.Event()
    stale_forwarded = asyncio.Event()
    transcripts_received = asyncio.Event()
    transcripts = []
    errors = []
    stale_audio = b"\x02\x00" * 160
    fresh_audio = b"\x03\x00" * 160

    async def process_request(connection, request):
        if sessions:
            reconnecting.set()
            await allow_reconnect.wait()

    async def handler(websocket):
        packets = []
        sessions.append((websocket.request.headers["X-Api-Request-Id"], packets))
        packets.append(await websocket.recv())
        packets.append(await websocket.recv())
        await websocket.send(
            _response(
                {
                    "result": {
                        "utterances": [
                            {"text": "Okay.", "definite": True, "start_time": 0, "end_time": 500}
                        ]
                    }
                }
            )
        )
        if len(sessions) == 1:
            await websocket.close(code=1011, reason="temporary outage")
        else:
            await _acknowledge_end(websocket)

    class Observer(BaseObserver):
        async def on_push_frame(self, data):
            if data.source is not service:
                return
            if isinstance(data.frame, InputAudioRawFrame) and data.frame.audio == stale_audio:
                stale_forwarded.set()
            if isinstance(data.frame, TranscriptionFrame):
                transcripts.append(data.frame)
                if len(transcripts) == 2:
                    transcripts_received.set()
            if isinstance(data.frame, ErrorFrame):
                errors.append(data.frame)

    async with serve(handler, "127.0.0.1", 0, process_request=process_request) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            reconnect_backoff_min_wait=0.01,
            reconnect_backoff_max_wait=0.01,
        )
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False, observers=[Observer()])
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(InputAudioRawFrame(b"\x01\x00" * 160, 16000, 1))

        @service.event_handler("on_connected")
        async def connected(service):
            if len(sessions) == 2:
                await worker.queue_frame(InputAudioRawFrame(fresh_audio, 16000, 1))

        async def drive():
            try:
                async with asyncio.timeout(3):
                    await reconnecting.wait()
                    await worker.queue_frame(InputAudioRawFrame(stale_audio, 16000, 1))
                    await stale_forwarded.wait()
                    allow_reconnect.set()
                    await transcripts_received.wait()
                await worker.queue_frame(EndFrame())
            finally:
                allow_reconnect.set()
                if not transcripts_received.is_set():
                    await worker.cancel()

        await asyncio.gather(runner.run(), drive())

    assert len(sessions) == 2
    assert sessions[0][0] != sessions[1][0]
    assert [struct.unpack(">i", packet[4:8])[0] for packet in sessions[1][1]] == [1, 2]
    assert gzip.decompress(sessions[1][1][1][12:]) == fresh_audio
    assert [frame.text for frame in transcripts] == ["Okay.", "Okay."]
    assert any("1011" in error.error and sessions[0][0] in error.error for error in errors)


@pytest.mark.asyncio
async def test_blocked_audio_send_recovers_without_replaying_queued_audio(monkeypatch):
    from websockets.asyncio.client import ClientConnection

    original_send = ClientConnection.send
    connections = []
    recovered = asyncio.Event()
    blocked = asyncio.Event()

    async def stalled_send(connection, message, *args, **kwargs):
        if message[1] == 0x21 and not blocked.is_set():
            blocked.set()
            await asyncio.Event().wait()
        return await original_send(connection, message, *args, **kwargs)

    monkeypatch.setattr(ClientConnection, "send", stalled_send)

    async def handler(websocket):
        messages = []
        connections.append(messages)
        messages.append(await websocket.recv())
        if len(connections) == 2:
            recovered.set()
        async for message in websocket:
            messages.append(message)
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {}}, sequence=-1))

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            send_timeout=0.05,
            reconnect_backoff_min_wait=0.01,
            reconnect_backoff_max_wait=0.01,
        )
        observer = _ResultObserver(service, 1)
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False, observers=[observer])
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(InputAudioRawFrame(b"\x01\x00" * 16000, 16000, 1))

        async def finish():
            try:
                await asyncio.wait_for(recovered.wait(), 2)
                await worker.queue_frame(EndFrame())
            finally:
                if not recovered.is_set():
                    await worker.cancel()

        await asyncio.gather(runner.run(), finish())

    assert len(connections) == 2
    assert len(connections[0]) == 1
    assert [struct.unpack(">i", message[4:8])[0] for message in connections[1]] == [1, -2]
    assert any("send timed out" in frame.error for frame in observer.frames)


@pytest.mark.asyncio
async def test_audio_queue_overflow_drops_backlog_and_reports_error():
    packets = []

    async def handler(websocket):
        await websocket.recv()
        async for message in websocket:
            packets.append(gzip.decompress(message[12:]))
            if struct.unpack(">i", message[4:8])[0] < 0:
                await websocket.send(_response({"result": {}}, sequence=-1))

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        _, up = await run_test(
            service,
            frames_to_send=[InputAudioRawFrame(b"\1\0" * 160000, 16000, 1)],
            start_timeout=5,
        )
    assert any(isinstance(frame, ErrorFrame) and "queue full" in frame.error for frame in up)
    assert packets == [b""]


@pytest.mark.asyncio
async def test_interruption_and_silence_keep_stream_open_for_resumed_speech():
    sessions = []

    async def handler(websocket):
        sessions.append(websocket.request.headers["X-Api-Request-Id"])
        await websocket.recv()
        index = 0
        async for message in websocket:
            sequence = struct.unpack(">i", message[4:8])[0]
            if sequence < 0:
                await websocket.send(_response({"result": {}}, sequence=-1))
                continue
            await websocket.send(
                _response(
                    {
                        "result": {
                            "utterances": [
                                {
                                    "text": "Okay.",
                                    "definite": True,
                                    "start_time": index * 1000,
                                    "end_time": index * 1000 + 500,
                                }
                            ]
                        }
                    }
                )
            )
            index += 1

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            audio_passthrough=False,
        )
        down, _ = await run_test(
            service,
            frames_to_send=[
                InputAudioRawFrame(b"\1\0" * 160, 16000, 1),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.05),
                InputAudioRawFrame(b"\2\0" * 160, 16000, 1),
                SleepFrame(sleep=0.05),
            ],
            expected_down_frames=[
                STTMetadataFrame,
                TranscriptionFrame,
                InterruptionFrame,
                TranscriptionFrame,
            ],
            start_timeout=5,
        )
    assert len(sessions) == 1
    assert all(not frame.finalized for frame in down if isinstance(frame, TranscriptionFrame))


@pytest.mark.asyncio
async def test_cancel_releases_blocked_sender_and_connection_without_flushing(monkeypatch):
    from websockets.asyncio.client import ClientConnection

    original_send = ClientConnection.send
    blocked = asyncio.Event()
    send_cancelled = asyncio.Event()
    closed = asyncio.Event()
    messages = []

    async def stalled_send(connection, message, *args, **kwargs):
        if message[1] == 0x21:
            blocked.set()
            try:
                await asyncio.Event().wait()
            finally:
                send_cancelled.set()
        return await original_send(connection, message, *args, **kwargs)

    monkeypatch.setattr(ClientConnection, "send", stalled_send)

    async def handler(websocket):
        async for message in websocket:
            messages.append(message)
        closed.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            send_timeout=10,
            flush_timeout=10,
        )
        worker = PipelineWorker(Pipeline([service]), enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(InputAudioRawFrame(b"\1\0" * 16000, 16000, 1))

        async def finish():
            await asyncio.wait_for(blocked.wait(), 5)
            async with asyncio.timeout(0.5):
                await worker.cancel()
                await closed.wait()
                await send_cancelled.wait()

        await asyncio.gather(runner.run(), finish())
    assert len(messages) == 1
    assert service.task_manager.current_tasks() == []


@pytest.mark.asyncio
async def test_handshake_failure_emits_error_and_releases_resources():
    request_ids = []

    async def reject(connection, request):
        request_ids.append(request.headers["X-Api-Request-Id"])
        return connection.respond(401, "Invalid API key")

    async def handler(websocket):
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0, process_request=reject) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        observer = _ResultObserver(service, 1)
        await run_test(service, frames_to_send=[], observers=[observer], start_timeout=2)
    assert len(observer.frames) == 1
    assert "401" in observer.frames[0].error
    assert request_ids[0] in observer.frames[0].error
    assert service.task_manager.current_tasks() == []


@pytest.mark.asyncio
async def test_reconnect_exhaustion_emits_errors_with_attempt_request_ids():
    request_ids = []

    async def reject_reconnect(connection, request):
        request_ids.append(request.headers["X-Api-Request-Id"])
        if len(request_ids) > 1:
            return connection.respond(503, "Temporarily unavailable")

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.close(code=1011, reason="temporary outage")

    async with serve(handler, "127.0.0.1", 0, process_request=reject_reconnect) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            reconnect_backoff_min_wait=0.01,
            reconnect_backoff_max_wait=0.01,
        )
        errors = await _run_until_results(service, 5)
    assert len(request_ids) == 4
    assert len(set(request_ids)) == 4
    for request_id, error in zip(request_ids, errors):
        assert request_id in error.error
        assert error.fatal is False
    assert "failed to reconnect after 3 attempts" in errors[-1].error
    assert service.is_usable is False
    assert service.task_manager.current_tasks() == []


@pytest.mark.asyncio
async def test_second_pass_segment_without_start_time_becomes_final_once():
    # Second-pass segments carry an end timestamp and no start timestamp.
    segment = {"text": "Okay.", "definite": True, "end_time": 2902}

    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        for utterances in ([segment], [segment]):
            await websocket.send(_response({"result": {"utterances": utterances}}))
        await websocket.send(_response({"result": {"text": "Done"}}))
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        frames = await _run_until_results(service, 2)

    assert [(type(frame), frame.text) for frame in frames] == [
        (TranscriptionFrame, "Okay."),
        (InterimTranscriptionFrame, "Done"),
    ]


@pytest.mark.asyncio
async def test_definite_result_without_end_timestamp_reports_protocol_error():
    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.send(
            _response({"result": {"utterances": [{"text": "Okay.", "definite": True}]}})
        )
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        frames = await _run_until_results(service, 1)
    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "timestamp" in frames[0].error


@pytest.mark.asyncio
async def test_invalid_response_model_emits_nonfatal_error():
    async def handler(websocket):
        await websocket.recv()
        await websocket.recv()
        await websocket.send(_response({"result": {"utterances": "not a list"}}))
        await _acknowledge_end(websocket)

    async with serve(handler, "127.0.0.1", 0) as server:
        service = VolcengineSTTService(
            api_key="test-key",
            resource_id="test-resource",
            ws_url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        )
        frames = await _run_until_results(service, 1)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "utterances" in frames[0].error
    assert frames[0].fatal is False


@pytest.mark.parametrize("setting", ["flush_timeout", "send_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_timeout_must_be_finite_and_positive(setting, value):
    with pytest.raises(ValueError, match="finite and positive"):
        VolcengineSTTService(api_key="test-key", resource_id="test-resource", **{setting: value})
