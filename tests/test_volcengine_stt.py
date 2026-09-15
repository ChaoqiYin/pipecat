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
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.services.volcengine.stt import VolcengineSTTService
from pipecat.tests.utils import run_test


def _response(payload, *, sequence=1, compressed=True):
    body = json.dumps(payload).encode()
    if compressed:
        body = gzip.compress(body)
    header = bytes((0x11, 0x93 if sequence < 0 else 0x91, 0x11 if compressed else 0x10, 0))
    return header + struct.pack(">iI", sequence, len(body)) + body


class _ResultObserver(BaseObserver):
    def __init__(self, service, count):
        super().__init__()
        self.service = service
        self.count = count
        self.received = asyncio.Event()

    async def on_push_frame(self, data):
        if data.source is self.service and isinstance(
            data.frame, (InterimTranscriptionFrame, TranscriptionFrame, ErrorFrame)
        ):
            self.count -= 1
            if self.count == 0:
                self.received.set()


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
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(
            api_key="test-key",
            ws_url=f"ws://127.0.0.1:{port}",
            audio_passthrough=False,
        )
        observer = _ResultObserver(service, 2)

        @service.event_handler("on_connected")
        async def connected(service):
            pass

        # End only after the receive task has emitted both results.
        from pipecat.frames.frames import EndFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.workers.runner import WorkerRunner

        received = []

        class Collector(BaseObserver):
            async def on_push_frame(self, data):
                if data.source is service:
                    received.append(data.frame)

        worker = PipelineWorker(
            Pipeline([service]),
            enable_rtvi=False,
            observers=[observer, Collector()],
        )
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(InputAudioRawFrame(b"\0\0" * 160, 16000, 1))

        async def finish():
            try:
                await asyncio.wait_for(observer.received.wait(), 3)
            finally:
                await worker.queue_frame(EndFrame())

        await asyncio.gather(runner.run(), finish())

    transcripts = [
        f for f in received if isinstance(f, (InterimTranscriptionFrame, TranscriptionFrame))
    ]
    assert [(type(f), f.text) for f in transcripts] == [
        (InterimTranscriptionFrame, "Hello"),
        (TranscriptionFrame, "Hello there."),
    ]
    assert transcripts[1].result["result"]["utterances"] == [utterance]
    assert transcripts[1].finalized is False


@pytest.mark.asyncio
async def test_stream_authenticates_and_sends_sequenced_pcm_with_end_marker():
    sessions = []

    async def handler(websocket):
        messages = []
        sessions.append((websocket.request.headers, messages))
        async for message in websocket:
            messages.append(message)

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
        assert request["request"]["enable_nonstream"] is False
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
        (b"\x11\x90\x10\x00\x00\x00\x00\x01{", "JSON"),
        ("unexpected text", "binary"),
    ],
)
async def test_protocol_failures_emit_nonfatal_error_with_request_id(message, expected):
    from pipecat.frames.frames import EndFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner

    request_ids = []
    errors = []

    async def handler(websocket):
        request_ids.append(websocket.request.headers["X-Api-Request-Id"])
        await websocket.recv()
        await websocket.recv()
        await websocket.send(message)
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(api_key="test-key", ws_url=f"ws://127.0.0.1:{port}")
        observer = _ResultObserver(service, 1)

        class Collector(BaseObserver):
            async def on_push_frame(self, data):
                if data.source is service and isinstance(data.frame, ErrorFrame):
                    errors.append(data.frame)

        worker = PipelineWorker(
            Pipeline([service]), enable_rtvi=False, observers=[observer, Collector()]
        )
        runner = WorkerRunner()
        await runner.add_workers(worker)
        await worker.queue_frame(InputAudioRawFrame(b"\0\0" * 160, 16000, 1))

        async def finish():
            try:
                await asyncio.wait_for(observer.received.wait(), 5)
            finally:
                await worker.queue_frame(EndFrame())

        await asyncio.gather(runner.run(), finish())

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

    audio = b"\x01\x00" * 16000
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = VolcengineSTTService(
            api_key="test-key",
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
