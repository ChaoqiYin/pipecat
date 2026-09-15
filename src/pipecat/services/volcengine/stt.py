#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Volcengine bidirectional streaming speech recognition.

Protocol reference: https://www.volcengine.com/docs/6561/1354869.
"""

import asyncio
import gzip
import json
import struct
from collections.abc import AsyncGenerator
from uuid import uuid4

from websockets.protocol import State

from pipecat.frames.frames import EndFrame, Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601


def _encode_request(payload: bytes, sequence: int, *, audio: bool = False) -> bytes:
    """Encode a sequenced, gzip-compressed client request."""
    body = gzip.compress(payload)
    message_type = 0x2 if audio else 0x1
    flags = 0x3 if sequence < 0 else 0x1
    serialization = 0x0 if audio else 0x1
    header = bytes((0x11, message_type << 4 | flags, serialization << 4 | 0x1, 0))
    return header + struct.pack(">iI", sequence, len(body)) + body


def _decode_response(message: bytes | str) -> dict:
    """Decode a full server response or raise a descriptive protocol error."""
    if not isinstance(message, bytes):
        raise ValueError("Expected a binary response")
    if len(message) < 4:
        raise ValueError("Response header is truncated")
    header_size = (message[0] & 0xF) * 4
    if message[0] >> 4 != 1 or header_size < 4:
        raise ValueError("Unsupported protocol header")
    message_type = message[1] >> 4
    flags = message[1] & 0xF
    serialization = message[2] >> 4
    compression = message[2] & 0xF
    if message_type not in (0x9, 0xF) or flags not in (0, 1, 2, 3):
        raise ValueError("Unsupported response message type or flags")
    if compression not in (0, 1) or serialization not in (0, 1):
        raise ValueError("Unsupported response encoding")
    offset = header_size + (4 if message_type == 0xF or flags & 0x1 else 0)
    if len(message) < offset + 4:
        raise ValueError("Response payload header is truncated")
    size = struct.unpack_from(">I", message, offset)[0]
    payload = message[offset + 4 :]
    if len(payload) != size:
        raise ValueError("Response payload size does not match its header")
    if compression == 1:
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise ValueError("Invalid gzip response") from exc
    if message_type == 0xF:
        code = struct.unpack_from(">I", message, header_size)[0]
        raise ValueError(f"Service error {code}: {payload.decode('utf-8', errors='replace')}")
    if serialization != 1:
        raise ValueError("Expected a JSON response")
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON response") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


class VolcengineSTTService(WebsocketSTTService):
    """Stream mono signed 16-bit PCM to Volcengine's v3 recognition API.

    Credentials belong in backend configuration. Definite recognition segments
    are independent of user turn completion, which remains the VAD's concern.
    """

    def __init__(
        self,
        *,
        api_key: str,
        resource_id: str = "volc.seedasr.sauc.duration",
        ws_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
        sample_rate: int | None = None,
        **kwargs,
    ):
        """Initialize streaming recognition.

        Args:
            api_key: Backend API key used for the WebSocket handshake.
            resource_id: Enabled recognition resource identifier.
            ws_url: Recognition endpoint.
            sample_rate: PCM sample rate, or the pipeline input rate when omitted.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate, settings=STTSettings(model="bigmodel", language=None), **kwargs
        )
        self._api_key = api_key
        self._resource_id = resource_id
        self._ws_url = ws_url
        self._request_id = ""
        self._sequence = 1
        self._receive_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

    async def setup(self, setup: FrameProcessorSetup):
        """Configure the audio rate and open the recognition stream."""
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Send the end-of-audio marker before closing the stream."""
        try:
            await self._audio_queue.join()
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(_encode_request(b"", -self._sequence, audio=True))
        except Exception as exc:
            await self.push_error(
                f"Volcengine STT request {self._request_id}: {exc}",
                exception=exc,
            )
        finally:
            await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Submit audio to the stream; recognition arrives in the receive task."""
        if self._websocket and self._websocket.state is State.OPEN:
            packet_size = max(2, self.sample_rate // 10 * 2)
            for offset in range(0, len(audio), packet_size):
                try:
                    self._audio_queue.put_nowait(audio[offset : offset + packet_size])
                except asyncio.QueueFull:
                    await self.push_error(
                        f"Volcengine STT request {self._request_id}: audio queue full; audio dropped"
                    )
                    break
        yield None

    async def _send_audio(self):
        while True:
            audio = await self._audio_queue.get()
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    await self._websocket.send(_encode_request(audio, self._sequence, audio=True))
                    self._sequence += 1
            except Exception as exc:
                await self.push_error(
                    f"Volcengine STT request {self._request_id}: {exc}",
                    exception=exc,
                )
            finally:
                self._audio_queue.task_done()

    async def _send_keepalive(self, silence: bytes):
        async for _ in self.run_stt(silence):
            pass
        self._record_stt_audio_usage(silence)

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        self._send_task = self.create_task(self._send_audio())
        self._receive_task = self.create_task(self._receive_messages())

    async def _disconnect(self):
        await super()._disconnect()
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
            self._audio_queue.task_done()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        self._request_id = str(uuid4())
        self._sequence = 1
        self._websocket = await self._websocket_connect(
            self._ws_url,
            additional_headers={
                "X-Api-Key": self._api_key,
                "X-Api-Resource-Id": self._resource_id,
                "X-Api-Request-Id": self._request_id,
            },
        )
        request = {
            "user": {"uid": self._request_id},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": self.sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "result_type": "single",
                "show_utterances": True,
                "enable_nonstream": False,
            },
        }
        assert self._websocket is not None
        await self._websocket.send(_encode_request(json.dumps(request).encode(), self._sequence))
        self._sequence += 1
        await self._call_event_handler("on_connected")

    async def _disconnect_websocket(self):
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        if self._websocket:
            async for message in self._websocket:
                try:
                    data = _decode_response(message)
                    result = data.get("result", {})
                    utterances = result.get("utterances", [])
                    if not utterances:
                        utterances = [{"text": result.get("text", ""), "definite": False}]
                    for utterance in utterances:
                        text = utterance.get("text", "")
                        if not text:
                            continue
                        frame_type = (
                            TranscriptionFrame
                            if utterance.get("definite")
                            else InterimTranscriptionFrame
                        )
                        await self.push_frame(
                            frame_type(
                                text,
                                self._user_id,
                                time_now_iso8601(),
                                result=data,
                            )
                        )
                except (ValueError, TypeError, AttributeError) as exc:
                    await self.push_error(
                        f"Volcengine STT request {self._request_id}: {exc}",
                        exception=exc,
                    )
