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
import math
import struct
import zlib
from collections.abc import AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from websockets.protocol import State

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.services.websocket_service import ReportErrorCallback
from pipecat.utils.time import time_now_iso8601


def _encode_request(payload: bytes, sequence: int, *, audio: bool = False) -> bytes:
    """Encode a sequenced, gzip-compressed client request."""
    body = gzip.compress(payload)
    message_type = 0x2 if audio else 0x1
    flags = 0x3 if sequence < 0 else 0x1
    serialization = 0x0 if audio else 0x1
    header = bytes((0x11, message_type << 4 | flags, serialization << 4 | 0x1, 0))
    return header + struct.pack(">iI", sequence, len(body)) + body


class _RequestUser(BaseModel):
    """Volcengine request user identity."""

    model_config = ConfigDict(extra="forbid")

    uid: str


class _RequestAudio(BaseModel):
    """Volcengine request audio format."""

    model_config = ConfigDict(extra="forbid")

    format: str = "pcm"
    codec: str = "raw"
    rate: int
    bits: int = 16
    channel: int = 1


class _RequestCorpus(BaseModel):
    """Volcengine request corpus configuration."""

    model_config = ConfigDict(extra="forbid")

    context: str


class _RecognitionRequest(BaseModel):
    """Volcengine streaming recognition request payload."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "bigmodel"
    result_type: str = "single"
    show_utterances: bool = True
    # The provider closes a segment, and marks it definite, only when its own VAD
    # detects the end of an utterance. That VAD runs in second-pass mode, so a
    # long-lived connection reports nothing but interim results without it.
    enable_nonstream: bool = True
    enable_itn: bool | None = None
    enable_punc: bool | None = None
    corpus: _RequestCorpus | None = None


class _StreamingRecognitionRequest(BaseModel):
    """Complete Volcengine streaming recognition request payload."""

    model_config = ConfigDict(extra="forbid")

    user: _RequestUser
    audio: _RequestAudio
    request: _RecognitionRequest


class _RecognitionUtterance(BaseModel):
    """One recognition utterance in a Volcengine response."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    definite: bool = False
    start_time: int | None = None
    end_time: int | None = None


class _RecognitionResult(BaseModel):
    """Recognition result in a Volcengine response."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    utterances: list[_RecognitionUtterance] = Field(default_factory=list)


class _StreamingRecognitionResponse(BaseModel):
    """Volcengine streaming recognition response payload."""

    model_config = ConfigDict(extra="allow")

    result: _RecognitionResult = Field(default_factory=_RecognitionResult)


def _decode_response(message: bytes | str) -> tuple[_StreamingRecognitionResponse, bool]:
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
        except (OSError, EOFError, zlib.error) as exc:
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
    return _StreamingRecognitionResponse.model_validate(data), bool(flags & 0x2)


class VolcengineSTTService(WebsocketSTTService):
    """Stream mono signed 16-bit PCM to Volcengine's v3 recognition API.

    Credentials belong in backend configuration. Definite recognition segments
    are independent of user turn completion, which remains the VAD's concern.
    Segment end timestamps identify committed results within each connection.
    Audio queued during an overflow or connection failure is discarded.
    """

    class InputParams(BaseModel):
        """Recognition options fixed for the lifetime of a session.

        Parameters:
            enable_itn: Inverse text normalization, or the provider default when omitted.
            enable_punc: Punctuation, or the provider default when omitted.
            corpus_context: Provider corpus context object, including supported hotword hints.
                Encoded as a JSON string in ``request.corpus.context``.
        """

        model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

        enable_itn: bool | None = None
        enable_punc: bool | None = None
        corpus_context: dict[str, JsonValue] | None = None

    def __init__(
        self,
        *,
        api_key: str,
        resource_id: str,
        ws_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
        sample_rate: int | None = None,
        flush_timeout: float = 2.0,
        send_timeout: float = 1.0,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize streaming recognition.

        Args:
            api_key: Backend API key used for the WebSocket handshake.
            resource_id: Enabled recognition resource identifier.
            ws_url: Recognition endpoint.
            sample_rate: PCM sample rate, or the pipeline input rate when omitted.
            flush_timeout: Maximum seconds to drain audio and await the final response.
            send_timeout: Maximum seconds to send a protocol packet.
            params: Recognition options. Second-pass recognition is always disabled.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate, settings=STTSettings(model="bigmodel", language=None), **kwargs
        )
        for name, value in (("flush_timeout", flush_timeout), ("send_timeout", send_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._send_timeout = send_timeout
        self._params = (params or self.InputParams()).model_copy(deep=True)
        self._flush_timeout = flush_timeout
        self._final_response = asyncio.Event()
        self._api_key = api_key
        self._resource_id = resource_id
        self._ws_url = ws_url
        self._accepting_audio = False
        self._request_id = ""
        self._sequence = 1
        self._committed_segments: set[tuple[str, int]] = set()
        self._receive_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Configure the audio rate and open the recognition stream.

        Args:
            setup: Pipeline setup parameters.
        """
        await super().setup(setup)
        try:
            await self._connect()
        except Exception as exc:
            await self._disconnect()
            await self._report_error(
                ErrorFrame(f"Connection failed: {exc}", exception=exc),
                force_treat_as_permanent=True,
            )

    async def stop(self, frame: EndFrame) -> None:
        """Drain audio and await the final response within the flush timeout.

        Args:
            frame: The end frame.
        """
        self._disconnecting = True
        await self._cancel_keepalive_task()
        try:
            async with asyncio.timeout(self._flush_timeout):
                await self._audio_queue.join()
                if self._websocket and self._websocket.state is State.OPEN:
                    await self._send_packet(b"", -self._sequence, audio=True)
                    await self._final_response.wait()
        except TimeoutError as exc:
            await self.push_error(
                f"Volcengine STT request {self._request_id}: final response timed out",
                exception=exc,
            )
        except Exception as exc:
            await self.push_error(
                f"Volcengine STT request {self._request_id}: {exc}",
                exception=exc,
            )
        finally:
            await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Submit audio to the stream; recognition arrives in the receive task."""
        if (
            self._accepting_audio
            and not self._disconnecting
            and self._websocket
            and self._websocket.state is State.OPEN
        ):
            packet_size = max(2, self.sample_rate // 10 * 2)
            for offset in range(0, len(audio), packet_size):
                try:
                    self._audio_queue.put_nowait(audio[offset : offset + packet_size])
                except asyncio.QueueFull:
                    self._discard_queued_audio()
                    await self.push_error(
                        f"Volcengine STT request {self._request_id}: audio queue full; audio dropped"
                    )
                    break
        yield None

    def _discard_queued_audio(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
            self._audio_queue.task_done()

    async def _send_packet(self, payload: bytes, sequence: int, *, audio: bool = False) -> None:
        if self._websocket is None:
            raise ConnectionError("Recognition stream is not connected")
        try:
            async with asyncio.timeout(self._send_timeout):
                await self._websocket.send(_encode_request(payload, sequence, audio=audio))
        except TimeoutError as exc:
            raise TimeoutError("Protocol packet send timed out") from exc

    async def _send_audio(self) -> None:
        while True:
            audio = await self._audio_queue.get()
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    await self._send_packet(audio, self._sequence, audio=True)
                    self._sequence += 1
            except Exception as exc:
                self._accepting_audio = False
                await self.push_error(
                    f"Volcengine STT request {self._request_id}: {exc}",
                    exception=exc,
                )
                if self._websocket:
                    await self._websocket.close()
                return
            finally:
                self._audio_queue.task_done()

    async def _send_keepalive(self, silence: bytes) -> None:
        async for _ in self.run_stt(silence):
            pass
        self._record_stt_audio_usage(silence)

    async def _connect(self) -> None:
        await super()._connect()
        await self._connect_websocket()
        self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        self._accepting_audio = False
        self._request_id = str(uuid4())
        self._sequence = 1
        self._committed_segments.clear()
        self._final_response.clear()
        self._websocket = await self._websocket_connect(
            self._ws_url,
            additional_headers={
                "X-Api-Key": self._api_key,
                "X-Api-Resource-Id": self._resource_id,
                "X-Api-Request-Id": self._request_id,
            },
        )
        request = _StreamingRecognitionRequest(
            user=_RequestUser(uid=self._request_id),
            audio=_RequestAudio(rate=self.sample_rate),
            request=_RecognitionRequest(
                **self._params.model_dump(exclude_none=True, exclude={"corpus_context"}),
                corpus=(
                    _RequestCorpus(
                        context=json.dumps(self._params.corpus_context, ensure_ascii=False)
                    )
                    if self._params.corpus_context is not None
                    else None
                ),
            ),
        )
        assert self._websocket is not None
        try:
            await self._send_packet(
                request.model_dump_json(exclude_none=True).encode(), self._sequence
            )
        except BaseException:
            await self._disconnect_websocket()
            raise
        self._sequence += 1
        self._accepting_audio = True
        self._send_task = self.create_task(self._send_audio())
        await self._call_event_handler("on_connected")

    async def _disconnect_websocket(self) -> None:
        self._accepting_audio = False
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None
        self._discard_queued_audio()
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _report_error(
        self, error: ErrorFrame, force_treat_as_permanent: bool = False
    ) -> None:
        error.error = f"Volcengine STT request {self._request_id}: {error.error}"
        await super()._report_error(error, force_treat_as_permanent)

    async def _maybe_try_reconnect(
        self,
        error_message: str,
        report_error: ReportErrorCallback,
        error: Exception | None = None,
    ) -> bool:
        if not self._disconnecting and self._reconnect_on_error:
            await report_error(ErrorFrame(error_message, exception=error))
        return await super()._maybe_try_reconnect(error_message, report_error, error)

    async def _receive_messages(self) -> None:
        if self._websocket:
            async for message in self._websocket:
                try:
                    response, is_final = _decode_response(message)
                    result = response.result
                    utterances = result.utterances
                    if not utterances:
                        utterances = [_RecognitionUtterance(text=result.text)]
                    for utterance in utterances:
                        text = utterance.text
                        if not text:
                            continue
                        if utterance.definite:
                            end = utterance.end_time
                            if end is None:
                                raise ValueError("Definite segment is missing an end timestamp")
                            segment = (self._request_id, end)
                            if segment in self._committed_segments:
                                continue
                            self._committed_segments.add(segment)
                        frame_type = (
                            TranscriptionFrame if utterance.definite else InterimTranscriptionFrame
                        )
                        await self.push_frame(
                            frame_type(
                                text,
                                self._user_id,
                                time_now_iso8601(),
                                result=response.model_dump(exclude_unset=True),
                            )
                        )
                    if is_final:
                        self._final_response.set()
                except (ValueError, TypeError, AttributeError) as exc:
                    await self.push_error(
                        f"Volcengine STT request {self._request_id}: {exc}",
                        exception=exc,
                    )
