#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#

"""Volcengine bidirectional streaming text-to-speech.

Protocol reference: https://docs.volcengine.com/docs/6561/2630027.
"""

import asyncio
import base64
import json
import math
import secrets
import struct
from collections.abc import AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.services.websocket_service import ReportErrorCallback

_SESSION_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Event identifiers for the bidirectional TTS protocol.
_START_CONNECTION = 1
_FINISH_CONNECTION = 2
_START_SESSION = 100
_CANCEL_SESSION = 101
_FINISH_SESSION = 102
_TASK_REQUEST = 200
_SERVER_SESSION_STARTED = 150
_SERVER_SESSION_CANCELED = 151
_SERVER_SESSION_FINISHED = 152
_SERVER_SESSION_FAILED = 153


def _new_session_id() -> str:
    """Generate a provider session identifier matching the reference SDK format."""
    return "".join(secrets.choice(_SESSION_ID_ALPHABET) for _ in range(12))


def _encode_event(event: int, payload: dict, *, session_id: str = "") -> bytes:
    """Encode a full client event frame carrying a JSON payload.

    Args:
        event: The protocol event identifier.
        payload: The JSON payload sent with the event.
        session_id: The session identifier carried in the frame header for
            session-scoped events, or empty for connection-scoped events.

    Returns:
        A binary frame with the FullClient message type, the WithEvent and
        JSON-serialization flags, and no sequence.
    """
    body = json.dumps(payload).encode()
    header = bytes((0x11, 0x14, 0x10, 0x00))
    sid = session_id.encode()
    return (
        header
        + struct.pack(">i", event)
        + (struct.pack(">I", len(sid)) + sid if session_id else b"")
        + struct.pack(">I", len(body))
        + body
    )


class VolcengineTTSService(WebsocketTTSService):
    """Stream text to Volcengine's bidirectional TTS API and play the resulting PCM.

    The WebSocket connection is opened once at setup and reused across turns. Each
    turn begins a fresh synthesis session: the service sends ``StartSession`` before
    the first sentence of a turn and ``FinishSession`` once the turn ends, so audio
    from one turn never reuses the previous turn's session. Text is sent as
    ``TaskRequest`` events; audio arrives asynchronously on the receive task as
    either raw ``AudioOnlyServer`` frames or ``Response`` events whose JSON payload
    base64 encodes the PCM. The final flag on an audio frame marks the end of a
    sentence and closes its audio context, while VAD and the aggregators remain
    responsible for turn management.

    Interruptions cancel the active session in place with a ``CancelSession`` event
    (the connection is kept open). A graceful end finishes the session and
    connection, waiting within a bounded timeout for the provider to acknowledge.
    """

    class InputParams(BaseModel):
        """Synthesis options fixed for the lifetime of the service.

        Parameters:
            speaker: Voice ID used to synthesize speech.
            audio_format: Audio container requested from the provider.
            sample_rate: Output sample rate, or the pipeline rate when omitted.
            additions: Optional provider extensions encoded as a JSON string.
        """

        model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

        speaker: str
        audio_format: str = "pcm"
        sample_rate: int | None = None
        additions: str | None = None

    def __init__(
        self,
        *,
        api_key: str,
        resource_id: str,
        ws_url: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        speaker: str | None = None,
        sample_rate: int | None = None,
        flush_timeout: float = 2.0,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize streaming synthesis.

        Args:
            api_key: Backend API key used for the WebSocket handshake.
            resource_id: Enabled TTS resource identifier.
            ws_url: Synthesis endpoint.
            speaker: Voice ID, or the value from ``params`` when omitted.
            sample_rate: PCM sample rate, or the pipeline output rate when omitted.
            flush_timeout: Maximum seconds to finish the active session and await
                the provider's ``SessionFinished`` acknowledgment on a graceful end.
            params: Synthesis options.
            **kwargs: Additional arguments passed to WebsocketTTSService.
        """
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(model=None, voice=None, language=None),
            push_start_frame=True,
            **kwargs,
        )
        if not math.isfinite(flush_timeout) or flush_timeout <= 0:
            raise ValueError("flush_timeout must be finite and positive")
        self._params = (
            params.model_copy(deep=True)
            if params is not None
            else self.InputParams(speaker=speaker or "")
        )
        if not self._params.speaker:
            raise ValueError("speaker must be set")
        self._params.sample_rate = self._params.sample_rate or sample_rate
        self._api_key = api_key
        self._resource_id = resource_id
        self._connect_id = str(uuid4())
        self._ws_url = ws_url
        self._flush_timeout = flush_timeout
        # Active provider session id, or "" when no session is open. Reset on
        # reconnect; a fresh session is started lazily for the next turn.
        self._session_id = ""
        # context_id of the turn the active session was opened for, or None.
        self._session_context_id: str | None = None
        # Set once the provider acknowledges SessionFinished for the final flush.
        self._session_finished = asyncio.Event()
        self._receive_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Configure the audio rate and open the synthesis stream.

        Args:
            setup: Pipeline setup parameters.
        """
        await super().setup(setup)
        self._params.sample_rate = self._params.sample_rate or self.sample_rate
        try:
            await self._connect()
        except Exception as exc:
            await self._disconnect()
            await self._report_error(
                ErrorFrame(f"Connection failed: {exc}", exception=exc),
                force_treat_as_permanent=True,
            )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Submit text for synthesis; audio arrives asynchronously in the receive task.

        A fresh provider session is opened if none is active for this turn. The
        ``TaskRequest`` carries no ``isLast`` marker: the SDK's ``isLast`` flag maps
        to a separate ``FinishSession`` event sent after the last request (see
        :meth:`flush_audio`), so each task request here just submits its text.

        Args:
            text: The text to synthesize.
            context_id: The audio context that will receive the produced frames.

        Yields:
            ``None`` on success, or an ``ErrorFrame`` describing a send failure.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            yield ErrorFrame(error="Volcengine TTS websocket is not connected")
            return
        error = await self._ensure_session(context_id)
        if error is not None:
            yield error
            return
        request = _encode_event(
            _TASK_REQUEST,
            {
                "user": {"uid": self._connect_id},
                "event": _TASK_REQUEST,
                "req_params": {"text": text},
            },
            session_id=self._session_id,
        )
        try:
            await self._websocket.send(request)
        except Exception as exc:
            yield ErrorFrame(error=f"Failed to send synthesis request: {exc}", exception=exc)
            return
        yield None

    async def _ensure_session(self, context_id: str) -> ErrorFrame | None:
        """Open a fresh provider session for ``context_id`` when none is active.

        If a session is already open for a previous turn, it is finished first so
        new text starts a new session rather than reusing stale audio. Starting a
        session does not block `process_frame`: it is a compact sequence of sends.

        Args:
            context_id: The audio context whose turn owns the session.

        Returns:
            ``None`` on success, or an ``ErrorFrame`` describing a failure.
        """
        assert self._websocket is not None
        if self._session_id and self._session_context_id == context_id:
            return None
        if self._session_id:
            await self._send_event(_FINISH_SESSION, {}, session_id=self._session_id)
            self._session_id = ""
            self._session_context_id = None
        self._session_id = _new_session_id()
        try:
            await self._websocket.send(
                _encode_event(
                    _START_SESSION,
                    self._start_session_payload(),
                    session_id=self._session_id,
                )
            )
        except Exception as exc:
            self._session_id = ""
            return ErrorFrame(error=f"Failed to start session: {exc}", exception=exc)
        self._session_context_id = context_id
        return None

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
        # Reconnecting abandons any previous session and synthesis state; a fresh
        # session is opened lazily for the next turn.
        self._session_id = ""
        self._session_context_id = None
        self._session_finished.clear()
        self._websocket = await self._websocket_connect(
            self._ws_url,
            additional_headers={
                "X-Api-Key": self._api_key,
                "X-Api-Resource-Id": self._resource_id,
                "X-Api-Connect-Id": self._connect_id,
            },
        )
        assert self._websocket is not None
        try:
            await self._websocket.send(
                _encode_event(_START_CONNECTION, {"namespace": "BidirectionalTTS"})
            )
        except BaseException:
            await self._disconnect_websocket()
            raise
        await self._call_event_handler("on_connected")

    async def _disconnect_websocket(self) -> None:
        self._session_id = ""
        self._session_context_id = None
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_event(self, event: int, payload: dict, *, session_id: str = "") -> bool:
        """Send a client event on the open connection without raising.

        Args:
            event: The protocol event identifier.
            payload: The JSON payload to send.
            session_id: The session identifier for session-scoped events.

        Returns:
            ``True`` when the event was sent, ``False`` when the connection is
            gone or a send failed (already reported as a non-fatal error).
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return False
        try:
            await self._websocket.send(_encode_event(event, payload, session_id=session_id))
            return True
        except Exception as exc:
            await self.push_error(
                f"Volcengine TTS: failed to send event {event}: {exc}", exception=exc
            )
            return False

    async def on_turn_context_created(self, context_id: str):
        """Finish any still-active previous session before a new turn starts.

        A session is opened lazily in :meth:`run_tts` for the first sentence of a
        turn. If a stray session from an earlier turn is still open here, finishing
        it prevents the new turn's text from reusing its audio.

        Args:
            context_id: The newly created turn context ID.
        """
        if self._session_id and self._session_context_id != context_id:
            await self._send_event(_FINISH_SESSION, {}, session_id=self._session_id)
            self._session_id = ""
            self._session_context_id = None

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active provider session in place when the bot is interrupted.

        ``CancelSession`` stops synthesis for the active session without dropping
        the connection, so the connection stays reusable for the next turn.

        Args:
            context_id: The audio context that was interrupted.
        """
        if self._session_id:
            await self._send_event(_CANCEL_SESSION, {}, session_id=self._session_id)
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Finish the active session so the provider releases its synthesis.

        Sends ``FinishSession`` for the turn's open session. Turning ``isLast``
        (the SDK's final-request marker) into a separate ``FinishSession`` event
        after the last ``TaskRequest`` is the protocol's sentence/turn close.

        Args:
            context_id: The context whose session to finish, or the active one.
        """
        if not self._session_id:
            return
        await self._send_event(_FINISH_SESSION, {}, session_id=self._session_id)
        self._session_id = ""
        self._session_context_id = None

    async def stop(self, frame: EndFrame):
        """Gracefully finish the session and connection within the flush timeout.

        Finishes the active session (if any) and the connection, then waits up to
        ``flush_timeout`` for the provider's ``SessionFinished`` acknowledgment
        before tearing down. A timeout is reported as a non-fatal error.

        Args:
            frame: The end frame.
        """
        self._disconnecting = True
        try:
            await self._graceful_finish()
        finally:
            await super().stop(frame)

    async def _graceful_finish(self) -> None:
        """Finish the active session and connection, awaiting acknowledgment."""
        if self._session_id:
            await self._send_event(_FINISH_SESSION, {}, session_id=self._session_id)
            self._session_id = ""
            self._session_context_id = None
        elif not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._send_event(_FINISH_CONNECTION, {})
            if self._websocket and self._websocket.state is State.OPEN:
                async with asyncio.timeout(self._flush_timeout):
                    await self._session_finished.wait()
        except TimeoutError as exc:
            await self.push_error(
                "Volcengine TTS: timed out waiting for SessionFinished", exception=exc
            )
        except Exception as exc:
            await self.push_error(f"Volcengine TTS: {exc}", exception=exc)

    async def cancel(self, frame: CancelFrame):
        """Cancel the service immediately, releasing the websocket and tasks.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    def _start_session_payload(self) -> dict:
        """Build the StartSession request payload."""
        return {
            "user": {"uid": self._connect_id},
            "event": _START_SESSION,
            "req_params": {
                "speaker": self._params.speaker,
                "additions": self._params.additions,
                "audio_params": {
                    "format": self._params.audio_format,
                    "sample_rate": self._params.sample_rate,
                },
            },
        }

    async def _report_error(
        self, error: ErrorFrame, force_treat_as_permanent: bool = False
    ) -> None:
        error.error = f"Volcengine TTS (connect {self._connect_id}): {error.error}"
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
                if isinstance(message, str):
                    await self.push_error(f"Volcengine TTS: unexpected text response {message!r}")
                    continue
                event = self._peek_event(message)
                if event in (
                    _SERVER_SESSION_STARTED,
                    _SERVER_SESSION_CANCELED,
                    _SERVER_SESSION_FINISHED,
                    _SERVER_SESSION_FAILED,
                ):
                    if event in (_SERVER_SESSION_FINISHED, _SERVER_SESSION_CANCELED):
                        self._session_finished.set()
                    continue
                try:
                    frames = self._decode_chunk(message)
                except ValueError as exc:
                    session = f" (session {self._session_id})" if self._session_id else ""
                    await self.push_error(f"Volcengine TTS{session}: {exc}", exception=exc)
                    continue
                context_id = self.get_active_audio_context_id()
                for frame in frames:
                    if isinstance(frame, TTSStoppedFrame):
                        await self.append_to_audio_context(context_id, frame)
                        await self.remove_audio_context(context_id)
                    else:
                        await self.append_to_audio_context(context_id, frame)

    def _peek_event(self, message: bytes) -> int | None:
        """Extract the event identifier from a server frame, when present.

        Returns the event identifier carried in the frame header's event field, or
        ``None`` for frames without one (e.g. raw ``AudioOnlyServer``).

        Args:
            message: The raw binary frame.

        Returns:
            The event identifier, or ``None`` when the frame has no event.
        """
        if len(message) < 4 or message[0] != 0x11:
            return None
        flags = message[1] & 0xF
        if not flags & 0x4:  # no WithEvent flag
            return None
        offset = 4 + (4 if flags & 0x3 else 0)  # skip optional sequence field
        if len(message) < offset + 4:
            return None
        return struct.unpack_from(">i", message, offset)[0]

    def _decode_chunk(self, message: bytes) -> list[Frame]:
        """Decode one server frame into a list of TTS frames.

        Mirrors the provider's server-frame parser: after the four-byte header
        come an optional sequence field, an optional event (with its session and
        connect identifiers), an optional error code, and then the length-prefixed
        payload. A raw ``AudioOnlyServer`` frame produces a ``TTSAudioRawFrame``; a
        ``Response`` (event 352) frame base64-decodes its ``data`` member. The
        final flag (the negative-sequence flag) appends a ``TTSStoppedFrame`` to
        close the sentence. An ``Error`` frame raises a descriptive ``ValueError``.

        Args:
            message: The raw binary frame.

        Returns:
            The audio and, when final, stopped frames decoded from the message.
        """
        if len(message) < 4:
            raise ValueError("response header is truncated")
        if message[0] != 0x11:
            raise ValueError("unsupported protocol header")
        message_type = message[1] >> 4
        flags = message[1] & 0xF
        serialization = message[2] >> 4
        compression = message[2] & 0xF
        offset = 4

        if flags & 0x3:  # hasSequenceField (server frames never send AudioOnlyClient)
            offset += 4
        if flags & 0x4:  # WithEvent: event identifier plus session/connect identifiers
            offset = self._skip_event(message, offset)
        error_code = None
        if message_type == 0xF:  # Error
            if len(message) < offset + 4:
                raise ValueError("response error header is truncated")
            error_code = struct.unpack_from(">I", message, offset)[0]
            offset += 4
        if len(message) < offset + 4:
            raise ValueError("response payload header is truncated")
        size = struct.unpack_from(">I", message, offset)[0]
        offset += 4
        if len(message) < offset + size:
            raise ValueError("response payload size does not match its header")
        payload = message[offset : offset + size]

        if message_type == 0xB:  # AudioOnlyServer: raw PCM payload
            return self._chunk_frames(payload, bool(flags & 0x2))

        if message_type == 0xF:
            raise ValueError(
                f"service error {error_code}: {payload.decode('utf-8', errors='replace')}"
            )

        if message_type != 0x9 or serialization != 1:
            raise ValueError("unsupported response message type or serialization")
        if compression != 0:
            raise ValueError("unsupported response compression")
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid JSON response") from exc
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        audio = base64.b64decode(data.get("data", ""))
        return self._chunk_frames(audio, bool(flags & 0x2))

    def _skip_event(self, message: bytes, offset: int) -> int:
        """Advance past an event identifier and its length-prefixed identifiers.

        Args:
            message: The raw binary frame.
            offset: The offset of the event identifier.

        Returns:
            The offset immediately after the event and its identifiers.
        """
        if len(message) < offset + 4:
            raise ValueError("response event header is truncated")
        event = struct.unpack_from(">i", message, offset)[0]
        offset += 4
        if event not in (1, 2, 50, 51, 52):  # hasSessionIDField
            offset = self._skip_length_prefixed(message, offset)
        if event in (50, 51, 52):  # connect identifier
            offset = self._skip_length_prefixed(message, offset)
        return offset

    def _skip_length_prefixed(self, message: bytes, offset: int) -> int:
        """Advance past a big-endian length and its string bytes.

        Args:
            message: The raw binary frame.
            offset: The offset of the length prefix.

        Returns:
            The offset immediately after the length-prefixed string.
        """
        if len(message) < offset + 4:
            raise ValueError("response identifier header is truncated")
        length = struct.unpack_from(">I", message, offset)[0]
        offset += 4 + length
        if len(message) < offset:
            raise ValueError("response identifier is truncated")
        return offset

    def _chunk_frames(self, audio: bytes, final: bool) -> list[Frame]:
        frames: list[Frame] = []
        if audio:
            assert self._params.sample_rate is not None  # resolved in setup()
            frames.append(
                TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self._params.sample_rate,
                    num_channels=1,
                )
            )
        if final:
            frames.append(TTSStoppedFrame(context_id=None))
        return frames
