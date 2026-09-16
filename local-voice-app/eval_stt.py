#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replay a shared recording over the application's eval transport and report STT behavior."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import sys
import time
import unicodedata
import wave
from pathlib import Path

from dotenv import dotenv_values
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
VOLCENGINE_STT_SETTINGS = (
    "VOLCENGINE_API_KEY",
    "VOLCENGINE_RESOURCE_ID",
    "VOLCENGINE_STT_OPTIONS",
)


def evaluate_turn(reference: str, events: list[dict]) -> dict:
    """Summarize raw RTVI captions and VAD-to-final arrival latency for one recording.

    Args:
        reference: The recording's reference transcription.
        events: Timestamped raw RTVI messages belonging to this replay.

    Returns:
        Caption counts, normalized character accuracy, and final arrival latency.
    """
    captions = [event for event in events if event["type"] == "user-transcription"]
    finals = [event for event in captions if event["data"].get("final", True)]
    interim = [event for event in captions if not event["data"].get("final", True)]
    stops = [event for event in events if event["type"] == "vad-user-stopped-speaking"]
    speaking = [event for event in events if event["type"] == "bot-started-speaking"]
    preceding_llm_stops = [
        event
        for event in events
        if event["type"] == "bot-llm-stopped" and speaking and event["at"] <= speaking[0]["at"]
    ]
    hypothesis = " ".join(event["data"].get("text", "") for event in finals)
    expected = "".join(
        c for c in unicodedata.normalize("NFKC", reference).casefold() if c.isalnum()
    )
    actual = "".join(c for c in unicodedata.normalize("NFKC", hypothesis).casefold() if c.isalnum())
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        next_row = [i]
        for j, right in enumerate(actual, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (left != right)))
        row = next_row
    error_rate = row[-1] / len(expected) if expected and finals else None
    latency = (finals[-1]["at"] - stops[-1]["at"]) * 1000 if finals and stops else None
    # Time from the end of the reply's LLM text to the first bot audio, the
    # closest the wire gets to the synthesis provider's first packet. A barge-in
    # can deliver the speech event without a matching LLM end, so an absent or
    # out-of-order pair is reported as unmeasured rather than zero.
    ttfb = (
        (speaking[0]["at"] - preceding_llm_stops[-1]["at"]) * 1000 if preceding_llm_stops else None
    )
    return {
        "status": "passed" if finals and interim else "failed",
        "reference": reference,
        "transcript": hypothesis,
        "final_segment_count": len(finals),
        "exact_match": bool(finals) and expected == actual,
        "interim_count": len(interim),
        "character_error_rate": error_rate,
        "character_accuracy": max(0.0, 1 - error_rate) if error_rate is not None else None,
        "final_latency_ms": round(latency, 3) if latency is not None and latency >= 0 else None,
        "latency_status": "measured" if latency is not None and latency >= 0 else "unavailable",
        "ttfb_ms": round(ttfb, 3) if ttfb is not None and ttfb >= 0 else None,
    }


async def replay_recording(
    bot_url: str,
    audio_path: Path,
    reference: str,
    *,
    timeout: float = 30,
    silence: float = 2,
) -> dict:
    """Replay a recording after silence, then repeat it while the bot speaks.

    Args:
        bot_url: A running application's eval WebSocket URL.
        audio_path: Mono 16-bit PCM WAV shared by both providers.
        reference: Reference words spoken in the recording.
        timeout: Maximum wait for each required bot event, in seconds.
        silence: Silence before the first recording and after the second.

    Returns:
        Raw caption evidence, turn measurements, and behavior checks. Teardown
        uses eval-cancel; it does not claim to exercise graceful EndFrame flushing.
    """
    import pipecat.processors.frameworks.rtvi.models as rtvi
    from pipecat.evals.serializer import EvalClientSerializer
    from pipecat.frames.frames import OutputAudioRawFrame

    def read_audio() -> tuple[bytes, int]:
        with wave.open(str(audio_path), "rb") as recording:
            if recording.getnchannels() != 1 or recording.getsampwidth() != 2:
                raise ValueError("Recording must be mono 16-bit PCM WAV")
            return recording.readframes(recording.getnframes()), recording.getframerate()

    audio, sample_rate = await asyncio.to_thread(read_audio)
    chunk_bytes = int(sample_rate * 0.04) * 2
    events: list[dict] = []
    pending: asyncio.Queue[bytes] = asyncio.Queue()
    changed = asyncio.Condition()
    serializer = EvalClientSerializer()
    origin = time.monotonic()
    turns: list[dict] = []
    checks: dict[str, bool] = {}
    first: int | None = None
    second: int | None = None
    waiting_for = "bot-ready"
    failure: str | None = None

    async with websocket_connect(bot_url, open_timeout=timeout) as websocket:

        async def receive() -> None:
            async for payload in websocket:
                message = json.loads(payload)
                if message.get("type") in {
                    "bot-ready",
                    "user-transcription",
                    "vad-user-started-speaking",
                    "vad-user-stopped-speaking",
                    "bot-started-speaking",
                    "bot-stopped-speaking",
                    "bot-llm-stopped",
                    "bot-interrupted",
                    "error",
                }:
                    message["at"] = time.monotonic() - origin
                    if message["type"] == "error":
                        message["data"] = {
                            "message": "Bot reported an error; inspect backend locally"
                        }
                    async with changed:
                        events.append(message)
                        changed.notify_all()

        async def wait_event(kind: str, since: int = 0, *, final: bool = False) -> dict:
            nonlocal waiting_for
            waiting_for = kind
            async with asyncio.timeout(timeout), changed:
                while True:
                    for event in events[since:]:
                        if event["type"] == kind and (
                            not final or event.get("data", {}).get("final", True)
                        ):
                            return event
                    await changed.wait()

        async def wait_bot_speaking(since: int) -> None:
            nonlocal waiting_for
            waiting_for = "active bot speech after the recording"
            async with asyncio.timeout(timeout), changed:
                while True:
                    speech = [
                        event
                        for event in events[since:]
                        if event["type"]
                        in {"bot-started-speaking", "bot-stopped-speaking", "bot-interrupted"}
                    ]
                    if speech and speech[-1]["type"] == "bot-started-speaking":
                        return
                    await changed.wait()

        async def send_audio() -> None:
            next_send = time.monotonic()
            while True:
                queued = not pending.empty()
                chunk = pending.get_nowait() if queued else bytes(chunk_bytes)
                wire = await serializer.serialize(OutputAudioRawFrame(chunk, sample_rate, 1))
                if wire is not None:
                    try:
                        await websocket.send(wire)
                    except ConnectionClosed:
                        checks["connection_remained_open"] = False
                        return
                await asyncio.sleep(max(0, next_send + 0.04 - time.monotonic()))
                next_send = max(next_send + 0.04, time.monotonic())
                if queued:
                    pending.task_done()

        async def play() -> None:
            for offset in range(0, len(audio), chunk_bytes):
                pending.put_nowait(audio[offset : offset + chunk_bytes].ljust(chunk_bytes, b"\0"))
            async with asyncio.timeout(timeout):
                await pending.join()

        async with asyncio.TaskGroup() as group:
            reader = group.create_task(receive())
            sender = None
            try:
                await websocket.send(
                    rtvi.Message(
                        type="client-ready",
                        id="stt-eval-ready",
                        data=rtvi.ClientReadyData(
                            version=rtvi.PROTOCOL_VERSION,
                            about=rtvi.AboutClientData(library="pipecat"),
                        ).model_dump(),
                    ).model_dump_json()
                )
                await wait_event("bot-ready")
                await websocket.send(
                    rtvi.Message(
                        type="client-message",
                        id="stt-eval-configure",
                        data={"t": "eval-configure", "d": {"vad_user_speaking": True}},
                    ).model_dump_json()
                )
                sender = group.create_task(send_audio())
                await wait_event("bot-started-speaking")
                await wait_event("bot-stopped-speaking")
                silence_start = len(events)
                await asyncio.sleep(silence)
                checks["silence_has_no_transcription"] = not any(
                    event["type"] == "user-transcription" for event in events[silence_start:]
                )
                first = len(events)
                await play()
                await wait_event("user-transcription", first, final=True)
                await wait_bot_speaking(first)
                second = len(events)
                checks["barge_in_started_during_bot_speech"] = (
                    next(
                        event["type"]
                        for event in reversed(events)
                        if event["type"] in {"bot-started-speaking", "bot-stopped-speaking"}
                    )
                    == "bot-started-speaking"
                )
                await play()
                await wait_event("user-transcription", second, final=True)
                await asyncio.sleep(silence)
                checks["bot_interrupted"] = any(
                    event["type"] == "bot-interrupted" for event in events[second:]
                )
            except TimeoutError:
                checks["required_events_within_timeout"] = False
                failure = f"Timed out waiting for {waiting_for}"
            finally:
                if sender:
                    sender.cancel()
                try:
                    await websocket.send(
                        rtvi.Message(
                            type="client-message",
                            id="stt-eval-cancel",
                            data={"t": "eval-cancel"},
                        ).model_dump_json()
                    )
                    await asyncio.wait_for(websocket.wait_closed(), timeout=5)
                    checks["eval_cancel_closed_connection"] = True
                except (TimeoutError, ConnectionClosed):
                    checks["eval_cancel_closed_connection"] = False
                reader.cancel()
    if first is not None:
        turns = [evaluate_turn(reference, events[first:second])]
        if second is not None:
            turns.append(evaluate_turn(reference, events[second:]))
        checks["resumed_after_silence"] = bool(turns[0]["final_segment_count"])
    checks["interim_and_final_both_turns"] = len(turns) == 2 and all(
        turn["status"] == "passed" for turn in turns
    )
    checks["reference_matches_both_turns"] = len(turns) == 2 and all(
        turn["exact_match"] for turn in turns
    )
    checks["no_bot_errors"] = not any(event["type"] == "error" for event in events)
    return {
        "status": "passed" if turns and all(checks.values()) else "failed",
        "checks": checks,
        "turns": turns,
        "events": events,
        "reason": failure,
    }


async def run_provider(provider: str, args: argparse.Namespace, config: dict[str, str]) -> dict:
    """Start one configured application, replay the recording, and reap its process.

    Args:
        provider: Recognition provider selected for this process.
        args: Parsed evaluation arguments.
        config: Backend environment, never included in the report.

    Returns:
        A measured result, or an explicit skipped/failed result.
    """
    required = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "DEEPSEEK_API_KEY"]
    if provider == "volcengine":
        required.extend(("VOLCENGINE_API_KEY", "VOLCENGINE_RESOURCE_ID"))
    missing = [key for key in required if not config.get(key, "").strip()]
    if missing:
        return {
            "provider": provider,
            "status": "skipped",
            "reason": "Missing " + ", ".join(missing),
        }
    # Fail before spawning if another application owns the selected local port.
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError:
        return {"provider": provider, "status": "failed", "reason": "Evaluation port unavailable"}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(ROOT / "local-voice-app" / "bot.py"),
        "-t",
        "eval",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        cwd=ROOT,
        env=_provider_environment(provider, config),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    result: dict = {"provider": provider, "status": "failed"}
    try:
        async with asyncio.timeout(args.timeout * 8 + 30):
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.returncode is not None:
                    raise RuntimeError("Application exited before accepting a connection")
                try:
                    _, writer = await asyncio.open_connection("127.0.0.1", args.port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.1)
            else:
                raise TimeoutError("Application did not start")
            result.update(
                await replay_recording(
                    f"ws://127.0.0.1:{args.port}",
                    args.audio,
                    args.reference,
                    timeout=args.timeout,
                )
            )
    except Exception as error:
        # Provider exceptions can contain request URLs or headers; do not persist them.
        result["reason"] = f"Evaluation could not complete ({type(error).__name__})"
    finally:
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
    result["application_exited"] = process.returncode == 0
    if not result["application_exited"]:
        result["status"] = "failed"
    return result


def _provider_environment(provider: str, config: dict[str, str]) -> dict[str, str]:
    """Build an isolated backend environment for one recognition provider."""
    environment = config.copy()
    if provider == "elevenlabs":
        for name in VOLCENGINE_STT_SETTINGS:
            environment.pop(name, None)
    return environment


def main() -> int:
    """Run selected providers and write a reproducible, credential-free JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["both", "elevenlabs", "volcengine"], default="both")
    parser.add_argument(
        "--audio", type=Path, default=ROOT / "scripts/release-evals/assets/capital_question.wav"
    )
    parser.add_argument("--reference", default="What is the capital of Germany?")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/stt-eval/report.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    args.audio = args.audio.resolve()
    config = {
        key: value for key, value in dotenv_values(args.env_file).items() if value is not None
    }
    config.update(os.environ)
    providers = ["elevenlabs", "volcengine"] if args.provider == "both" else [args.provider]

    async def run() -> list[dict]:
        return [await run_provider(provider, args, config) for provider in providers]

    report = {
        "audio": str(args.audio.relative_to(ROOT))
        if args.audio.is_relative_to(ROOT)
        else str(args.audio),
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "reference": args.reference,
        "accuracy_method": "NFKC, casefold, alphanumeric characters; Levenshtein CER; accuracy=max(0,1-CER)",
        "latency_method": "Last final caption arrival minus last raw VAD stop arrival, same replay; unavailable if absent or negative",
        "ttfb_method": "First bot-started-speaking arrival minus last bot-llm-stopped arrival, same replay; null if absent or negative",
        "second_pass": "disabled; verified by deterministic provider protocol tests",
        "teardown": "eval-cancel and process exit; graceful end-of-stream is covered by service tests",
        "results": asyncio.run(run()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    for result in report["results"]:
        print(f"{result['provider']}: {result['status']}")
    print(f"Report: {args.output}")
    return 1 if any(result["status"] == "failed" for result in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
