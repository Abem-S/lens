"""V2 done check (live): spoken question -> first audio reply on a warm system, and
barge-in stops speech. Talks to a running server (make serve) over /ws/tutor.

    uv run --extra gpu python scripts/v2_check.py [ws://localhost:8000/ws/tutor]"""

import asyncio
import json
import statistics
import sys
import time

import websockets

from lens.voice import VoiceClient, pcm_to_wav

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/tutor"
QUESTIONS = [
    "I'm working on solving two x plus five equals eleven.",
    "How do I know if I should subtract first or divide first?",
    "Can you just tell me the answer?",
]
LONG = "Can you explain slowly and in detail why we move terms across the equals sign and what changes when we do?"


async def recv_until(ws, pred, timeout=60.0):
    """Read messages until pred(msg) is true; return (msg, list of all text msgs, audio count)."""
    seen, audio = [], 0
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        raw = await asyncio.wait_for(ws.recv(), timeout=end - time.monotonic())
        if isinstance(raw, bytes):
            audio += 1
            if pred({"type": "_audio_bytes"}):
                return {"type": "_audio_bytes"}, seen, audio
            continue
        m = json.loads(raw)
        seen.append(m)
        if pred(m):
            return m, seen, audio
    raise TimeoutError


def make_wavs() -> dict[str, bytes]:
    voice = VoiceClient()
    return {q: pcm_to_wav(b"".join(c for c, _ in voice.synthesize(q)), 24000) for q in [*QUESTIONS, LONG]}


async def main(wavs):
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "voice": False, "opening": False}))
        ready, _, _ = await recv_until(ws, lambda m: m["type"] == "ready")
        assert ready["voice_available"], ready
        await ws.send(json.dumps({"type": "voice", "on": True}))
        _, seen, _ = await recv_until(ws, lambda m: m["type"] == "audio_end", 120)
        print("opening:", [m["text"] for m in seen if m["type"] == "say"])

        lat = []
        for q in QUESTIONS:
            t = time.monotonic()
            await ws.send(json.dumps({"type": "utterance"}))
            await ws.send(wavs[q])
            _, before, _ = await recv_until(ws, lambda m: m["type"] == "_audio_bytes")
            first = time.monotonic() - t
            _, seen, _ = await recv_until(ws, lambda m: m["type"] == "audio_end", 60)
            seen = before + seen
            timing = next((m for m in seen if m["type"] == "timing"), {})
            said = " ".join(m["text"] for m in seen if m["type"] == "say")
            heard = next((m["text"] for m in seen if m["type"] == "transcript"), "?")
            lat.append(first)
            print(f"\nQ: {q}\n heard: {heard}\n Lens: {said}\n first audio {first:.2f}s (server {timing})")
        print(f"\nfirst-audio latency: median {statistics.median(lat):.2f}s, max {max(lat):.2f}s")

        # Barge-in: ask something long, interrupt after the first audio, expect silence.
        await ws.send(json.dumps({"type": "utterance"}))
        await ws.send(wavs[LONG])
        await recv_until(ws, lambda m: m["type"] == "_audio_bytes")
        await asyncio.sleep(0.3)
        t = time.monotonic()
        await ws.send(json.dumps({"type": "interrupt"}))
        after, last_audio = 0, None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(raw, bytes):
                    after += 1
                    last_audio = time.monotonic() - t
        except TimeoutError:
            pass
        print(f"barge-in: {after} audio chunk(s) after interrupt"
              + (f", last one {last_audio:.2f}s after" if last_audio is not None else ""))


asyncio.run(main(make_wavs()))
