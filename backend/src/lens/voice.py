"""Client for the V1 voice services on Modal (modal_apps/voice.py)."""

from __future__ import annotations

import io
import logging
import time
import wave
from collections.abc import Iterator

log = logging.getLogger("lens.voice")
APP_NAME = "muna-lens-voice"


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class VoiceClient:
    def __init__(self) -> None:
        import modal

        self._obj = modal.Cls.from_name(APP_NAME, "Voice")()

    def keep_warm(self, on: bool) -> None:
        self._obj.update_autoscaler(min_containers=1 if on else 0)

    def transcribe(self, wav: bytes) -> str:
        t = time.monotonic()
        out = self._obj.transcribe.remote(wav)
        log.info("asr %.1fs audio -> %d chars gpu=%.2fs total=%.2fs", out["audio_seconds"], len(out["text"]),
                 out["seconds"], time.monotonic() - t)
        return out["text"]

    def synthesize(self, text: str) -> Iterator[tuple[bytes, int]]:
        """Yield (pcm16 mono, sample_rate) chunks as Kokoro produces them."""
        t = time.monotonic()
        for i, chunk in enumerate(self._obj.synthesize.remote_gen(text)):
            if i == 0:
                log.info("tts first audio %.2fs (gpu %.2fs) for %d chars", time.monotonic() - t, chunk["seconds"],
                         len(text))
            yield chunk["pcm"], chunk["sample_rate"]
