"""V1 voice services on Modal: Omnilingual ASR (speech -> text) and Kokoro (text -> speech).

Deploy:  cd backend && uv run --extra gpu modal deploy modal_apps/voice.py
Client:  lens.voice

Both models share one small GPU container, so a session keeps a single container warm.
"""

from __future__ import annotations

import io
import time

import modal

APP_NAME = "muna-lens-voice"
ASR_CARD = "omniASR_CTC_1B"  # backup: omniASR_CTC_300M
KOKORO_VOICE = "af_heart"
SAMPLE_RATE_OUT = 24000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "espeak-ng")
    # fairseq2n 0.6 is built against torch 2.8.0 exactly; torchaudio must match it.
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", "omnilingual-asr==0.1.0", "kokoro>=0.9.4", "soundfile", "numpy",
                 "pyarrow==20.0.0")  # kokoro keeps NumPy 1.26; newer pyarrow needs NumPy 2
)
# Set at container start, not in the image: build steps would otherwise create
# these folders inside the image and the volume could not mount over them.
CACHE_ENV = {"FAIRSEQ2_CACHE_DIR": "/vcache/fairseq2", "HF_HOME": "/vcache/hf", "XDG_CACHE_HOME": "/vcache/xdg"}
cache = modal.Volume.from_name("muna-lens-voice-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(image=image, gpu=["L4", "A10G", "T4"], volumes={"/vcache": cache}, scaledown_window=120, timeout=600)
class Voice:
    @modal.enter()
    def load(self) -> None:
        import os

        os.environ.update(CACHE_ENV)
        import numpy as np
        from kokoro import KPipeline
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        t = time.monotonic()
        self.asr = ASRInferencePipeline(model_card=ASR_CARD)
        self.tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        cache.commit()
        # Warm both paths once so the first real request is not slow.
        self.asr.transcribe([{"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}])
        list(self.tts("Hello.", voice=KOKORO_VOICE))
        self.load_seconds = time.monotonic() - t

    @modal.method()
    def transcribe(self, wav: bytes) -> dict:
        """WAV (any rate, mono or stereo) -> text. Clips longer than 40 s are refused by the model."""
        import numpy as np
        import soundfile as sf

        t = time.monotonic()
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text = self.asr.transcribe([{"waveform": np.ascontiguousarray(audio), "sample_rate": int(sr)}])[0]
        return {"text": text.strip(), "seconds": time.monotonic() - t, "audio_seconds": len(audio) / sr}

    @modal.method(is_generator=True)
    def synthesize(self, text: str, voice: str = KOKORO_VOICE, speed: float = 1.0):
        """Text -> 16-bit mono PCM chunks at 24 kHz, one per sentence-sized piece, as they are ready."""
        import numpy as np

        t = time.monotonic()
        for i, result in enumerate(self.tts(text, voice=voice, speed=speed)):
            pcm = (np.clip(result.audio.numpy(), -1, 1) * 32767).astype("<i2").tobytes()
            yield {"index": i, "pcm": pcm, "sample_rate": SAMPLE_RATE_OUT, "seconds": time.monotonic() - t}

    @modal.method()
    def ping(self) -> dict:
        return {"ok": True, "load_seconds": self.load_seconds}
