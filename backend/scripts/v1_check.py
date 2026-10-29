"""V1 done check (live, costs GPU time): a sentence synthesises, a ~5 s utterance
transcribes; both latencies logged. The utterance is Kokoro's own speech, so the
round trip also shows the two services agree.

    uv run --extra gpu python scripts/v1_check.py"""

import logging
import time

from lens.voice import VoiceClient, pcm_to_wav

SENTENCE = "When you moved the five to the other side, what happened to its sign?"
UTTERANCE = "I think the derivative of x squared is two x, so this line should be fine."

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
for noisy in ("httpx", "httpx2", "hpack", "grpclib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

v = VoiceClient()
t = time.monotonic()
v._obj.ping.remote()
print(f"container ready in {time.monotonic() - t:.1f}s")

for label, text in [("hint", SENTENCE), ("utterance", UTTERANCE)]:
    t = time.monotonic()
    chunks = list(v.synthesize(text))
    pcm = b"".join(c for c, _ in chunks)
    sr = chunks[0][1]
    print(f"tts {label}: {len(pcm) / 2 / sr:.1f}s of audio in {time.monotonic() - t:.2f}s")
    if label == "utterance":
        wav = pcm_to_wav(pcm, sr)
        for run in range(3):
            t = time.monotonic()
            text_out = v.transcribe(wav)
            print(f"asr run {run}: {time.monotonic() - t:.2f}s -> {text_out!r}")
