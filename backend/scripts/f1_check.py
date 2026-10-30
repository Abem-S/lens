"""F1 done check (live): a full tutor session against a running server.

The student writes a correct line, then a wrong one: Lens outlines it and asks a
question. The student points at it and asks "is this line right?" (a real hand
photo pasted on the page; SAM finds the fingertip). The student fixes the line:
Lens says so and moves on. Nothing spoken may give the answer away.

    uv run --extra gpu python scripts/f1_check.py [ws://localhost:8000/ws/tutor]
Needs .cache/eval/hands/one_3.jpg (see scripts/c1_check.py)."""

import asyncio
import io
import json
import sys
import time

import numpy as np
import websockets
from PIL import Image, ImageDraw, ImageFont

from lens.config import REPO_ROOT
from lens.sam import SamClient
from lens.voice import VoiceClient, pcm_to_wav

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/tutor"
W, H = 1200, 900
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 54)
LINE_Y = [140, 260, 380]


def page(lines: list[str]) -> Image.Image:
    im = Image.new("RGB", (W, H), (246, 244, 236))
    d = ImageDraw.Draw(im)
    for y in range(100, H, 60):
        d.line([(0, y), (W, y)], fill=(195, 210, 228), width=2)
    d.text((120, 40), "Solve 2x + 5 = 11", fill=(30, 40, 60), font=FONT)
    for y, text in zip(LINE_Y, lines):
        d.text((120, y), text, fill=(30, 40, 60), font=FONT)
    return im


def jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def with_hand(im: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Cut a real pointing hand out of a photo with SAM and put its fingertip at target."""
    src = Image.open(REPO_ROOT / ".cache/eval/hands/one_3.jpg").convert("RGB")
    masks = SamClient().segment(jpeg(src), text="hand", max_masks=1)
    mask = masks[0].mask
    ys, xs = np.nonzero(mask)
    tip = (int(xs[np.argmin(ys)]), int(ys.min()))  # finger points up in this photo
    scale = 1.6
    src = src.resize((int(src.width * scale), int(src.height * scale)))
    alpha = Image.fromarray((mask * 255).astype("uint8")).resize(src.size)
    out = im.copy()
    out.paste(src, (target[0] - int(tip[0] * scale), target[1] - int(tip[1] * scale)), alpha)
    return out


async def collect(ws, until: str, timeout: float = 240.0):
    msgs, audio = [], 0
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        raw = await asyncio.wait_for(ws.recv(), timeout=end - time.monotonic())
        if isinstance(raw, bytes):
            audio += 1
            continue
        m = json.loads(raw)
        msgs.append(m)
        if m["type"] == until:
            break
    return msgs, audio


def show(label, msgs, audio):
    said = [m["text"] for m in msgs if m["type"] == "say"]
    res = next((m for m in msgs if m["type"] == "result"), None)
    errs = [m["message"] for m in msgs if m["type"] == "error"]
    print(f"\n== {label}")
    if res:
        f = res["finding"]
        print("   lines:", [(s["latex"], s["status"]) for s in res["page"]["steps"]])
        print("   finding:", None if not f else (f["step_id"], f["kind"], f"outline {len(f['mask'] or [])} pts"))
    for m in msgs:
        if m["type"] in ("transcript", "pointing"):
            print("  ", m)
    print("   Lens said:", said, f"({audio} audio chunks)")
    if errs:
        print("   ERRORS:", errs)
    return res, said


async def main(question_wav: bytes):
    pages = [
        page(["2x + 5 = 11"]),
        page(["2x + 5 = 11", "2x = 11 + 5"]),
        page(["2x + 5 = 11", "2x = 11 - 5", "2x = 6"]),
    ]
    pointed = with_hand(pages[1], (300, LINE_Y[1] + 70))
    pointed.save(REPO_ROOT / ".cache/eval/runs/f1_pointing.jpg")
    all_said = []
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "voice": False, "opening": False}))
        await collect(ws, "ready")
        await ws.send(json.dumps({"type": "voice", "on": True}))
        msgs, audio = await collect(ws, "audio_end")
        all_said += show("opening", msgs, audio)[1]

        for label, im in [("writes a correct line", pages[0]), ("writes a wrong line", pages[1])]:
            await ws.send(json.dumps({"type": "check", "reason": "new_ink"}))
            await ws.send(jpeg(im))
            msgs, audio = await collect(ws, "result")
            if msgs[-1]["finding"]:
                more, a2 = await collect(ws, "audio_end")
                msgs, audio = msgs + more, audio + a2
            all_said += show(label, msgs, audio)[1]

        await ws.send(json.dumps({"type": "frame"}))
        await ws.send(jpeg(pointed))
        await ws.send(json.dumps({"type": "utterance"}))
        await ws.send(question_wav)
        msgs, audio = await collect(ws, "audio_end")
        all_said += show('points and asks "is this line right?"', msgs, audio)[1]

        await ws.send(json.dumps({"type": "check", "reason": "new_ink"}))
        await ws.send(jpeg(pages[2]))
        msgs, audio = await collect(ws, "audio_end")
        all_said += show("fixes the line", msgs, audio)[1]

    print("\nEverything Lens said:")
    for s in all_said:
        print("  -", s)


if __name__ == "__main__":
    wav = pcm_to_wav(b"".join(c for c, _ in VoiceClient().synthesize("Is this line right?")), 24000)
    asyncio.run(main(wav))
