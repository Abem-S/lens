"""Live tracking check (costs a little GPU time): stream a clip through the app's /ws/track
as the browser does, and report latency, frame rate, events and outline following.

    uv run --extra gpu python scripts/track_check.py CLIP_DIR [--app ws://localhost:8000] [--fps 5]

CLIP_DIR holds f0000.jpg... (e.g. from scripts/make_track_clip.py, with truth.json) or
frames cut from a real video. Without --app, it relays in-process (same code as the app).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

MAX_IN_FLIGHT = 2  # same as the browser (frontend/src/lib/tracking.ts)
CHECK_DELAY_FRAMES = 10  # the finding comes back ~2 s after the frame it was read from


def shrink(jpeg: bytes, width: int) -> bytes:
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=65)
    return buf.getvalue()


async def stream(ws_send, ws_recv, frames: list[bytes], fps: float, follow: tuple[int, list] | None) -> list[dict]:
    """Offer frames at `fps`; like the browser, keep at most one in flight (skip the rest),
    so latency stays at one round trip + GPU time even on a slow uplink."""
    sent: dict[int, float] = {}
    got: list[dict] = []
    done = asyncio.Event()
    inflight: list[int] = []

    async def reader():
        while True:
            msg = await ws_recv()
            if msg.get("type") == "track":
                msg["latency_ms"] = (time.monotonic() - sent[msg["seq"]]) * 1000
                got.append(msg)
                inflight[:] = [k for k in inflight if k > msg["seq"]]
                if msg["seq"] == max(sent) and max(sent) >= len(frames) - 3:
                    done.set()
            elif msg.get("type") == "unavailable":
                print("unavailable:", msg.get("reason"))
                done.set()
                return

    task = asyncio.create_task(reader())
    t0 = time.monotonic()
    for i, f in enumerate(frames):
        if follow and i == follow[0] + CHECK_DELAY_FRAMES:
            # A finding arrives a few seconds after its check frame; the outline starts from that frame.
            anchor = max((k for k in sent if k <= follow[0]), default=None)
            await ws_send({"type": "follow", "key": "finding", "box": follow[1], "seq": anchor})
        inflight[:] = [k for k in inflight if time.monotonic() - sent[k] < 2.0]
        if len(inflight) < MAX_IN_FLIGHT:
            inflight.append(i)
            sent[i] = time.monotonic()
            await ws_send(i.to_bytes(4, "big") + f)
        await asyncio.sleep(max(0.0, t0 + (i + 1) / fps - time.monotonic()))
    try:
        await asyncio.wait_for(done.wait(), timeout=15)
    except TimeoutError:
        pass
    task.cancel()
    return got


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--app", help="app base URL, e.g. ws://localhost:8000 (default: relay in-process)")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--width", type=int, default=512, help="re-encode frames this wide, as the browser sends them")
    a = ap.parse_args()
    clip = Path(a.clip)
    frames = [shrink(p.read_bytes(), a.width) for p in sorted(clip.glob("f*.jpg"))][: a.limit]
    truth = json.loads((clip / "truth.json").read_text())["frames"] if (clip / "truth.json").exists() else None
    follow = None
    if truth:
        f = next((f for f in truth[: len(frames)] if f["lines_done"] and f["phase"] == "clear"), None)
        follow = (f["frame"], f["lines_done"][-1]) if f else None  # box in that frame

    import websockets

    t = time.monotonic()
    if a.app:
        ws = await websockets.connect(f"{a.app.rstrip('/')}/ws/track?id=check", max_size=2**22, open_timeout=180)

        async def send(m):
            await ws.send(m if isinstance(m, bytes) else json.dumps(m))

        async def recv():
            return json.loads(await ws.recv())
    else:
        from lens.tracking import endpoint_url, track_token

        url = await asyncio.to_thread(endpoint_url)
        ws = await websockets.connect(f"{url}?token={track_token()}", max_size=2**22, open_timeout=180)
        from lens.tracking import TrackEvents, parse_objects

        events, start = TrackEvents(), time.monotonic()

        async def send(m):
            await ws.send(m if isinstance(m, bytes) else json.dumps(m))

        async def recv():
            msg = json.loads(await ws.recv())
            if msg.get("type") == "tracks":
                out = events.push(time.monotonic() - start, parse_objects(msg["objects"]))
                msg = {**msg, **out, "type": "track"}
            return msg

    first = await recv()
    print(f"connected + {first.get('type')} on {first.get('device')} in {time.monotonic() - t:.1f}s (cold start if > 5 s)")
    got = await stream(send, recv, frames, a.fps, follow)
    await ws.close()
    if not got:
        print("no results")
        return
    lat = sorted(m["latency_ms"] for m in got[3:]) or [0]
    gpu = sorted(m.get("gpu_ms") or 0 for m in got[3:]) or [0]
    span = (len(frames) - 1) / a.fps
    print(f"offered {len(frames)} frames at {a.fps} fps; sent and processed {len(got)} "
          f"({len(got) / max(span, 1e-6):.1f} fps); mean frame {sum(map(len, frames)) / len(frames) / 1000:.0f} kB")
    print(f"frame -> result latency ms: median {statistics.median(lat):.0f}, p90 {lat[int(0.9 * len(lat))]:.0f}; "
          f"GPU ms median {statistics.median(gpu):.0f}")
    for m in got:
        for e in m["events"]:
            if e["type"] == "line_done":
                tr = truth[m["seq"]] if truth else None
                print(f"  frame {m['seq']}: line_done {e['box']}" + (f"  (truth: {tr['phase']}, line {tr['line']})" if tr else ""))
    phases = [m["phase"] for m in got]
    print("phases:", {p: phases.count(p) for p in set(phases)})
    fl = [(m["seq"], m["follow"].get("finding")) for m in got if m.get("follow")]
    if fl and truth and follow:
        k = len(truth[follow[0]]["lines_done"]) - 1
        errs = []
        for seq, f in fl:
            tb = truth[seq]["lines_done"][k] if f and len(truth[seq]["lines_done"]) > k else None
            if tb:
                errs.append(abs(f["box"][0] + f["box"][2] / 2 - tb[0] - tb[2] / 2) + abs(f["box"][1] + f["box"][3] / 2 - tb[1] - tb[3] / 2))
        if errs:
            print(f"outline follow: centre error median {statistics.median(errs):.3f}, max {max(errs):.3f} over {len(errs)} frames")


if __name__ == "__main__":
    asyncio.run(main())
