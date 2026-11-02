"""Event logic over tracked hands and pens (no GPU: synthetic masks)."""

import base64
from pathlib import Path

import numpy as np

from lens.tracking import EventConfig, Tracked, TrackEvents, TrackHub, box_cover, parse_objects, pen_tip, pointer_of

W, H = 160, 120


def rect(x0, y0, x1, y1) -> np.ndarray:
    """Bool mask from normalised corners."""
    m = np.zeros((H, W), bool)
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
    return m


def box_of(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return (xs.min() / W, ys.min() / H, (xs.max() + 1 - xs.min()) / W, (ys.max() + 1 - ys.min()) / H)


def hand_and_pen(tip_x: float, tip_y: float) -> list[Tracked]:
    """A hand coming in from the lower right, holding a pen whose tip is at (tip_x, tip_y)."""
    pen = np.zeros((H, W), bool)
    for k in range(20):  # a diagonal pen from the tip up-right to the hand
        x, y = int(tip_x * W) + k, int(tip_y * H) + k // 2
        if 0 <= x < W and 0 <= y < H:
            pen[max(0, y - 1):y + 1, x:x + 2] = True
    hand = rect(min(1, tip_x + 0.11), min(1, tip_y + 0.04), 1.0, 1.0)
    return [Tracked("hand", 1, 0.9, box_of(hand), hand), Tracked("pen", 2, 0.9, box_of(pen), pen)]


def test_pen_tip_is_the_end_away_from_the_hand():
    objs = hand_and_pen(0.4, 0.5)
    tip = pen_tip(objs[1], [objs[0]])
    assert abs(tip[0] - 0.4) < 0.02 and abs(tip[1] - 0.5) < 0.03


def test_pointer_prefers_pen_then_the_deepest_finger():
    p, kind = pointer_of(hand_and_pen(0.3, 0.4))
    assert kind == "pen" and abs(p[0] - 0.3) < 0.02
    # Two hands hold a notebook at the bottom edge; one finger reaches up to a line.
    left = rect(0.0, 0.85, 0.15, 1.0)
    pointing = rect(0.5, 0.45, 0.53, 1.0)
    objs = [Tracked("hand", 1, 0.9, box_of(left), left), Tracked("hand", 2, 0.9, box_of(pointing), pointing)]
    p, kind = pointer_of(objs)
    assert kind == "finger" and p[1] < 0.5


def test_box_cover():
    hand = rect(0.5, 0.5, 1.0, 1.0)
    objs = [Tracked("hand", 1, 0.9, box_of(hand), hand)]
    assert box_cover(objs, (0.1, 0.1, 0.2, 0.1)) == 0.0
    assert box_cover(objs, (0.6, 0.6, 0.2, 0.1)) == 1.0


def run(frames, fps=5.0, cfg=None):
    ev = TrackEvents(cfg or EventConfig())
    return [ev.push(i / fps, objs) for i, objs in enumerate(frames)]


def writing(x0, x1, y, n):
    return [hand_and_pen(x0 + (x1 - x0) * k / (n - 1), y) for k in range(n)]


def events(outs, kind):
    return [(i, e) for i, o in enumerate(outs) for e in o["events"] if e["type"] == kind]


def test_line_done_after_writing_and_hand_leaves():
    frames = writing(0.2, 0.6, 0.3, 12) + [[]] * 8  # write ~2.4 s, then the hand leaves
    outs = run(frames)
    assert outs[5]["phase"] == "writing"
    assert outs[-1]["phase"] == "clear"
    done = events(outs, "line_done")
    assert len(done) == 1
    i, e = done[0]
    assert i >= 12  # only after the hand is gone, and after it settled
    x, y, w, h = e["box"]
    assert x < 0.2 and x + w > 0.6 and y < 0.3 < y + h and h < 0.1


def test_no_check_while_the_hand_covers_the_new_line():
    cover = rect(0.1, 0.2, 0.8, 0.45)  # the hand rests on the line just written
    still = [Tracked("hand", 1, 0.9, box_of(cover), cover)]
    frames = writing(0.2, 0.6, 0.3, 12) + [still] * 10
    outs = run(frames)
    assert events(outs, "line_done") == []
    outs = run(frames + [[]] * 6)  # then it moves away
    assert len(events(outs, "line_done")) == 1


def test_resting_pen_beside_the_line_still_triggers():
    rest = hand_and_pen(0.7, 0.55)  # pen parked below and right of the line, line visible
    frames = writing(0.2, 0.6, 0.3, 12) + [rest] * 8
    outs = run(frames)
    assert outs[-1]["phase"] == "paused"
    assert len(events(outs, "line_done")) == 1


def test_a_fidget_is_not_a_line():
    frames = writing(0.4, 0.45, 0.3, 3) + [[]] * 10
    assert events(run(frames), "line_done") == []


def test_hands_holding_the_notebook_are_not_pointing():
    left, right = rect(0.0, 0.8, 0.12, 1.0), rect(0.88, 0.8, 1.0, 1.0)
    objs = [Tracked("hand", 1, 0.9, box_of(left), left), Tracked("hand", 2, 0.9, box_of(right), right)]
    outs = run([objs] * 6)
    assert all(o["phase"] == "clear" for o in outs)
    assert events(outs, "line_done") == []


def test_hub_prefers_a_fresh_tracked_pointer():
    hub = TrackHub(max_age_s=10)
    asked = []
    locate = hub.locator("t1", lambda img: asked.append(img) or (0.1, 0.1))
    assert locate(b"img") == (0.1, 0.1) and asked == [b"img"]  # nothing tracked yet: SAM still frame
    hub.update("t1", [0.5, 0.4])
    assert locate(b"img") == (0.5, 0.4) and len(asked) == 1
    assert hub.locator(None, None) is None


def _settings():
    from lens.config import Settings

    return Settings(baseten_key=None, spark_base_url="", spark_model="", spark_price_in=None, spark_price_out=None,
                    public_url=None, cache_dir=Path("/tmp"))


def test_track_socket_says_unavailable_without_modal(monkeypatch):
    from fastapi.testclient import TestClient

    from lens.app import create_app

    monkeypatch.delenv("LENS_TRACK_URL", raising=False)
    with TestClient(create_app(_settings())).websocket_connect("/ws/track?id=a") as ws:
        assert ws.receive_json()["type"] == "unavailable"


def test_relay_turns_gpu_tracks_into_events(monkeypatch):
    """A fake tracking service: replies to each frame with a pen writing across a line, then nothing."""
    import asyncio
    import json
    import threading

    import websockets
    from fastapi.testclient import TestClient

    from lens.app import create_app

    def pen_frame(i: int) -> list[dict]:
        if i >= 12:
            return []
        out = []
        for o in hand_and_pen(0.2 + 0.4 * i / 11, 0.3):
            bits = base64.b64encode(np.packbits(o.mask).tobytes()).decode()
            out.append({"concept": o.concept, "id": o.id, "score": o.score, "box": list(o.box),
                        "mask": {"w": W, "h": H, "bits": bits}})
        return out

    async def fake(ws):
        await ws.send(json.dumps({"type": "ready", "device": "fake", "load_s": 0}))
        async for msg in ws:
            if isinstance(msg, bytes):
                seq = int.from_bytes(msg[:4], "big")
                await ws.send(json.dumps({"type": "tracks", "seq": seq, "objects": pen_frame(seq), "follow": {},
                                          "gpu_ms": 1, "wall_ms": 1}))

    ready = threading.Event()
    port: list[int] = []
    loop = asyncio.new_event_loop()

    async def serve():
        async with websockets.serve(fake, "127.0.0.1", 0) as server:
            port.append(server.sockets[0].getsockname()[1])
            ready.set()
            await asyncio.Future()

    threading.Thread(target=lambda: loop.run_until_complete(serve()), daemon=True).start()
    assert ready.wait(5)
    monkeypatch.setenv("LENS_TRACK_URL", f"http://127.0.0.1:{port[0]}")
    monkeypatch.setattr("lens.tracking._now", _Clock())
    phases, done = [], []
    with TestClient(create_app(_settings())).websocket_connect("/ws/track?id=t") as ws:
        assert ws.receive_json()["type"] == "ready"
        for i in range(22):
            ws.send_bytes(i.to_bytes(4, "big") + b"jpeg")
            msg = ws.receive_json()
            assert msg["type"] == "track" and msg["seq"] == i
            phases.append(msg["phase"])
            done += [e for e in msg["events"] if e["type"] == "line_done"]
    assert "writing" in phases and phases[-1] == "clear"
    assert len(done) == 1


class _Clock:
    """monotonic() that advances 0.2 s per call: frames arrive at 5 fps in the relay test."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.2
        return self.t


def test_parse_objects_round_trip():
    m = rect(0.25, 0.25, 0.5, 0.5)
    o = {"concept": "hand", "id": 3, "score": 0.8, "box": [0.25, 0.25, 0.25, 0.25],
         "mask": {"w": W, "h": H, "bits": base64.b64encode(np.packbits(m).tobytes()).decode()}}
    (t,) = parse_objects([o])
    assert t.concept == "hand" and (t.mask == m).all()
