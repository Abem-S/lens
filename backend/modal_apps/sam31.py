"""SAM 3.1 live tracking on Modal: a stream of small frames in, tracked hands and pens out.

Deploy:  cd backend && uv run --extra gpu modal deploy modal_apps/sam31.py
Client:  lens.tracking (the app relays the browser's frames over one WebSocket per session).

One WebSocket = one tracking session, so its memory stays on one GPU container.
Meta's `sam3` package (SAM 3.1, Object Multiplex) is built for whole videos; this
file feeds it a live stream instead: frames are appended as they arrive and each
one runs the detector + multiplex tracker once (`LiveConcept.step`). One session
tracks each concept ("hand", "pen") on the same frames.

Protocol (all JSON text, except frames):
  -> binary: 4-byte big-endian frame number + JPEG (~640 px long side)
  -> {"type": "follow", "key": "finding", "box": [x, y, w, h]}   track a region (normalised)
  -> {"type": "unfollow", "key": "finding"}
  <- {"type": "ready", "load_s": ..}
  <- {"type": "tracks", "seq": n, "gpu_ms": .., "size": [w, h], "objects": [...], "follow": {...}}
Only the newest frame is processed when frames arrive faster than the GPU (no queue).
"""

import os
import time

import modal

APP_NAME = "muna-lens-sam31"
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # facebookresearch/sam3, Oct 2026
CONCEPTS = ("hand", "pen")
CAPACITY = 1200  # frames per session before it restarts (~4 min at 5 fps); bounds memory
MASK_W = 160  # masks go back at this width (packed bits), enough for fingertips
GRAY_HISTORY = 240  # frames kept for outline templates (~1 min at 4 fps, ~50 MB)
IDLE_CLOSE_S = 30  # close a tracking socket that has sent no frame for this long


def _track_token() -> str:
    """Shared secret between the app and this endpoint (derived, so no new key to manage)."""
    import hashlib

    explicit = os.environ.get("LENS_TRACK_TOKEN")
    if explicit:
        return explicit
    secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    return hashlib.sha256(f"muna-lens-track:{secret}".encode()).hexdigest()[:32] if secret else ""


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        f"git+https://github.com/facebookresearch/sam3@{SAM3_COMMIT}",
        # The package imports a few training-side modules even for inference.
        "einops", "psutil", "scipy", "opencv-python-headless", "setuptools<81", "pillow", "pycocotools",
        "huggingface_hub[hf_xet]", "fastapi[standard]",
    )
    # Fail the build (not every container start) if a module is missing; CUDA errors are expected here.
    .run_commands("""python -c "exec('try: import sam3.model_builder\\nexcept ModuleNotFoundError: raise\\n"""
                  """except Exception: pass')" """)
    .env({"HF_HOME": "/cache/hf"})
)
cache = modal.Volume.from_name("muna-lens-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)

with image.imports():
    import numpy as np
    import torch


class LiveFrames:
    """The frame list the model indexes into, filled as frames arrive.

    Only the last few frames are kept: the tracker keeps its own memory features,
    so older images are never read again.
    """

    def __init__(self, capacity: int, keep: int = 4):
        self.capacity = capacity
        self.keep = keep
        self.frames: dict = {}

    def put(self, idx: int, tensor) -> None:
        self.frames[idx] = tensor
        for old in [k for k in self.frames if k < idx - self.keep]:
            del self.frames[old]

    def __getitem__(self, idx):
        if isinstance(idx, slice) or not isinstance(idx, int):
            raise TypeError("live frames are read one at a time")
        if idx in self.frames:
            return self.frames[idx]
        return self.frames[max(self.frames)]  # never expected; keeps a stray read harmless

    def __len__(self) -> int:
        return self.capacity


class LiveConcept:
    """One concept ("hand") tracked over a live stream with SAM 3.1's detector + multiplex tracker."""

    def __init__(self, model, prompt: str, first, size: tuple[int, int], capacity: int = CAPACITY):
        self.model, self.prompt, self.capacity = model, prompt, capacity
        self.frames = LiveFrames(capacity)
        self.frames.put(0, first)
        w, h = size
        state: dict = {
            "image_size": model.image_size, "num_frames": capacity, "device": torch.device("cuda"),
            "orig_height": h, "orig_width": w, "constants": {},
        }
        model._construct_initial_input_batch(state, self.frames)
        state.update({"sam2_inference_states": [], "tracker_metadata": {}, "feature_cache": {},
                      "cached_frame_outputs": {}, "is_image_only": False, "action_history": []})
        self.state = state
        self.t = 0
        _, out = model.add_prompt(state, frame_idx=0, text_str=prompt)
        # The detector prefetches the next frame's detections; live, that frame does not
        # exist yet (it would be computed on this one and arrive one frame late).
        state["feature_cache"].get("multigpu_buffer", {}).pop(1, None)
        self.last = out

    @property
    def full(self) -> bool:
        return self.t >= self.capacity - 1

    def step(self, frame) -> dict:
        self.t += 1
        t = self.t
        self.frames.put(t, frame)
        # Tracking window ends at this frame: no look-ahead into frames that have not arrived.
        self.state["feature_cache"]["tracking_bounds"] = {"max_frame_num_to_track": 1,
                                                          "propagate_in_video_start_frame_idx": t}
        out = self.model._run_single_frame_inference(self.state, t, reverse=False)
        post = self.model._postprocess_output(
            self.state, out, removed_obj_ids=out.get("removed_obj_ids"),
            suppressed_obj_ids=out.get("suppressed_obj_ids"), unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"),
        )
        self.state["cached_frame_outputs"].pop(t - 2, None)
        self.last = post
        return post


def _pack(mask) -> dict:
    """A bool mask, shrunk to MASK_W wide and packed to bits (a few hundred bytes)."""
    import base64

    import cv2

    h, w = mask.shape
    mw = min(MASK_W, w)
    mh = max(1, round(h * mw / w))
    small = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_AREA) > 0
    return {"w": mw, "h": mh, "bits": base64.b64encode(np.packbits(small).tobytes()).decode()}


def _objects(concept: str, post: dict, min_score: float = 0.3) -> list[dict]:
    objs = []
    for i, oid in enumerate(post["out_obj_ids"].tolist()):
        score = float(post["out_probs"][i])
        if score < min_score:
            continue
        box = [round(float(v), 4) for v in post["out_boxes_xywh"][i]]
        objs.append({"concept": concept, "id": int(oid), "score": round(score, 3), "box": box,
                     "mask": _pack(post["out_binary_masks"][i])})
    return objs


class Follow:
    """Keep a region (the line with the mistake) located as the page or camera moves.

    The region's appearance in the frame the check was read from is the template; it is
    first searched for over the whole frame (the page may have moved while the check ran),
    then each frame near its last place, over a few scales (normalised cross-correlation).
    """

    def __init__(self, gray, box):
        h, w = gray.shape
        x, y, bw, bh = box
        self.size = (w, h)
        x0, y0 = max(0, int(x * w)), max(0, int(y * h))
        x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
        self.tpl = gray[y0:y1, x0:x1].copy()
        self.box = [x0, y0, x1 - x0, y1 - y0]
        self.score = 1.0
        self.lost = 0
        self.wide = True  # first search: the whole frame

    def update(self, gray, hidden=None) -> dict | None:
        import cv2

        if self.tpl.size == 0 or min(self.tpl.shape) < 4:
            return None
        h, w = gray.shape
        x, y, bw, bh = self.box
        best = None
        for s in (0.9, 1.0, 1.1):
            tw, th = max(4, int(self.tpl.shape[1] * s)), max(4, int(self.tpl.shape[0] * s))
            if tw >= w or th >= h:
                continue
            tpl = cv2.resize(self.tpl, (tw, th))
            reach = 1.0 if self.wide else 0.25  # search window around the last place
            mx, my = max(tw, int(reach * w)), max(th, int(reach * h))
            sx0, sy0 = max(0, x + bw // 2 - tw // 2 - mx), max(0, y + bh // 2 - th // 2 - my)
            sx1, sy1 = min(w, x + bw // 2 + tw // 2 + mx), min(h, y + bh // 2 + th // 2 + my)
            win = gray[sy0:sy1, sx0:sx1]
            if win.shape[0] < th or win.shape[1] < tw:
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, [sx0 + loc[0], sy0 + loc[1], tw, th])
        if best is None:
            return None
        score, box = best
        self.score = float(score)
        if score >= 0.45:
            self.box, self.lost, self.wide = box, 0, False
        else:
            self.lost += 1  # covered by the hand, or off the page: keep the last place
        bx, by, bw2, bh2 = self.box
        return {"box": [round(bx / w, 4), round(by / h, 4), round(bw2 / w, 4), round(bh2 / h, 4)],
                "score": round(self.score, 3), "lost": self.lost}


@app.cls(
    image=image,
    # Measured per frame (hand + pen): L40S 150-210 ms, A10G ~380 ms (too slow for ~5 fps).
    gpu=["L40S", "A100"],
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", ""), "LENS_TRACK_TOKEN": _track_token()})],
    scaledown_window=60,  # GPU only while someone is tracking; idle containers go quickly
    timeout=3600,
    max_containers=2,
)
@modal.concurrent(max_inputs=4)
class Tracker:
    @modal.enter()
    def load(self) -> None:
        import threading

        from sam3.model_builder import build_sam3_predictor

        t = time.monotonic()
        predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False,
                                         async_loading_frames=False)
        cache.commit()
        model = predictor.model
        # Live use: no look-ahead. Outputs come out on the frame that produced them.
        model.hotstart_delay = 0
        model.use_batched_grounding = False
        model.postprocess_batch_size = 1
        model.masklet_confirmation_enable = False
        self.predictor, self.model = predictor, model
        self.gpu = threading.Lock()  # one frame on the GPU at a time across sessions
        # Every concept sees the same frame, so the image encoder (most of the cost) runs once
        # per frame and its features are shared; only the text-conditioned heads run per concept.
        self.share_backbone = os.environ.get("SAM31_SHARE_BACKBONE", "1") == "1"
        self._frame_key = 0
        self._bb_cache: dict = {}
        backbone = model.detector.backbone
        encode = backbone.forward_image

        def forward_image_once(samples, **kw):
            if not self.share_backbone:
                return encode(samples, **kw)
            key = (self._frame_key, tuple(sorted(kw.items())), tuple(samples.shape))
            if key not in self._bb_cache:
                self._bb_cache = {key: encode(samples, **kw)}
            return self._bb_cache[key]

        backbone.forward_image = forward_image_once
        self.load_seconds = time.monotonic() - t
        self.device_name = torch.cuda.get_device_name(0)

    def _tensor(self, rgb):
        """uint8 HxWx3 -> the model's normalised 3x1008x1008 half tensor on the GPU."""
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float().div_(255.0)
        size = self.model.image_size
        x = torch.nn.functional.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        return ((x - 0.5) / 0.5).half()

    def _decode(self, jpeg: bytes):
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        return np.array(img)

    def new_session(self) -> dict:
        import collections

        # Recent grey frames by number, so an outline can start from the frame a check read.
        return {"concepts": {}, "follow": {}, "frames": 0, "grays": collections.OrderedDict()}

    def process(self, sess: dict, jpeg: bytes, commands: list[dict], seq: int | None = None) -> dict:
        """One frame through every concept session (+ followed regions). Thread-safe."""
        import cv2

        rgb = self._decode(jpeg)
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t0 = time.monotonic()
        objects: list[dict] = []
        with self.gpu, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._frame_key += 1
            frame = self._tensor(rgb)
            for concept in CONCEPTS:
                live = sess["concepts"].get(concept)
                if live is None or live.full or live.state["orig_width"] != w or live.state["orig_height"] != h:
                    live = sess["concepts"][concept] = LiveConcept(self.model, concept, frame, (w, h))
                    post = live.last
                else:
                    post = live.step(frame)
                objects += _objects(concept, post)
            torch.cuda.synchronize()
            self._bb_cache = {}
        gpu_ms = (time.monotonic() - t0) * 1000
        grays = sess["grays"]
        if seq is not None:
            grays[seq] = gray
            while len(grays) > GRAY_HISTORY:
                grays.popitem(last=False)
        for cmd in commands:
            if cmd.get("type") == "follow" and cmd.get("box"):
                src = grays.get(cmd.get("seq"), gray)
                sess["follow"][cmd.get("key", "finding")] = Follow(src if src.shape == gray.shape else gray, cmd["box"])
            elif cmd.get("type") == "unfollow":
                sess["follow"].pop(cmd.get("key", "finding"), None)
        follow = {k: f.update(gray) for k, f in sess["follow"].items()}
        sess["frames"] += 1
        return {"type": "tracks", "gpu_ms": round(gpu_ms, 1), "size": [w, h], "objects": objects,
                "follow": follow, "device": self.device_name}

    @modal.method()
    def replay(self, frames: list[bytes], follow_at: int | None = None, follow_box=None,
               share_backbone: bool = True) -> list[dict]:
        """Offline check: run a clip through exactly the live path; returns one result per frame."""
        self.share_backbone = share_backbone
        sess = self.new_session()
        out = []
        for i, f in enumerate(frames):
            cmds = [{"type": "follow", "key": "finding", "box": follow_box}] if i == follow_at and follow_box else []
            t = time.monotonic()
            r = self.process(sess, f, cmds, seq=i)
            r["wall_ms"] = round((time.monotonic() - t) * 1000, 1)
            r["seq"] = i
            out.append(r)
        return out

    @modal.method()
    def ping(self) -> dict:
        return {"ok": True, "load_seconds": self.load_seconds, "device": self.device_name}

    @modal.asgi_app()
    def web(self):
        import asyncio
        import json

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from starlette.concurrency import run_in_threadpool

        api = FastAPI()
        token = os.environ.get("LENS_TRACK_TOKEN", "")

        @api.get("/health")
        def health():
            return {"ok": True, "device": self.device_name, "load_s": round(self.load_seconds, 1)}

        @api.websocket("/track")
        async def track(ws: WebSocket):
            await ws.accept()
            if not token or ws.query_params.get("token") != token:
                await ws.close(code=4403, reason="bad token")
                return
            await ws.send_text(json.dumps({"type": "ready", "load_s": round(self.load_seconds, 1),
                                           "device": self.device_name}))
            sess = self.new_session()
            latest: dict = {"frame": None}
            commands: list[dict] = []
            arrived = asyncio.Event()
            closed = False

            async def receive():
                nonlocal closed
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if msg.get("bytes"):
                            b = msg["bytes"]
                            latest["frame"] = (int.from_bytes(b[:4], "big"), b[4:])  # newest frame wins
                            arrived.set()
                        elif msg.get("text"):
                            commands.append(json.loads(msg["text"]))
                except WebSocketDisconnect:
                    pass
                finally:
                    closed = True
                    arrived.set()

            receiver = asyncio.create_task(receive())
            try:
                while True:
                    try:  # a session that stops sending frames gives the GPU back
                        await asyncio.wait_for(arrived.wait(), timeout=IDLE_CLOSE_S)
                    except TimeoutError:
                        await ws.close(code=4408, reason="no frames")
                        break
                    arrived.clear()
                    if closed:
                        break
                    item, latest["frame"] = latest["frame"], None
                    if item is None:
                        continue
                    seq, jpeg = item
                    cmds, commands[:] = list(commands), []
                    t = time.monotonic()
                    try:
                        result = await run_in_threadpool(self.process, sess, jpeg, cmds, seq)
                    except Exception as e:  # a bad frame must not end the session
                        await ws.send_text(json.dumps({"type": "error", "seq": seq, "message": str(e)[:200]}))
                        continue
                    result["seq"] = seq
                    result["wall_ms"] = round((time.monotonic() - t) * 1000, 1)
                    await ws.send_text(json.dumps(result))
                    if latest["frame"] is not None:
                        arrived.set()
            except WebSocketDisconnect:
                pass
            finally:
                receiver.cancel()
                sess["concepts"].clear()
                torch.cuda.empty_cache()

        return api
