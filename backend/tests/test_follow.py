"""The outline follows the mistaken step as the page moves (lens.follow; pure numpy, no GPU).

Frames are a synthetic ruled notebook page (tests/test_lines.py) filmed by a camera that
shifts, turns and zooms; truth is the step's box carried by the same motion."""

import io
import math

import numpy as np
from PIL import Image, ImageDraw
from test_lines import notebook

from lens.follow import Follow, FollowConfig, gray_of, ink, ncc
from lens.tracking import OutlineFollower, TrackHub

TEXTS = ["2x + 5 = 11", "2x = 11 + 5", "2x = 16", "x = 8", "so x is eight"]
FW, FH = 512, 683  # a tracking frame: phone held over the notebook, portrait


def camera(page: Image.Image, dx=0.0, dy=0.0, angle=0.0, zoom=1.0, size=(FW, FH), cover=None) -> Image.Image:
    """The page as the camera sees it: fit to the frame, then shifted (fractions of the frame),
    turned (degrees, about the centre) and zoomed. `cover`: a hand-coloured box (normalised)."""
    w, h = size
    s = min(w / page.width, h / page.height) * zoom
    a = math.radians(angle)
    # frame point -> page point: undo the shift, the turn about the centre, then the fit.
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = w / 2 + dx * w, h / 2 + dy * h
    coeffs = (ca / s, sa / s, page.width / 2 - (ca * cx + sa * cy) / s,
              -sa / s, ca / s, page.height / 2 - (-sa * cx + ca * cy) / s)
    img = page.transform(size, Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(70, 62, 58))
    if cover:
        x, y, bw, bh = cover
        ImageDraw.Draw(img).rectangle([x * w, y * h, (x + bw) * w, (y + bh) * h], fill=(196, 150, 120))
    return img


def where(page: Image.Image, box, dx=0.0, dy=0.0, angle=0.0, zoom=1.0, size=(FW, FH)):
    """Centre (normalised) of a page box (normalised to the page) on the camera frame."""
    w, h = size
    s = min(w / page.width, h / page.height) * zoom
    px = (box[0] + box[2] / 2 - 0.5) * page.width * s
    py = (box[1] + box[3] / 2 - 0.5) * page.height * s
    a = math.radians(angle)
    return ((w / 2 + dx * w + math.cos(a) * px - math.sin(a) * py) / w,
            (h / 2 + dy * h + math.sin(a) * px + math.cos(a) * py) / h)


def jpeg(img: Image.Image, q=70) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return buf.getvalue()


def setup():
    data, truth = notebook(TEXTS, shadow=False)
    page = Image.open(io.BytesIO(data)).convert("RGB")
    return page, truth


def frame_box(page, box, **cam):
    """The page box on a camera frame, axis-aligned (how the reader would box it there)."""
    c = where(page, box, **cam)
    s = min(FW / page.width, FH / page.height) * cam.get("zoom", 1.0)
    w, h = box[2] * page.width * s / FW, box[3] * page.height * s / FH
    return (c[0] - w / 2, c[1] - h / 2, w, h)


def centre(quad):
    return sum(p[0] for p in quad) / 4, sum(p[1] for p in quad) / 4


def test_ncc_matches_the_textbook_formula():
    rng = np.random.default_rng(1)
    img = rng.random((30, 40)).astype(np.float32)
    tpl = img[7:15, 11:23].copy()
    res = ncc(img, tpl)
    assert res.shape == (23, 29)
    assert np.unravel_index(int(np.argmax(res)), res.shape) == (7, 11) and abs(res[7, 11] - 1) < 1e-4
    y, x = 3, 5
    win = img[y:y + 8, x:x + 12].astype(np.float64)
    t = tpl - tpl.mean()
    want = ((win - win.mean()) * t).sum() / math.sqrt(((win - win.mean()) ** 2).sum() * (t ** 2).sum())
    assert abs(res[y, x] - want) < 1e-4
    assert not ncc(np.full((20, 20), 5.0), tpl[:5, :5]).any()  # blank paper scores 0


def test_ink_keeps_strokes_and_drops_light_and_flat_colour():
    img = np.full((40, 60), 200.0, dtype=np.float32)
    img[:, 30:] = 120.0  # a shadow over half the page (or a sleeve)
    img[20, 5:25] = 40.0  # a pen stroke
    out = ink(img, 3, soften=0)
    assert out[20, 15] > 100 and out[5, 10] == 0 and out[5, 50] == 0  # flat paper or flat shadow: nothing
    assert (out >= 0).all()


def test_follow_is_quick_enough_for_the_app():
    """The app runs it on a CPU for each tracked frame (~3 a second) while a mistake is outlined."""
    import time

    page, truth = setup()
    f = Follow(gray_of(jpeg(camera(page), 85)), frame_box(page, truth[2]))
    frame = gray_of(jpeg(camera(page, dx=0.03, angle=2.0)))
    t = time.perf_counter()
    f.update(frame)  # the first look searches the whole frame
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(5):
        f.update(frame)
    assert first < 2.0 and (time.perf_counter() - t) / 5 < 0.2


def test_outline_follows_a_shift_a_turn_and_a_zoom():
    page, truth = setup()
    step = truth[2]
    check = camera(page)
    f = Follow(gray_of(jpeg(check, 85)), frame_box(page, step))
    # The page drifts, then is nudged ~3 cm and turned 4 degrees (the check took a while).
    for k, cam in enumerate([{"dx": 0.01}, {"dx": 0.03, "dy": -0.02, "angle": 1.0}, {"dx": 0.1, "dy": 0.03, "angle": 4.0},
                             {"dx": 0.11, "dy": 0.035, "angle": 4.5, "zoom": 1.06}]):
        r = f.update(gray_of(jpeg(camera(page, **cam))))
        cx, cy = centre(r["quad"])
        tx, ty = where(page, step, **cam)
        assert r["state"] == "ok", (k, r)
        assert math.hypot(cx - tx, cy - ty) < 0.008, (k, cx, cy, tx, ty)
        assert abs(r["angle"] - cam.get("angle", 0.0)) <= 1.0
    assert r["anchor"] == [round(v, 4) for v in frame_box(page, step)]


def test_found_again_after_a_long_check_moved_the_page():
    """The first frame after the result may show the page far from where the check read it."""
    page, truth = setup()
    step = truth[3]
    f = Follow(gray_of(jpeg(camera(page), 85)), frame_box(page, step))
    cam = {"dx": -0.12, "dy": 0.08, "angle": -6.0, "zoom": 0.92}
    r = f.update(gray_of(jpeg(camera(page, **cam))))
    assert r["state"] == "ok"
    cx, cy = centre(r["quad"])
    tx, ty = where(page, step, **cam)
    assert math.hypot(cx - tx, cy - ty) < 0.01


def test_a_covered_step_is_held_a_moment_then_hidden_then_found_again():
    page, truth = setup()
    step = truth[2]
    box = frame_box(page, step)
    f = Follow(gray_of(jpeg(camera(page), 85)), box, FollowConfig(hold=3))
    assert f.update(gray_of(jpeg(camera(page))))["state"] == "ok"
    hand = (box[0] - 0.05, box[1] - 0.06, box[2] + 0.1, box[3] + 0.12)
    states = [f.update(gray_of(jpeg(camera(page, cover=hand))))["state"] for _ in range(5)]
    assert states == ["held", "held", "held", "lost", "lost"]
    cam = {"dx": 0.04, "angle": 2.0}  # the hand leaves; the page moved meanwhile
    states = [f.update(gray_of(jpeg(camera(page, **cam))))["state"] for _ in range(4)]
    assert "ok" in states
    cx, cy = centre(f.report()["quad"])
    tx, ty = where(page, step, **cam)
    assert math.hypot(cx - tx, cy - ty) < 0.01


def test_never_sure_between_two_places_that_look_the_same():
    """Two identical blocks of writing: the outline is hidden rather than put on either."""
    data, truth = notebook(["2x = 16", "x = 8", "2x = 16"], shadow=False)
    page = Image.open(io.BytesIO(data)).convert("RGB").resize((FW, FH), Image.LANCZOS)  # frame-sized: whole pixels
    page.paste(page.crop((0, 0, FW, FH // 2)), (0, FH // 2))  # the same writing twice
    f = Follow(gray_of(jpeg(camera(page), 85)), frame_box(page, truth[1]))
    r = f.update(gray_of(jpeg(camera(page, dx=0.05))))
    assert r["state"] == "lost"


def test_another_aspect_is_not_compared():
    page, truth = setup()
    f = Follow(gray_of(jpeg(camera(page), 85)), frame_box(page, truth[2]))
    assert f.update(gray_of(jpeg(camera(page, size=(512, 288))))) is None


def test_relay_follower_starts_from_the_check_frame_the_tutor_handed_over():
    page, truth = setup()
    hub = TrackHub()
    follower = OutlineFollower(hub, "t")
    frame = jpeg(camera(page, dx=0.05, angle=2.0))
    assert follower.update(frame) == {}  # nothing outlined
    box = frame_box(page, truth[2])
    big = camera(page, size=(1024, 1366))  # the check frame, grabbed at full size
    hub.follow("t", jpeg(big, 85), box)
    out = follower.update(frame)["finding"]
    assert out["state"] == "ok" and out["anchor"] == [round(v, 4) for v in box]
    tx, ty = where(page, truth[2], dx=0.05, angle=2.0)
    cx, cy = centre(out["quad"])
    assert math.hypot(cx - tx, cy - ty) < 0.01
    hub.follow("t", None)  # the mistake is fixed
    assert follower.update(frame) == {}
    hub.follow("t", jpeg(big, 85), frame_box(page, truth[3]))  # a new one: starts afresh
    assert follower.update(frame)["finding"]["anchor"] == [round(v, 4) for v in frame_box(page, truth[3])]
    hub.drop("t")  # tracking reconnecting keeps the outline: it belongs to the tutor
    assert hub.followed("t") is not None
