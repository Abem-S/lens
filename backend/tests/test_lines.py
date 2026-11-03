"""Handwritten lines from the ink, and Spark's step boxes snapped onto them.

Synthetic pages: text drawn on lined or grid paper, with a shadow across the page
and a slight tilt, like a phone photo of a notebook."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from lens.contract import Step
from lens.lines import InkLine, find_lines, snap_steps

W, H = 900, 1200
PITCH = 44  # ruled line spacing, px


def notebook(texts, *, rot=0.0, grid=False, shadow=True, every=2, start=1, x0=120, extra=()):
    """Lines of text on ruled paper; returns (jpeg, truth boxes normalised)."""
    img = Image.new("RGB", (W, H), (250, 248, 240))
    d = ImageDraw.Draw(img)
    for y in range(80, H, PITCH):
        d.line([(0, y), (W, y)], fill=(170, 200, 230), width=2)
    if grid:
        for x in range(0, W, PITCH):
            d.line([(x, 0), (x, H)], fill=(170, 200, 230), width=2)
    d.line([(90, 0), (90, H)], fill=(230, 150, 150), width=2)
    font = ImageFont.load_default(size=30)
    truth = []
    for i, t in enumerate(texts):
        y = 80 + PITCH * (every * i + start) - 34
        x = x0 + 15 * (i % 3)
        d.text((x, y), t, fill=(30, 30, 90), font=font)
        l, top, r, b = d.textbbox((x, y), t, font=font)
        truth.append((l / W, top / H, (r - l) / W, (b - top) / H))
    for x, y, t in extra:
        d.text((x, y), t, fill=(30, 30, 90), font=font)
    if rot:
        img = img.rotate(rot, fillcolor=(250, 248, 240))
        truth = [_rotated(b, rot) for b in truth]
    if shadow:  # light falling off across the page
        a = np.asarray(img).astype(float)
        img = Image.fromarray((a * np.linspace(1.0, 0.55, W)[None, :, None]).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), truth


def _rotated(b, angle):
    import math

    a = -math.radians(angle)
    cx, cy = W / 2, H / 2
    pts = [(b[0] * W, b[1] * H), ((b[0] + b[2]) * W, b[1] * H), (b[0] * W, (b[1] + b[3]) * H),
           ((b[0] + b[2]) * W, (b[1] + b[3]) * H)]
    xs = [cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a) for x, y in pts]
    ys = [cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a) for x, y in pts]
    return (min(xs) / W, min(ys) / H, (max(xs) - min(xs)) / W, (max(ys) - min(ys)) / H)


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def matches(found, truth, min_iou=0.4):
    """Each true line overlapped by exactly one found line, in the same order."""
    picked = [max(range(len(found)), key=lambda j: iou(found[j], t)) for t in truth]
    return all(iou(found[j], t) >= min_iou for j, t in zip(picked, truth)) and picked == sorted(set(picked))


TEXTS = ["2x + 5 = 11", "2x = 11 - 5", "2x = 6", "x = 3"]


@pytest.mark.parametrize("rot,grid", [(0, False), (2.5, False), (-3, False), (0, True), (1.5, True)])
def test_lines_found_on_ruled_and_grid_paper(rot, grid):
    jpeg, truth = notebook(TEXTS, rot=rot, grid=grid)
    lines = find_lines(jpeg)
    assert len(lines) == len(TEXTS), [ln.box for ln in lines]
    assert matches([ln.box for ln in lines], truth)


def test_blank_page_has_no_lines():
    jpeg, _ = notebook([])
    assert find_lines(jpeg) == []


def test_lines_on_neighbouring_rules_stay_apart():
    jpeg, truth = notebook(TEXTS * 2, every=1)  # one line per rule, no blank rule between
    lines = find_lines(jpeg)
    assert len(lines) == 8 and matches([ln.box for ln in lines], truth)


def test_a_distant_note_beside_a_line_is_its_own_line():
    jpeg, truth = notebook(TEXTS, extra=[(680, 80 + PITCH * 3 - 34, "(div 2)")])
    lines = find_lines(jpeg)
    side = [ln for ln in lines if ln.box[0] > 0.7]
    assert len(side) == 1 and abs(side[0].cy - (truth[1][1] + truth[1][3] / 2)) < 0.02


def _steps(boxes, texts=None):
    texts = texts or [f"x = {i}" for i in range(len(boxes))]
    return [Step(id=f"s{i + 1}", latex=t, bbox=b) for i, (t, b) in enumerate(zip(texts, boxes))]


def _line(y, x=0.1, w=0.5, h=0.03, segments=None):
    return InkLine((x, y, w, h), 0.01, segments or [(x, y, w, h)])


def test_snap_fixes_drifting_spark_boxes():
    lines = [_line(0.1 + 0.06 * i) for i in range(8)]
    # Spark's boxes drift down and squeeze: off by up to two lines by the end.
    spark = [(0.12, 0.13 + 0.045 * i + 0.012 * i, 0.4, 0.05) for i in range(8)]
    out = snap_steps(_steps(spark), lines)
    assert [round(s.bbox[1] + s.bbox[3] / 2, 3) for s in out] == [round(ln.cy, 3) for ln in lines]


def test_snap_skips_the_question_and_crossed_out_lines():
    steps = [0.25, 0.31, 0.43, 0.49]
    lines = [_line(0.05, w=0.8), _line(0.12, w=0.8)] + [_line(y) for y in steps]  # the question, then the work
    lines.insert(4, _line(0.37, x=0.6, w=0.2))  # a crossed-out scribble between two steps
    out = snap_steps(_steps([(0.1, y - 0.01, 0.5, 0.05) for y in steps]), lines)
    assert [round(s.bbox[1], 2) for s in out] == [0.25, 0.31, 0.43, 0.49]


def test_a_step_written_over_two_lines_takes_both():
    lines = [_line(0.10), _line(0.16), _line(0.205, x=0.2, w=0.2), _line(0.25)]  # line 2-3: a tall fraction
    spark = [(0.1, 0.1, 0.5, 0.03), (0.1, 0.155, 0.5, 0.08), (0.1, 0.25, 0.5, 0.03)]
    out = snap_steps(_steps(spark), lines)
    top, bottom = out[1].bbox[1], out[1].bbox[1] + out[1].bbox[3]
    assert top < 0.165 and bottom > 0.23


def test_two_steps_side_by_side_share_one_line():
    segs = [(0.1, 0.2, 0.25, 0.03), (0.5, 0.2, 0.2, 0.03)]
    lines = [_line(0.1), InkLine((0.1, 0.2, 0.6, 0.03), 0.01, segs), _line(0.3)]
    spark = [(0.1, 0.1, 0.5, 0.03), (0.1, 0.21, 0.25, 0.03), (0.5, 0.21, 0.2, 0.03), (0.1, 0.3, 0.5, 0.03)]
    out = snap_steps(_steps(spark), lines)
    assert out[1].bbox[0] < 0.11 and out[1].bbox[0] + out[1].bbox[2] < 0.4
    assert out[2].bbox[0] > 0.45
    assert abs(out[3].bbox[1] - 0.3) < 0.01


def test_no_lines_or_no_steps_change_nothing():
    steps = _steps([(0.1, 0.1, 0.5, 0.05)])
    assert snap_steps(steps, []) == steps
    assert snap_steps([], [_line(0.1)]) == []


def test_real_render_end_to_end():
    jpeg, truth = notebook(TEXTS, rot=2)
    rough = [(t[0] + 0.05, t[1] + 0.07, t[2], t[3] * 2) for t in truth]  # off by more than a line
    out = snap_steps(_steps(rough, TEXTS), find_lines(jpeg))
    assert all(iou(s.bbox, t) > 0.4 for s, t in zip(out, truth))
