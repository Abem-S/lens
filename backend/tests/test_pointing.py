"""C2 geometry on synthetic masks. The live done criterion (10 real photos, right
step >= 9 times) needs SAM on Modal (C1)."""

import numpy as np
import pytest

from lens.contract import Step
from lens.pointing import bbox_outline, fingertip_from_mask, mask_outline, nearest_step, outline_for_step

LINES = [Step(id=f"s{i + 1}", latex="x", bbox=(0.1, 0.1 + 0.12 * i, 0.6, 0.06)) for i in range(6)]


def finger(h=400, w=600, tip=(300, 150), from_edge="bottom", width=24):
    """A finger-shaped mask from an image edge to `tip` (x, y in pixels)."""
    m = np.zeros((h, w), bool)
    tx, ty = tip
    if from_edge == "bottom":
        m[ty:, tx - width // 2 : tx + width // 2] = True
    elif from_edge == "right":
        m[ty - width // 2 : ty + width // 2, tx:] = True
    return m


@pytest.mark.parametrize("edge,tip", [("bottom", (300, 150)), ("right", (200, 220)), ("bottom", (90, 60))])
def test_fingertip_is_far_end_from_edge(edge, tip):
    x, y = fingertip_from_mask(finger(tip=tip, from_edge=edge))
    assert abs(x * 600 - tip[0]) <= 13 and abs(y * 400 - tip[1]) <= 13


def test_fingertip_of_floating_mask_is_topmost_point():
    m = np.zeros((100, 100), bool)
    m[40:60, 45:55] = True
    assert fingertip_from_mask(m)[1] == pytest.approx(0.405)


def test_empty_mask_has_no_tip():
    assert fingertip_from_mask(np.zeros((10, 10), bool)) is None


@pytest.mark.parametrize("i", range(6))
def test_pointing_on_or_just_below_a_line_picks_it(i):
    x, y, w, h = LINES[i].bbox
    for px, py in [(x + w / 2, y + h / 2), (x + 0.02, y + h + 0.02), (x - 0.03, y + h / 2), (x + w + 0.05, y + h)]:
        assert nearest_step((px, py), LINES).id == LINES[i].id, (px, py)


def test_pointing_far_away_picks_nothing():
    assert nearest_step((0.95, 0.95), LINES) is None


def test_ten_simulated_pointing_photos():
    """Mirror of the C2 done criterion: 10 fingers on lines, right step >= 9 times."""
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(10):
        i = int(rng.integers(0, 6))
        x, y, w, h = LINES[i].bbox
        tip = (int((x + rng.uniform(0, w)) * 600), int((y + h * rng.uniform(0.3, 1.3)) * 400))
        point = fingertip_from_mask(finger(tip=tip, from_edge=rng.choice(["bottom", "right"])))
        hits += nearest_step(point, LINES).id == LINES[i].id
    assert hits >= 9


def test_bbox_outline_surrounds_box_and_stays_in_frame():
    poly = bbox_outline((0.0, 0.5, 0.4, 0.05))
    xs, ys = zip(*poly)
    assert min(xs) >= 0.0 and max(xs) <= 0.42 and min(ys) <= 0.5 and max(ys) >= 0.55
    assert len(poly) >= 8


def test_mask_outline_and_fallback():
    m = np.zeros((100, 200), bool)
    m[20:30, 40:160] = True
    poly = mask_outline(m)
    xs, ys = zip(*poly)
    assert min(xs) == pytest.approx(0.2) and max(xs) == pytest.approx(0.8)
    assert min(ys) == pytest.approx(0.2) and max(ys) == pytest.approx(0.3)
    assert outline_for_step(LINES[0], np.zeros((5, 5), bool)) == bbox_outline(LINES[0].bbox)


def test_pointer_tip_uses_the_hand_and_extends_along_a_held_pen():
    from lens.pointing import pointer_tip

    hand = finger(tip=(300, 250), from_edge="bottom", width=60)
    pen = np.zeros_like(hand)
    pen[180:252, 296:304] = True  # pen sticking up out of the hand
    stray_pen = np.zeros_like(hand)
    stray_pen[10:40, 500:510] = True  # a pen lying on the desk elsewhere

    def seg(masks):
        return lambda image, text: masks.get(text, [])

    _, y = pointer_tip(seg({"hand": [hand], "pen": [pen]}), b"img")
    assert abs(y * 400 - 180) <= 3  # the pen tip, not the knuckles
    _, y = pointer_tip(seg({"hand": [hand], "pen": [stray_pen]}), b"img")
    assert abs(y * 400 - 250) <= 3  # a pen not in the hand is ignored
    assert pointer_tip(seg({}), b"img") is None
