"""A2 offline: model replies are parsed into a valid Page. The live done criterion
(10 eval pages read correctly) runs through the A5 harness with a key."""

import pytest

from lens.reader import ReadError, parse_page, read_page
from lens.spark import CallLog, Reply


def test_parses_plain_json():
    page = parse_page('{"problem": "Solve 2x+5=11", "steps": ['
                      '{"latex": "2x+5=11", "bbox": [0.1, 0.2, 0.5, 0.05]},'
                      '{"latex": "2x=6", "bbox": [0.1, 0.3, 0.3, 0.05]}]}')
    assert page.problem == "Solve 2x+5=11"
    assert [s.id for s in page.steps] == ["s1", "s2"]
    assert page.steps[1].bbox == (0.1, 0.3, 0.3, 0.05)
    assert all(s.status == "unchecked" for s in page.steps)


def test_parses_fenced_json_with_chatter_and_rescales_bad_boxes():
    page = parse_page('Here you go:\n```json\n{"problem": null, "steps": ['
                      '{"latex": "x=3", "bbox": [100, 200, 300, 50]},'
                      '{"latex": "", "bbox": [0,0,0,0]},'
                      '{"latex": "= 4", "bbox": "oops"}]}\n```')
    assert page.problem is None
    assert [s.latex for s in page.steps] == ["x=3", "= 4"]
    assert page.steps[0].bbox == pytest.approx((0.1, 0.2, 0.3, 0.05))
    assert page.steps[1].bbox == (0.0, 0.0, 0.0, 0.0)


def test_boxes_are_clamped_to_the_frame():
    page = parse_page('{"steps": [{"latex": "a", "bbox": [0.9, 0.9, 0.5, 0.5]}]}')
    x, y, w, h = page.steps[0].bbox
    assert x + w <= 1.0 and y + h <= 1.0


def test_no_json_raises():
    with pytest.raises(ReadError):
        parse_page("I cannot read this")


class FakeSpark:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def ask(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return Reply(self.replies.pop(0), CallLog(kw.get("purpose", ""), kw.get("effort")))


def test_read_page_sends_image_with_big_budget_and_retries_bad_json():
    spark = FakeSpark(["not json", '{"problem": null, "steps": [{"latex": "x=1", "bbox": [0,0,1,0.1]}]}'])
    page = read_page(spark, b"jpeg")
    assert [s.latex for s in page.steps] == ["x=1"]
    first_prompt, kw = spark.calls[0]
    assert first_prompt[1]["type"] == "image_url"
    assert kw["max_tokens"] >= 3000
    assert len(spark.calls) == 2


def test_corner_boxes_are_converted_to_xywh():
    page = parse_page('{"steps": [{"latex": "x=1", "box": [0.1, 0.30, 0.6, 0.36]}]}')
    assert page.steps[0].bbox == pytest.approx((0.1, 0.30, 0.5, 0.06))
