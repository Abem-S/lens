"""E0 loaders and A5 scorecard, on local fixtures shaped like the real datasets."""

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from lens.contract import Finding, Page, Step
from lens.eval.harness import Score, format_scorecard, reading_similarity, run
from lens.eval.loaders import EvalItem, load_edu_circuit, load_fermat, load_own


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    return buf.getvalue()


def test_fermat_loader_reads_local_shards(tmp_path):
    shard = tmp_path / "fermat" / "data" / "train-00000-of-00010.parquet"
    shard.parent.mkdir(parents=True)
    rows = [
        {"new_custom_id": "a", "image": {"bytes": png(), "path": "a.png"}, "orig_q": "Solve 2x=4", "orig_a": "x=2",
         "pert_a": None, "pert_reasoning": None, "has_error": False, "grade": "7", "domain_code": "AL", "rotation": "0"},
        {"new_custom_id": "b", "image": {"bytes": png(), "path": "b.png"}, "orig_q": "Solve 2x=4", "orig_a": "x=2",
         "pert_a": "x=3", "pert_reasoning": "divided wrong", "has_error": True, "grade": "7", "domain_code": "AL",
         "rotation": "0"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    items = list(load_fermat(cache=tmp_path, shards=1))
    assert [(i.id, i.first_error_step) for i in items] == [("a", "none"), ("b", "unknown")]
    assert items[0].image.startswith(b"\x89PNG") and items[1].reference == "x=3"
    assert items[1].meta["pert_reasoning"] == "divided wrong"


def test_fermat_without_token_explains(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        list(load_fermat(cache=tmp_path, shards=1))


def test_edu_circuit_prefers_rectified_transcripts(tmp_path):
    root = tmp_path / "edu_circuit" / "release"

    def put(base, student, qid, text):
        d = root / base / f"Homework_collected_database_trial_Homework1_{student}" / "models" / "gemini-2.5-pro" / "Compare"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{qid}_markdown.md").write_text(text)
        (d / f"{qid}_source.png").write_bytes(png())

    put("Observationset_Final/v6_Gemini_2p5", "student_2", "1_5-2", "V = IR wrong")
    put("Observationset_Final/v6_Gemini_2p5", "student_2", "3_6-1_1", "KCL")
    put("Rectified_recognized_markdown_done_Anon/Final_4_LLM_judge", "student_2", "1_5-2", "V = IR")
    items = {i.id: i for i in load_edu_circuit(cache=tmp_path)}
    assert set(items) == {"Homework1_student_2/1_5-2", "Homework1_student_2/3_6-1_1"}
    assert items["Homework1_student_2/1_5-2"].reference == "V = IR"
    assert items["Homework1_student_2/1_5-2"].meta["rectified"] is True
    assert all(i.first_error_step is None for i in items.values())


def test_own_loader_finds_images_in_repo_or_cache(tmp_path):
    labels = tmp_path / "own" / "labels.json"
    labels.parent.mkdir()
    (tmp_path / "cache" / "own").mkdir(parents=True)
    (tmp_path / "cache" / "own" / "p01.jpg").write_bytes(b"jpg")
    labels.write_text(json.dumps([
        {"file": "p01.jpg", "problem": "d/dx x^3", "first_error_step": 2, "subject": "calculus", "camera": "phone"},
        {"file": "missing.jpg", "first_error_step": "none"},
    ]))
    items = list(load_own(labels=labels, cache=tmp_path / "cache"))
    assert len(items) == 1 and items[0].first_error_step == 2 and items[0].meta["camera"] == "phone"


def test_repo_own_set_labels_are_valid():
    from lens.eval.loaders import OWN_DIR

    entries = json.loads((OWN_DIR / "labels.json").read_text())
    for e in entries:
        assert {"file", "first_error_step", "subject", "camera"} <= set(e)
        assert e["first_error_step"] == "none" or isinstance(e["first_error_step"], int)


def item(i, label, ref=None):
    return EvalItem("own", f"p{i}", lambda: b"img", None, label, reference=ref)


def page_with_error(at):
    steps = [Step(id=f"s{k + 1}", latex=f"x={k}") for k in range(4)]
    return Page(steps=steps), (Finding(step_id=f"s{at + 1}", kind="k") if at is not None else None)


def test_scorecard_counts_every_outcome():
    plan = {  # item id -> (label, predicted error index)
        "p0": (1, 1), "p1": (1, 2), "p2": (2, None), "p3": ("none", None), "p4": ("none", 0), "p5": ("unknown", 3),
    }
    items = [item(int(k[1]), lab) for k, (lab, _) in plan.items()]
    preds = iter([page_with_error(p) for _, p in plan.values()])
    costs = iter([0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06])
    s = run(items, lambda img, prob: next(preds), cost=lambda: next(costs))["own"]
    assert (s.caught, s.wrong_line, s.missed, s.false_alarms, s.clean_ok) == (2, 1, 1, 1, 1)
    assert s.false_alarm_rate == 0.5 and s.gate_passed is False
    assert s.summary()["cost_per_page"] == pytest.approx(0.01)
    assert "own" in format_scorecard({"own": s})


def test_failures_are_counted_and_do_not_stop_the_run():
    def pipeline(img, prob):
        raise ValueError("bad page")

    s = run([item(0, "none"), item(1, "none")], pipeline)["own"]
    assert s.failures == 2 and s.pages == 0


def test_gate_passes_at_one_in_twenty():
    s = Score("x", false_alarms=1, clean_ok=19)
    assert s.gate_passed is True
    assert Score("x", false_alarms=2, clean_ok=18).gate_passed is False


def test_reading_similarity_ignores_latex_markup():
    page = Page(steps=[Step(id="s1", latex="V = I R"), Step(id="s2", latex="\\text{so } I = 2")])
    assert reading_similarity(page, "$V = IR$\n\nso $I = 2$") == pytest.approx(1.0)
