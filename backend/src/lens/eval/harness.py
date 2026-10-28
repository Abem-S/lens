"""A5 Eval harness: reader + checker over the eval sets, scored per source.

    uv run lens-eval --source own --limit 20
    uv run lens-eval --source fermat --limit 40 --shards 1

Gate (AGENTS.md): false alarms <= 1 in 20 correct pages.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

from lens.contract import Finding, Page
from lens.eval.loaders import SOURCES, EvalItem, load

# pipeline(image, problem) -> (checked page, first finding or None)
Pipeline = Callable[[bytes, str | None], tuple[Page, Finding | None]]
FALSE_ALARM_GATE = 1 / 20


@dataclass
class Score:
    source: str
    pages: int = 0
    caught: int = 0          # error page, flagged at the right line (or any line when unlabelled)
    wrong_line: int = 0      # error page, flagged at another line
    missed: int = 0          # error page, nothing flagged
    false_alarms: int = 0    # correct page, something flagged
    clean_ok: int = 0        # correct page, nothing flagged
    failures: int = 0        # pipeline raised
    reading_similarity: list[float] = field(default_factory=list)
    seconds: float = 0.0
    cost_usd: float | None = 0.0

    @property
    def error_pages(self) -> int:
        return self.caught + self.wrong_line + self.missed

    @property
    def clean_pages(self) -> int:
        return self.false_alarms + self.clean_ok

    @property
    def false_alarm_rate(self) -> float | None:
        return self.false_alarms / self.clean_pages if self.clean_pages else None

    @property
    def gate_passed(self) -> bool | None:
        r = self.false_alarm_rate
        return None if r is None else r <= FALSE_ALARM_GATE

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "reading_similarity"}
        d.update(
            error_pages=self.error_pages, clean_pages=self.clean_pages,
            catch_rate=round(self.caught / self.error_pages, 3) if self.error_pages else None,
            false_alarm_rate=None if self.false_alarm_rate is None else round(self.false_alarm_rate, 3),
            gate_passed=self.gate_passed,
            reading_similarity=round(sum(self.reading_similarity) / len(self.reading_similarity), 3)
            if self.reading_similarity else None,
            cost_per_page=None if self.cost_usd is None or not self.pages else round(self.cost_usd / self.pages, 5),
            seconds_per_page=round(self.seconds / self.pages, 2) if self.pages else None,
        )
        return d


def _plain(text: str) -> str:
    """Rough normalisation so LaTeX and markdown transcriptions compare fairly."""
    text = re.sub(r"\\(text|mathrm|mathbf)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+|[$#*`{}\\]|\s+", "", text)
    return text.lower()


def reading_similarity(page: Page, reference: str) -> float:
    return difflib.SequenceMatcher(None, _plain(" ".join(s.latex for s in page.steps)), _plain(reference)).ratio()


def score_item(score: Score, item: EvalItem, page: Page, finding: Finding | None) -> str:
    """Update the score; return the outcome name."""
    score.pages += 1
    if item.reference:
        score.reading_similarity.append(reading_similarity(page, item.reference))
    label = item.first_error_step
    if label is None:
        return "read"
    if label == "none":
        if finding is None:
            score.clean_ok += 1
            return "clean_ok"
        score.false_alarms += 1
        return "false_alarm"
    if finding is None:
        score.missed += 1
        return "missed"
    idx = next((i for i, s in enumerate(page.steps) if s.id == finding.step_id), None)
    if label == "unknown" or idx == label:
        score.caught += 1
        return "caught"
    score.wrong_line += 1
    return "wrong_line"


def run(items: Iterable[EvalItem], pipeline: Pipeline, cost: Callable[[], float | None] = lambda: None,
        log: Callable[[str], None] = lambda s: None) -> dict[str, Score]:
    scores: dict[str, Score] = {}
    for item in items:
        score = scores.setdefault(item.source, Score(item.source))
        before = cost()
        start = time.monotonic()
        try:
            page, finding = pipeline(item.image, item.problem)
        except Exception as exc:  # one bad page must not stop the run
            score.failures += 1
            log(f"{item.source}/{item.id}: FAILED {type(exc).__name__}: {exc}")
            continue
        score.seconds += time.monotonic() - start
        after = cost()
        if score.cost_usd is not None:
            score.cost_usd = None if before is None or after is None else score.cost_usd + (after - before)
        outcome = score_item(score, item, page, finding)
        log(f"{item.source}/{item.id}: {outcome}")
    return scores


def format_scorecard(scores: dict[str, Score]) -> str:
    cols = ["source", "pages", "caught", "wrong_line", "missed", "false_alarms", "clean_ok", "failures",
            "catch_rate", "false_alarm_rate", "gate_passed", "reading_similarity", "cost_per_page",
            "seconds_per_page"]
    rows = [[str(s.summary()[c]) for c in cols] for s in scores.values()]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)] if rows else [len(c) for c in cols]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(cols), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def live_pipeline():
    from lens.checker import check_page
    from lens.config import Settings
    from lens.judge import spark_judge
    from lens.reader import read_page
    from lens.spark import SparkClient

    spark = SparkClient.from_settings(Settings.from_env())
    judge = spark_judge(spark)

    def pipeline(image: bytes, problem: str | None):
        page = read_page(spark, image, mime="image/png" if image[:4] == b"\x89PNG" else "image/jpeg")
        if problem and not page.problem:
            page.problem = problem
        return check_page(page, judge)

    return pipeline, lambda: spark.total_cost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muna Lens eval scorecard (reader + checker)")
    ap.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--limit", type=int, default=20, help="items per source")
    ap.add_argument("--shards", type=int, default=1, help="FERMAT parquet shards to fetch (~480 MB each)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)

    import itertools
    import logging

    logging.basicConfig(level=logging.WARNING)
    try:
        pipeline, cost = live_pipeline()
    except RuntimeError as exc:
        print(f"lens-eval: {exc}", file=sys.stderr)
        return 2
    sources = SOURCES if args.source == "all" else (args.source,)
    items = []
    for src in sources:
        kw = {"shards": args.shards} if src == "fermat" else {}
        try:
            items.extend(itertools.islice(load(src, **kw), args.limit))
        except RuntimeError as exc:
            print(f"skip {src}: {exc}", file=sys.stderr)
    scores = run(items, pipeline, cost, log=lambda s: print(s, file=sys.stderr))
    print(json.dumps({k: v.summary() for k, v in scores.items()}, indent=2) if args.json
          else format_scorecard(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
