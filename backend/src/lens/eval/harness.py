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
from pathlib import Path

from lens.contract import Finding, Page
from lens.eval.loaders import SOURCES, EvalItem, load

# pipeline(image, problem) -> (checked page, first finding or None). A pipeline may
# also accept trace= (a dict it fills with what the confirmation saw); it is saved.
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
    # Pages the dataset mislabels as correct (loaders.FERMAT_LABEL_ERRORS): scored apart.
    disputed_flagged: int = 0
    disputed_quiet: int = 0
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
            raw_false_alarms=self.false_alarms + self.disputed_flagged,
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
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # markdown image links
    # Expert references describe drawings in blockquotes ("> A circuit diagram..."); Lens
    # transcribes working, not pictures, so those descriptions are not compared.
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(text|mathrm|mathbf)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+|[$#*`{}\\]|\s+", "", text)
    return text.lower()


def reading_similarity(page: Page, reference: str) -> float:
    # autojunk off: on long strings it discards frequent characters and sinks the score
    return difflib.SequenceMatcher(None, _plain(" ".join(s.latex for s in page.steps)), _plain(reference),
                                   autojunk=False).ratio()


def score_item(score: Score, item: EvalItem, page: Page, finding: Finding | None) -> str:
    """Update the score; return the outcome name."""
    score.pages += 1
    if item.reference:
        score.reading_similarity.append(reading_similarity(page, item.reference))
    label = item.first_error_step
    if label is None:
        return "read"
    if label == "disputed":
        if finding is None:
            score.disputed_quiet += 1
            return "disputed_quiet"
        score.disputed_flagged += 1
        return "disputed_flagged"
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
        log: Callable[[str], None] = lambda s: None, workers: int = 1,
        record: Callable[[EvalItem, Page, Finding | None, str, dict], None] | None = None) -> dict[str, Score]:
    """Score every item. With workers > 1 pages run in parallel; cost is then
    the total spend split evenly over pages."""
    from concurrent.futures import ThreadPoolExecutor

    scores: dict[str, Score] = {}
    items = list(items)
    total_before = cost() if workers > 1 else None

    import inspect

    traced = "trace" in inspect.signature(pipeline).parameters
    traces: dict[int, dict] = {}

    def one(item: EvalItem):
        start = time.monotonic()
        before = cost() if workers == 1 else None
        trace = traces.setdefault(id(item), {})
        try:
            page, finding = pipeline(item.image, item.problem, trace=trace) if traced else pipeline(
                item.image, item.problem)
        except Exception as exc:  # one bad page must not stop the run
            return item, None, None, time.monotonic() - start, exc, None
        after = cost() if workers == 1 else None
        spent = None if before is None or after is None else after - before
        return item, page, finding, time.monotonic() - start, None, spent

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for item, page, finding, secs, exc, spent in pool.map(one, items):
            score = scores.setdefault(item.source, Score(item.source))
            if exc is not None:
                score.failures += 1
                log(f"{item.source}/{item.id}: FAILED {type(exc).__name__}: {exc}")
                continue
            score.seconds += secs
            if workers == 1 and score.cost_usd is not None:
                score.cost_usd = None if spent is None else score.cost_usd + spent
            outcome = score_item(score, item, page, finding)
            if record:
                record(item, page, finding, outcome, traces.get(id(item), {}))
            log(f"{item.source}/{item.id}: {outcome}")
    if workers > 1:
        total_after = cost()
        n = sum(s.pages for s in scores.values())
        for s in scores.values():
            s.cost_usd = (None if total_before is None or total_after is None or not n
                          else (total_after - total_before) * s.pages / n)
    return scores


def format_scorecard(scores: dict[str, Score]) -> str:
    cols = ["source", "pages", "caught", "wrong_line", "missed", "false_alarms", "clean_ok", "disputed_flagged",
            "raw_false_alarms", "failures",
            "catch_rate", "false_alarm_rate", "gate_passed", "reading_similarity", "cost_per_page",
            "seconds_per_page"]
    rows = [[str(s.summary()[c]) for c in cols] for s in scores.values()]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)] if rows else [len(c) for c in cols]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(cols), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def live_pipeline(read_only: bool = False):
    from lens.config import Settings
    from lens.pipeline import check_image
    from lens.reader import read_page
    from lens.spark import SparkClient

    spark = SparkClient.from_settings(Settings.from_env())

    def pipeline(image: bytes, problem: str | None, trace: dict | None = None):
        if read_only:  # reading-only sources (EDU-CIRCUIT-HW) have no error labels
            return read_page(spark, image), None
        return check_image(spark, image, problem, hint=False, trace=trace)

    return pipeline, lambda: spark.total_cost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muna Lens eval scorecard (reader + checker)")
    ap.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--limit", type=int, default=20, help="items per source")
    ap.add_argument("--shards", type=int, default=1, help="FERMAT parquet shards to fetch (~480 MB each)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--balanced", action="store_true", help="FERMAT: half error pages, half clean, seeded sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-seed", type=int, help="FERMAT: leave out the pages that --seed N with the same "
                    "--limit would pick (a held-out sample, disjoint from the tuning one)")
    ap.add_argument("--only", choices=["clean", "error"], help="run only the clean (or error) pages of the "
                    "sample, e.g. to measure false alarms on a held-out sample for half the cost")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--save", help="write per-page results (JSONL) here, e.g. .cache/eval/runs/x.jsonl")
    args = ap.parse_args(argv)

    import itertools
    import logging

    logging.basicConfig(level=logging.WARNING)
    try:
        pipeline, cost = live_pipeline()
        reader, reader_cost = live_pipeline(read_only=True)
    except RuntimeError as exc:
        print(f"lens-eval: {exc}", file=sys.stderr)
        return 2
    sources = SOURCES if args.source == "all" else (args.source,)
    items = []
    for src in sources:
        kw = {"shards": args.shards} if src == "fermat" else {}
        if src == "edu_circuit":
            from lens.eval.loaders import eval_cache, fetch_edu_circuit_sample

            if not (eval_cache() / "edu_circuit").is_dir():
                fetch_edu_circuit_sample(args.limit)
        if src == "fermat" and args.balanced:
            from lens.eval.loaders import eval_cache, sample_fermat_ids

            exclude = (sample_fermat_ids(eval_cache(), args.limit, args.exclude_seed, args.shards)
                       if args.exclude_seed is not None else None)
            kw["ids"] = sample_fermat_ids(eval_cache(), args.limit, args.seed, args.shards, exclude=exclude)
        try:
            items.extend(itertools.islice(load(src, **kw), args.limit))
        except RuntimeError as exc:
            print(f"skip {src}: {exc}", file=sys.stderr)
    if args.only:
        keep = ("none", "disputed") if args.only == "clean" else ("unknown",)
        items = [i for i in items if i.first_error_step is None or i.first_error_step in keep
                 or (args.only == "error" and isinstance(i.first_error_step, int))]
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.save, "w") if args.save else None  # noqa: SIM115

    def record(item, page, finding, outcome, trace):
        if out:
            out.write(json.dumps({"source": item.source, "id": item.id, "label": item.first_error_step,
                                  "outcome": outcome, "page": page.model_dump(),
                                  "finding": finding.model_dump() if finding else None, "trace": trace,
                                  "meta": {k: v for k, v in item.meta.items() if isinstance(v, (str, int, float))}})
                      + "\n")
            out.flush()

    log = lambda s: print(s, file=sys.stderr)
    scores = run([i for i in items if i.first_error_step is not None], pipeline, cost, log=log,
                 workers=args.workers, record=record)
    scores.update(run([i for i in items if i.first_error_step is None], reader, reader_cost, log=log,
                      workers=args.workers, record=record))
    if out:
        out.close()
    print(json.dumps({k: v.summary() for k, v in scores.items()}, indent=2) if args.json
          else format_scorecard(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
