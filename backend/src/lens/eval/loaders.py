"""E0 Eval sets. Each loader yields EvalItem(image, problem, first_error_step | "none").

Datasets are fetched at run time into .cache/eval/ (gitignored) and never committed.
- FERMAT: Hugging Face, gated (accept the agreement, then set HF_TOKEN).
- EDU-CIRCUIT-HW: download the release zip from https://doi.org/10.35090/gatech/81252
  and extract it to .cache/eval/edu_circuit/. Private testing only.
- Own set: eval/own/labels.json; images in eval/own/ (consented) or .cache/eval/own/.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lens.config import REPO_ROOT, Settings

Source = Literal["fermat", "edu_circuit", "own"]
SOURCES: tuple[Source, ...] = ("fermat", "edu_circuit", "own")


@dataclass
class EvalItem:
    source: Source
    id: str
    load_image: Callable[[], bytes] = field(repr=False)
    problem: str | None
    # 0-based index of the first wrong line; "none" for a correct page; "unknown"
    # when the page has an error but the dataset does not say which line;
    # None for reading-only items (no error label).
    first_error_step: int | Literal["none", "unknown"] | None
    reference: str | None = None  # expert transcription, for reading accuracy
    meta: dict = field(default_factory=dict)

    @property
    def image(self) -> bytes:
        return self.load_image()

    @property
    def has_error(self) -> bool | None:
        if self.first_error_step is None:
            return None
        return self.first_error_step != "none"


def eval_cache(settings: Settings | None = None) -> Path:
    return (settings or Settings.from_env()).cache_dir / "eval"


# ---------------- FERMAT ----------------

FERMAT_REPO = "ai4bharat/FERMAT"
FERMAT_SHARDS = 10


def fermat_shards(cache: Path, shards: int | None = None) -> list[Path]:
    """Download (once) and return local parquet shards. Each shard is ~480 MB."""
    local = cache / "fermat"
    have = sorted(local.glob("data/train-*.parquet"))
    want = shards or FERMAT_SHARDS
    if len(have) >= want:
        return have[:want]
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("FERMAT is gated: accept its agreement on Hugging Face and set HF_TOKEN in .env")
    return [
        Path(hf_hub_download(FERMAT_REPO, f"data/train-{i:05d}-of-{FERMAT_SHARDS:05d}.parquet",
                             repo_type="dataset", local_dir=local, token=token))
        for i in range(want)
    ]


def load_fermat(cache: Path | None = None, shards: int | None = None) -> Iterator[EvalItem]:
    import pyarrow.parquet as pq

    cache = cache or eval_cache()
    for shard in fermat_shards(cache, shards):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                img = row["image"]
                data = img["bytes"] if isinstance(img, dict) else img
                has_error = bool(row.get("has_error"))
                yield EvalItem(
                    source="fermat",
                    id=str(row.get("new_custom_id")),
                    load_image=lambda d=data: d,
                    problem=row.get("orig_q"),
                    # FERMAT labels the error, not the line; the reasoning text
                    # describes where it is.
                    first_error_step="unknown" if has_error else "none",
                    reference=row.get("pert_a") if has_error else row.get("orig_a"),
                    meta={k: row.get(k) for k in ("grade", "domain_code", "pert_reasoning", "rotation")},
                )


# ---------------- EDU-CIRCUIT-HW ----------------

def _qid(name: str) -> str:
    return re.sub(r"_(markdown|source)\.(md|png)$", "", name)


def load_edu_circuit(cache: Path | None = None) -> Iterator[EvalItem]:
    """Observation-set transcripts (Gemini 2.5 v6), replaced by the expert-rectified
    version where one exists. Reading accuracy only: no error labels."""
    root = (cache or eval_cache()) / "edu_circuit"
    if not root.is_dir():
        raise RuntimeError(f"extract the EDU-CIRCUIT-HW release into {root} (https://doi.org/10.35090/gatech/81252)")
    # Some zips extract into one extra folder level.
    obs = next(root.glob("**/Observationset_Final/v6_Gemini_2p5"), None)
    rect = next(root.glob("**/Rectified_recognized_markdown_done_Anon/Final_4_LLM_judge"), None)

    def pairs(base: Path | None) -> dict[str, tuple[Path, Path]]:
        out = {}
        if base is None:
            return out
        for md in base.glob("*/models/*/Compare/*_markdown.md"):
            png = md.with_name(md.name.replace("_markdown.md", "_source.png"))
            if png.is_file():
                student = md.parts[-5].replace("Homework_collected_database_trial_", "")
                out[f"{student}/{_qid(md.name)}"] = (png, md)
        return out

    items = pairs(obs)
    rectified = pairs(rect)
    items.update(rectified)
    for key in sorted(items):
        png, md = items[key]
        yield EvalItem(
            source="edu_circuit", id=key, load_image=lambda p=png: p.read_bytes(), problem=None,
            first_error_step=None, reference=md.read_text(encoding="utf-8", errors="replace"),
            meta={"rectified": key in rectified},
        )


# ---------------- Own set ----------------

OWN_DIR = REPO_ROOT / "eval" / "own"


def load_own(labels: Path = OWN_DIR / "labels.json", cache: Path | None = None) -> Iterator[EvalItem]:
    """Pages listed in labels.json; images may live next to it (consented) or in the cache."""
    if not labels.is_file():
        return
    search = [labels.parent, (cache or eval_cache()) / "own"]
    for entry in json.loads(labels.read_text()):
        path = next((d / entry["file"] for d in search if (d / entry["file"]).is_file()), None)
        if path is None:
            continue
        fes = entry.get("first_error_step", "none")
        yield EvalItem(
            source="own", id=entry["file"], load_image=lambda p=path: p.read_bytes(),
            problem=entry.get("problem"), first_error_step=fes if fes == "none" else int(fes),
            reference=entry.get("transcription"),
            meta={k: entry.get(k) for k in ("subject", "camera", "conditions")},
        )


def load(source: Source, **kw) -> Iterator[EvalItem]:
    return {"fermat": load_fermat, "edu_circuit": load_edu_circuit, "own": load_own}[source](**kw)
