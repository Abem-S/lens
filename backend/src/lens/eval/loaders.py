"""E0 Eval sets. Each loader yields EvalItem(image, problem, first_error_step | "none").

Datasets are fetched at run time into .cache/eval/ (gitignored) and never committed.
- FERMAT: Hugging Face, gated (accept the agreement, then set HF_TOKEN).
- EDU-CIRCUIT-HW: download the release zip from https://doi.org/10.35090/gatech/81252
  and extract it to .cache/eval/edu_circuit/. Private testing only.
- Own set: eval/own/labels.json; images in eval/own/ (consented) or .cache/eval/own/.
"""

from __future__ import annotations

import io
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
    # when the page has an error but the dataset does not say which line; "disputed"
    # when the dataset's label is wrong (FERMAT_LABEL_ERRORS); None for reading-only
    # items (no error label).
    first_error_step: int | Literal["none", "unknown", "disputed"] | None
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

# Pages FERMAT labels correct whose handwriting carries a real mathematical mistake,
# checked by hand against the photo. Most inherit it from FERMAT's own reference
# solution. Flagging one is not a false alarm, and missing it is not a miss, so they
# are scored apart ("disputed"); the raw FERMAT count is still printed.
FERMAT_LABEL_ERRORS = {
    "img_296_pert_5.3": "the added line '= 1960000 - 8200 + 9' does not equal the line above "
                        "(1960000 + 9 - 8400)",
    "img_423_pert_5.3": "the page writes (1/6) \\int t^{-1/2} dt and then (1/6)(2/3) t^{3/2}; with t^{-1/2} "
                        "the integral is 2 t^{1/2} (the reference solution has the same slip)",
    "img_441_pert_5.1": "the page claims f''(u) >= 0 for all u, but f''(u) = 12u - 12 "
                        "(the reference solution says f''(x) where f'(x) is meant)",
    "img_28_pert_5.3": "the added note writes (-2 + 6) + (4 + 8) = 2 + 12 = 14; -2 + 6 is 4",
    "img_558_pert_5.2": "the table gives z^2 as the coefficient of y in y^2 + 5 (copied from the reference "
                        "solution); y^2 + 5 has no z",
}


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


def sample_fermat_ids(cache: Path, n: int, seed: int = 0, shards: int | None = None,
                      exclude: set[str] | None = None) -> set[str]:
    """Pick n/2 pages with a planted error and n/2 clean pages, reproducibly.

    `exclude` keeps a held-out sample apart from the one a change was tuned on."""
    import random

    import pyarrow.parquet as pq

    rows = []
    for shard in fermat_shards(cache, shards):
        rows += pq.read_table(shard, columns=["new_custom_id", "has_error"]).to_pylist()
    rows = [r for r in rows if not exclude or r["new_custom_id"] not in exclude]
    rng = random.Random(seed)
    err = sorted(r["new_custom_id"] for r in rows if r["has_error"])
    clean = sorted(r["new_custom_id"] for r in rows if not r["has_error"])
    k = n // 2
    return set(rng.sample(err, min(k, len(err))) + rng.sample(clean, min(n - k, len(clean))))


def load_fermat(cache: Path | None = None, shards: int | None = None, ids: set[str] | None = None,
                ) -> Iterator[EvalItem]:
    import pyarrow.parquet as pq

    cache = cache or eval_cache()
    for shard in fermat_shards(cache, shards):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if ids is not None and str(row.get("new_custom_id")) not in ids:
                    continue
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
                    first_error_step=("disputed" if str(row.get("new_custom_id")) in FERMAT_LABEL_ERRORS
                                      else "unknown" if has_error else "none"),
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


EDU_CIRCUIT_ZIP = "https://repository.gatech.edu/server/api/core/bitstreams/09496897-10b3-4675-bcf5-73b5b9802cd2/content"


class _HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests,
    so zipfile can pull a few members out of a 9 GB archive."""

    def __init__(self, url: str, block: int = 1 << 20):
        import httpx

        self.url, self.block, self.pos = url, block, 0
        self.client = httpx.Client(follow_redirects=True, timeout=120)
        head = self.client.get(url, headers={"Range": "bytes=0-0"})
        self.size = int(head.headers["content-range"].split("/")[-1])
        self.cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            self.cache[i] = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"}).content
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_edu_circuit_sample(n: int = 15, seed: int = 0, cache: Path | None = None) -> Path:
    """Extract n expert-rectified (image, transcript) pairs from the release zip
    without downloading all of it. Private testing only; never committed."""
    import random
    import zipfile

    root = (cache or eval_cache()) / "edu_circuit"
    zf = zipfile.ZipFile(_HttpRangeFile(EDU_CIRCUIT_ZIP))
    names = set(zf.namelist())
    md = sorted(x for x in names if "Rectified_recognized_markdown_done_Anon/Final_4_LLM_judge/" in x
                and x.endswith("_markdown.md") and x.replace("_markdown.md", "_source.png") in names)
    for name in random.Random(seed).sample(md, min(n, len(md))):
        for member in (name, name.replace("_markdown.md", "_source.png")):
            target = root / member
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
    return root
