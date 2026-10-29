"""C1 done check (live, costs GPU time): a mask for a step bbox and for "fingertip"
on test photos; warm latency measured and logged.

    uv run --extra gpu --extra eval python scripts/c1_check.py
Photos come from the gitignored cache (.cache/eval/hands, .cache/eval/fermat)."""

import glob
import statistics
import time
from pathlib import Path

from lens.config import REPO_ROOT
from lens.eval.loaders import load_fermat
from lens.pointing import fingertip_from_mask
from lens.sam import SamClient


def reader_boxes(n_pages: int = 3) -> list[tuple[bytes, list]]:
    """Pages read by A2 in the last saved eval run, with their step boxes."""
    import json

    runs = sorted(glob.glob(str(REPO_ROOT / ".cache/eval/runs/*.jsonl")), key=lambda p: Path(p).stat().st_mtime)
    rows = [json.loads(line) for line in Path(runs[-1]).read_text().splitlines()]
    rows = [r for r in rows if r["source"] == "fermat" and len(r["page"]["steps"]) >= 3][:n_pages]
    want = {r["id"]: r for r in rows}
    out = []
    for item in load_fermat(ids=set(want)):
        out.append((item.image, [s["bbox"] for s in want[item.id]["page"]["steps"]]))
    return out


def iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def main() -> None:
    sam = SamClient()
    t = time.monotonic()
    sam._obj.ping.remote()
    print(f"cold start (or already warm): {time.monotonic() - t:.1f}s")

    got, tried = 0, 0
    for image, boxes in reader_boxes():
        for box in boxes[:4]:
            tried += 1
            masks = sam.segment(image, boxes=[box], max_masks=1)
            if masks:
                got += 1
                print(f"step box {[round(v, 3) for v in box]} -> mask box {[round(v, 3) for v in masks[0].box]} "
                      f"score {masks[0].score:.2f} IoU {iou(box, masks[0].box):.2f}")
            else:
                print(f"step box {[round(v, 3) for v in box]} -> NO MASK")
    print(f"masks for {got}/{tried} reader step boxes")

    hits, lat = 0, []
    for path in sorted(glob.glob(str(REPO_ROOT / ".cache/eval/hands/*.jpg"))):
        data = Path(path).read_bytes()
        t = time.monotonic()
        ms = sam.segment(data, text="fingertip")
        lat.append(time.monotonic() - t)
        if not ms:
            ms = sam.segment(data, text="index finger")
        tip = fingertip_from_mask(ms[0].mask) if ms else None
        hits += tip is not None
        print(f"{path.rsplit('/', 1)[1]}: {len(ms)} masks, best {ms[0].score:.2f} box "
              f"{tuple(round(v, 2) for v in ms[0].box)} tip {tuple(round(v, 2) for v in tip)}" if ms else
              f"{path}: no mask")
    print(f"fingertip found on {hits}/{len(lat)} photos; warm latency median {statistics.median(lat):.2f}s "
          f"(min {min(lat):.2f}, max {max(lat):.2f})")


if __name__ == "__main__":
    main()
