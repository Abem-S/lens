"""Client for the C1 SAM service on Modal (modal_apps/sam.py)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("lens.sam")
APP_NAME = "muna-lens-sam"


@dataclass
class Mask:
    score: float
    box: tuple[float, float, float, float]  # normalised x, y, w, h
    mask: np.ndarray  # bool, image-sized


def _unpack(m: dict) -> Mask:
    h, w = m["shape"]
    bits = np.unpackbits(np.frombuffer(m["bits"], dtype=np.uint8))[: h * w]
    return Mask(m["score"], tuple(m["box"]), bits.reshape(h, w).astype(bool))


class SamClient:
    def __init__(self) -> None:
        import modal

        self._cls = modal.Cls.from_name(APP_NAME, "Sam")
        self._obj = self._cls()

    def keep_warm(self, on: bool) -> None:
        """During a session keep one GPU container up (cold start is ~30-60 s)."""
        self._obj.update_autoscaler(min_containers=1 if on else 0)
        log.info("sam keep_warm=%s", on)

    def segment(self, image: bytes, *, text: str | None = None,
                boxes: list[tuple[float, float, float, float]] | None = None,
                threshold: float = 0.4, max_masks: int = 5) -> list[Mask]:
        t = time.monotonic()
        out = self._obj.segment.remote(image, text=text, boxes=[list(b) for b in boxes] if boxes else None,
                                       threshold=threshold, max_masks=max_masks)
        log.info("sam segment text=%r boxes=%d masks=%d gpu=%.2fs total=%.2fs", text, len(boxes or []),
                 len(out["masks"]), out["seconds"], time.monotonic() - t)
        return [_unpack(m) for m in out["masks"]]
