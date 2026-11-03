"""Settings from the environment, with an optional `.env` at the repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from lens.limits import Limits

REPO_ROOT = Path(__file__).resolve().parents[3]

# Estimated Spark price (USD per million tokens) from a third-party listing for
# muse-spark-open-1026; Baseten's own price is not confirmed yet. Override in .env.
DEFAULT_PRICE_IN = 1.25
DEFAULT_PRICE_OUT = 4.25
# Estimated Modal GPU price per hour, for the per-session cost tally.
DEFAULT_GPU_RATES = {"l40s": 1.95, "l4": 0.80}


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment wins."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _float(name: str, default: float | None = None) -> float | None:
    v = os.environ.get(name)
    return float(v) if v else default


@dataclass(frozen=True)
class Settings:
    # Key for the Spark-compatible endpoint: SPARK_API_KEY when set (e.g. self-hosted
    # Muse Glimmer on Modal), else BASETEN. Field name kept for existing callers.
    baseten_key: str | None
    spark_base_url: str
    spark_model: str
    # USD per million tokens (from_env defaults to the estimate above; None = cost unknown).
    spark_price_in: float | None
    spark_price_out: float | None
    public_url: str | None  # where phones reach the app, e.g. https://lens.example.com
    cache_dir: Path
    modal_configured: bool = False  # SAM + voice services on Modal
    access_code: str | None = None  # tester gate; None = open
    limits: Limits = field(default_factory=Limits)
    gpu_rates: dict = field(default_factory=lambda: dict(DEFAULT_GPU_RATES))  # USD per hour, estimate

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            baseten_key=os.environ.get("SPARK_API_KEY") or os.environ.get("BASETEN") or None,
            spark_base_url=os.environ.get("SPARK_BASE_URL", "https://inference.baseten.co/v1"),
            spark_model=os.environ.get("SPARK_MODEL", "meta-models/muse-spark-open-1026"),
            spark_price_in=_float("SPARK_PRICE_IN_PER_M", DEFAULT_PRICE_IN),
            spark_price_out=_float("SPARK_PRICE_OUT_PER_M", DEFAULT_PRICE_OUT),
            public_url=os.environ.get("PUBLIC_URL") or None,
            cache_dir=Path(os.environ.get("LENS_CACHE_DIR", REPO_ROOT / ".cache")),
            modal_configured=bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
            and os.environ.get("LENS_USE_MODAL", "1") != "0",
            access_code=os.environ.get("LENS_ACCESS_CODE") or None,
            limits=Limits.from_env(),
            gpu_rates={"l40s": _float("LENS_GPU_PRICE_L40S_PER_H", DEFAULT_GPU_RATES["l40s"]),
                       "l4": _float("LENS_GPU_PRICE_L4_PER_H", DEFAULT_GPU_RATES["l4"])},
        )
