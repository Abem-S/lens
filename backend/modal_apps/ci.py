"""Run the backend tests and the eval scorecard on Modal instead of the laptop.

    make test-modal            # pytest in a cloud container
    make eval-modal            # A5 scorecard in a cloud container (same flags as lens-eval)

The working tree is copied in at run time, so uncommitted changes are tested too.
Keys come from the repo's `.env` as a Modal secret; nothing secret is baked in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

# Only meaningful on the deploying machine; inside the container this file sits at /root.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
REMOTE = "/root/repo"
IGNORE = [".venv", "**/__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".cache"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir(REPO / "backend", f"{REMOTE}/backend", copy=True, ignore=IGNORE)
    .run_commands(f"cd {REMOTE}/backend && uv sync --extra eval --extra gpu")
    .add_local_dir(REPO / "backend", f"{REMOTE}/backend", ignore=IGNORE + ["uv.lock", "pyproject.toml"])
    .add_local_dir(REPO / "eval", f"{REMOTE}/eval")
)
cache = modal.Volume.from_name("muna-lens-eval-cache", create_if_missing=True)
app = modal.App("muna-lens-ci")


def _run(cmd: str) -> int:
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=f"{REMOTE}/backend", check=False).returncode


@app.function(image=image, secrets=[modal.Secret.from_dotenv(REPO)], timeout=1800, cpu=2)
def pytest(args: str = "-q") -> int:
    return _run(f"uv run --extra eval pytest {args}")


@app.function(image=image, secrets=[modal.Secret.from_dotenv(REPO)], volumes={"/cache": cache},
              timeout=3 * 3600, cpu=2)
def scorecard(args: str = "--source all --limit 20") -> int:
    code = _run(f"LENS_CACHE_DIR=/cache uv run --extra eval lens-eval {args}")
    cache.commit()
    return code


@app.local_entrypoint()
def main(task: str = "test", args: str = "") -> None:
    fn = {"test": pytest, "eval": scorecard}[task]
    code = fn.remote(args) if args else fn.remote()
    raise SystemExit(code)
