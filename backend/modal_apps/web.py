"""The Lens app itself on Modal: real HTTPS, so a phone can pair from any network.

Deploy:  make deploy-web   (builds the frontend, then `modal deploy modal_apps/web.py`)

Keys come from the repo's `.env` as a Modal secret at deploy time; nothing secret is
baked into the image. Phone pairing keeps its sessions in memory, so the app runs as
one container that serves many connections.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "muna-lens-web"
# Only meaningful on the deploying machine; inside the container this file sits at /root.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
REMOTE = "/root/repo"

# Dependencies come from pyproject.toml + uv.lock (the same set the tests run with), never a
# hand-kept list: a hand list once missed `httpx` and every page check failed with a 500.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_file(REPO / "backend" / "pyproject.toml", f"{REMOTE}/backend/pyproject.toml", copy=True)
    .add_local_file(REPO / "backend" / "uv.lock", f"{REMOTE}/backend/uv.lock", copy=True)
    .run_commands(f"cd {REMOTE}/backend && uv export --frozen --no-dev --no-emit-project --extra gpu -o /tmp/req.txt"
                  " && uv pip install --system -r /tmp/req.txt")
    .env({"PYTHONPATH": f"{REMOTE}/backend/src", "LENS_CACHE_DIR": "/tmp/lens-cache"})
    .add_local_dir(REPO / "backend" / "src", f"{REMOTE}/backend/src")
    .add_local_dir(REPO / "frontend" / "dist", f"{REMOTE}/frontend/dist")
)
app = modal.App(APP_NAME)


@app.function(image=image, secrets=[modal.Secret.from_dotenv(REPO)], max_containers=1, region="us-east",
              scaledown_window=300, timeout=3600)  # CPU only; no GPU here
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    from lens.app import create_app

    return create_app()
