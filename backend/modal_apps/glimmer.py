"""Muse Glimmer on Modal: an OpenAI-compatible /v1/chat/completions (vLLM) that
accepts `image_url` data URLs. A self-hosted stand-in for Muse Spark Open when
the shared endpoint is slow on image calls.

Deploy:  GLIMMER_API_KEY=<random> make deploy-glimmer   (weights download once, on CPU)
Use:     SPARK_BASE_URL=https://<workspace>--muna-lens-glimmer-serve.us-east.modal.run/v1
         SPARK_MODEL=muse-glimmer  SPARK_API_KEY=<the same random key>

Weights: RedHatAI/Muse-Glimmer-30B-FP8-block (FP8 block-scaled, ~33 GB) plus
Meta's DFlash draft head (meta-models/Muse-Glimmer-30B-assistant, ~5 GB) for
speculative decoding, on one L40S (48 GB, ~$1.95/h). Scales to zero after 180 s
idle; the next request waits for a cold start (measured 229 s). Warm page
reads measured 8-14 s.

`reasoning_effort` (OpenAI style, as the Spark client sends it) is mapped onto
Glimmer's own "Reasoning strength" by a small chat-template patch below.
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "muna-lens-glimmer"
MODEL_ID = os.environ.get("GLIMMER_MODEL_ID", "RedHatAI/Muse-Glimmer-30B-FP8-block")
DRAFT_ID = "meta-models/Muse-Glimmer-30B-assistant"
SERVED_NAME = "muse-glimmer"
GPU = os.environ.get("GLIMMER_GPU", "L40S")
PORT = 8000
MAX_MODEL_LEN = 16384  # a page read is ~1-2k prompt + up to 6k out; the L40S has little room for KV cache

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.28.0", add_python=None)
    .entrypoint([])
    # The vLLM image ships python3 but no `python`, which Modal looks for.
    .dockerfile_commands(['RUN ln -sf "$(command -v python3)" /usr/local/bin/python && python --version'])
    .env({"HF_HOME": "/cache/hf", "VLLM_CACHE_ROOT": "/cache/vllm", "HF_XET_HIGH_PERFORMANCE": "1"})
)
cache = modal.Volume.from_name("muna-lens-glimmer-cache", create_if_missing=True)
app = modal.App(APP_NAME)
secrets = [modal.Secret.from_dict({
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "GLIMMER_API_KEY": os.environ.get("GLIMMER_API_KEY", ""),
    # Deploy-time choices travel to the container as environment.
    "GLIMMER_MODEL_ID": MODEL_ID,
    "GLIMMER_SPECULATIVE": os.environ.get("GLIMMER_SPECULATIVE", "1"),
})]

# The stock template reads `reasoning_strength` (low/medium/high/xhigh) and
# defaults to high. vLLM passes the request's `reasoning_effort` to the template
# as `reasoning_effort`; map it, with "minimal" -> "low".
_TEMPLATE_OLD = "{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}"
_TEMPLATE_NEW = (
    "{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else "
    "(reasoning_effort if reasoning_effort is defined and reasoning_effort else 'high') -%}"
    "{%- if rs == 'minimal' or rs == 'none' -%}{%- set rs = 'low' -%}{%- endif -%}"
)


def _template(model_dir: str) -> str:
    src = open(os.path.join(model_dir, "chat_template.jinja")).read()
    if _TEMPLATE_OLD not in src:
        raise RuntimeError("chat template changed upstream; update the reasoning_effort patch")
    path = "/tmp/glimmer_template.jinja"
    with open(path, "w") as f:
        f.write(src.replace(_TEMPLATE_OLD, _TEMPLATE_NEW))
    return path


@app.function(image=image, volumes={"/cache": cache}, secrets=secrets, timeout=3600, cpu=4, memory=8192)
def download() -> None:
    """Fetch the weights into the volume once (CPU only, so no GPU time is spent on it)."""
    from huggingface_hub import snapshot_download

    for repo in (MODEL_ID, DRAFT_ID):
        snapshot_download(repo, token=os.environ.get("HF_TOKEN") or None)
    cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=secrets,
    scaledown_window=180,
    timeout=1800,
    max_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=900)
def serve() -> None:
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL_ID, token=os.environ.get("HF_TOKEN") or None)
    cmd = [
        "vllm", "serve", model_dir,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.95",
        "--reasoning-parser", "muse_glimmer",
        "--generation-config", "auto",
        "--enable-prefix-caching",
        "--chat-template", _template(model_dir),
        "--limit-mm-per-prompt", '{"image": 2}',
    ]
    env = dict(os.environ)
    if os.environ.get("GLIMMER_API_KEY"):
        # Through the environment, not --api-key: vLLM prints its CLI args to the log.
        env["VLLM_API_KEY"] = os.environ["GLIMMER_API_KEY"]
    if os.environ.get("GLIMMER_SPECULATIVE", "1") == "1":
        cmd += ["--speculative-config",
                '{"method": "dflash", "model": "%s", "num_speculative_tokens": 15}' % DRAFT_ID]
    subprocess.Popen(cmd, env=env)
