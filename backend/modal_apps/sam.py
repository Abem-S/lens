"""C1 SAM service on Modal: image + (boxes or a short noun phrase) -> masks.

Deploy:  cd backend && uv run --extra gpu modal deploy modal_apps/sam.py
Client:  lens.sam (keeps the container warm during a session).

SAM 3 (`facebook/sam3`) through Transformers. SAM 3.1's checkpoint only loads in
Meta's own `sam3` package, and its gain (Object Multiplex) is faster multi-object
video tracking, which still frames do not use. See AGENTS.md.
"""

from __future__ import annotations

import io
import os
import time

import modal

APP_NAME = "muna-lens-sam"
MODEL_ID = "facebook/sam3"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers>=5.0", "accelerate", "pillow", "numpy", "huggingface_hub[hf_xet]")
    .env({"HF_HOME": "/cache/hf"})
)
cache = modal.Volume.from_name("muna-lens-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=["L4", "A10G", "L40S"],  # first free one wins
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    scaledown_window=120,
    timeout=600,
)
class Sam:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        t = time.monotonic()
        self.torch = torch
        self.model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        cache.commit()
        self.load_seconds = time.monotonic() - t

    @modal.method()
    def segment(
        self,
        image_bytes: bytes,
        text: str | None = None,
        boxes: list[list[float]] | None = None,  # normalised [x, y, w, h], like Step.bbox
        threshold: float = 0.4,
        max_masks: int = 5,
    ) -> dict:
        """Masks for the prompt, best first. Each mask is packed bits (np.packbits) + shape."""
        import numpy as np
        from PIL import Image

        t = time.monotonic()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        kwargs: dict = {"images": img, "return_tensors": "pt"}
        if text:
            kwargs["text"] = text
        if boxes:
            kwargs["input_boxes"] = [[[x * w, y * h, (x + bw) * w, (y + bh) * h] for x, y, bw, bh in boxes]]
            kwargs["input_boxes_labels"] = [[1] * len(boxes)]
        inputs = self.processor(**kwargs).to("cuda")
        for k, v in inputs.items():
            if hasattr(v, "dtype") and v.dtype == self.torch.float32:
                inputs[k] = v.to(self.torch.bfloat16)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        order = res["scores"].float().argsort(descending=True)[:max_masks].tolist()
        masks = []
        for i in order:
            m = res["masks"][i].cpu().numpy().astype(bool)
            x0, y0, x1, y1 = res["boxes"][i].float().tolist()
            masks.append({
                "score": float(res["scores"][i]),
                "box": [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                "shape": list(m.shape),
                "bits": np.packbits(m).tobytes(),
            })
        return {"masks": masks, "seconds": time.monotonic() - t, "size": [w, h]}

    @modal.method()
    def ping(self) -> dict:
        return {"ok": True, "load_seconds": self.load_seconds}
