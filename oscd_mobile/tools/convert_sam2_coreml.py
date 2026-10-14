#!/usr/bin/env python
"""Convert SAM2.1-hiera-tiny's image encoder to Core ML exactly as O-SCD calls it.

O-SCD feeds raw [0, 1] images (no ImageNet normalisation), nearest-resized to 1024×1024, as one
batch [render, live], and keeps get_image_embeddings(...)[-1] (last FPN level plus the
no-memory embedding). The encoder has no cross-batch ops, so the model below takes one image
(batch 1): the app encodes the live image while its pose is estimated, and two batch-1 calls are
faster on the Neural Engine than one batch-2 call.
"""
import argparse
import os

import coremltools as ct
import numpy as np
import torch
from transformers import Sam2Model


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.get_image_embeddings(images)[-1]


def freeze_pos_embed(model, example):
    """Hiera resizes its position embedding with bicubic interpolation (no Core ML converter). The
    result depends only on weights and the fixed 1024×1024 input, so compute it once and serve it as
    a constant buffer."""
    for mod in model.modules():
        if hasattr(mod, "_get_pos_embed") and hasattr(mod, "pos_embed"):
            original, cache = mod._get_pos_embed, {}

            def capture(hw, original=original, cache=cache):
                cache[tuple(int(v) for v in hw)] = original(hw).detach()
                return cache[tuple(int(v) for v in hw)]

            mod._get_pos_embed = capture
            with torch.no_grad():
                model.get_image_embeddings(example)
            assert len(cache) == 1, f"position embedding requested at sizes {list(cache)}"
            mod.register_buffer("frozen_pos_embed", next(iter(cache.values())))
            mod._get_pos_embed = lambda hw, mod=mod: mod.frozen_pos_embed


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "..", "Models"))
    ap.add_argument("--fixtures", default=os.path.join(here, "..", "fixtures"))
    a = ap.parse_args()

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval()
    enc = Encoder(model).eval()
    example = torch.rand(1, 3, 1024, 1024)
    with torch.no_grad():
        before = enc(example)
        assert tuple(before.shape) == (1, 256, 64, 64)
        freeze_pos_embed(model, example)
        assert torch.allclose(enc(example), before, atol=1e-5), "freezing the position embedding changed the output"
        traced = torch.jit.trace(enc, example, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, 1024, 1024), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "SAM2_1Tiny_1x1024.mlpackage")
    ml.save(path)
    print("saved", path)

    x_path = os.path.join(a.fixtures, "cues", "frame_000", "sam_input.npy")
    if not os.path.exists(x_path):
        print("no fixtures; skipped parity check")
        return
    x = np.load(x_path).astype(np.float32)
    got = np.concatenate([ml.predict({"images": x[i:i + 1]})["embedding"] for i in range(len(x))])
    want = np.load(os.path.join(a.fixtures, "cues", "frame_000", "emb.npy"))
    d_got = np.abs(got[0] - got[1]).mean(axis=0).ravel()
    d_want = np.abs(want[0] - want[1]).mean(axis=0).ravel()
    r_emb = np.corrcoef(got.ravel(), want.ravel())[0, 1]
    r_diff = np.corrcoef(d_got, d_want)[0, 1]
    print(f"embedding r = {r_emb:.4f}, feature-difference map r = {r_diff:.4f}")
    assert r_diff >= 0.98, "feature-difference map differs from O-SCD"
    print("parity OK")


if __name__ == "__main__":
    main()
