#!/usr/bin/env python
"""Convert XFeat's network (+ O-SCD's descriptor normalisation and heatmap) to Core ML.

Output Models/XFeat_<W>x<H>.mlpackage. The Swift side (XFeatPostprocess) does NMS, scoring,
top-k and descriptor sampling exactly as O-SCD's Detector does.
"""
import argparse
import os

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


def unfold2d_reshape(x, ws=2):
    """XFeatModel._unfold2d without Tensor.unfold (no Core ML converter): the same space-to-depth
    with channel index c * ws² + i * ws + j, written as reshape + permute."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // ws, ws, W // ws, ws).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * ws * ws, H // ws, W // ws)


class XFeatCore(torch.nn.Module):
    def __init__(self, xf):
        super().__init__()
        self.net = xf.net
        probe = torch.rand(1, 1, 64, 96)
        assert torch.equal(self.net._unfold2d(probe, ws=8), unfold2d_reshape(probe, ws=8)), "space-to-depth order differs"
        self.net._unfold2d = unfold2d_reshape

    def forward(self, x):
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        scores = F.softmax(K1, 1)[:, :64]                      # XFeat.get_kpts_heatmap, softmax_temp = 1
        B, _, H, W = scores.shape
        heat = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heat = heat.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return M1, heat, H1


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--xfeat-repo", default=os.path.join(here, "..", "third_party", "accelerated_features"))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--out", default=os.path.join(here, "..", "Models"))
    ap.add_argument("--fixtures", default=os.path.join(here, "..", "fixtures"))
    a = ap.parse_args()

    xf = torch.hub.load(a.xfeat_repo, "XFeat", source="local", pretrained=True, top_k=6144).eval()
    core = XFeatCore(xf).eval()
    example = torch.rand(1, 3, a.height, a.width)
    with torch.no_grad():
        ref_heat = xf.get_kpts_heatmap(xf.net(example)[1])
        assert torch.allclose(core(example)[1], ref_heat, atol=1e-6), "heatmap reshape differs from XFeat"
        traced = torch.jit.trace(core, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, a.height, a.width), dtype=np.float32)],
        outputs=[ct.TensorType(name="feats", dtype=np.float32),
                 ct.TensorType(name="heatmap", dtype=np.float32),
                 ct.TensorType(name="reliability", dtype=np.float32)],
        # FP32: every FP16 build tested on the M4 produced wrong outputs on some compute unit
        # (CPU-only FP16 breaks descriptors; mixed precision breaks everywhere). XFeat is small.
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"XFeat_{a.width}x{a.height}.mlpackage")
    mlmodel.save(path)
    print("saved", path)

    # Parity against the Sol fixture (fp16 CUDA network outputs), when present.
    img_path = os.path.join(a.fixtures, "xfeat", "image.npy")
    if not os.path.exists(img_path):
        print("no fixtures; skipped parity check")
        return
    img = torch.from_numpy(np.load(img_path)).permute(2, 0, 1)[None]
    x = F.interpolate(img, (a.height, a.width), mode="bilinear", align_corners=False)
    pred = mlmodel.predict({"image": x.numpy().astype(np.float32)})
    # O-SCD runs XFeat in fp16 on CUDA; this model is fp32, so compare robust statistics.
    for name, fx in [("feats", "net_feats"), ("heatmap", "net_heatmap"), ("reliability", "net_reliability")]:
        got = pred[name].reshape(-1)
        want = np.load(os.path.join(a.fixtures, "xfeat", fx + ".npy")).reshape(-1)
        err = np.abs(got - want)
        p99 = np.percentile(err, 99)
        print(f"{name}: |coreml - oscd| p99 = {p99:.4g}, max = {err.max():.4g}")
        assert p99 < {"feats": 1e-2, "heatmap": 1e-2, "reliability": 1e-2}[name], name
        if name == "heatmap":
            agree = np.mean((got > 0.05) == (want > 0.05))
            print(f"heatmap above-threshold agreement = {agree:.5f}")
            assert agree >= 0.995, "detection-threshold agreement"
    print("parity OK")


if __name__ == "__main__":
    main()
