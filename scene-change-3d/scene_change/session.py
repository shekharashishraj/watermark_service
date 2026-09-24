"""A recorded walkthrough: RGB-D keyframes with intrinsics and gravity-aligned poses.

On-disk format (one folder per session)::

    session.json   name, width, height, K (3x3), poses (N x 4x4, camera-to-world,
                   OpenCV camera axes, z-up world), timestamps, meta
    frames.npz     rgb   (N,H,W,3) uint8
                   depth (N,H,W)   uint16 millimetres, 0 = invalid

Real captures plug in by writing this format. For an ARKit capture, convert each
``ARFrame.camera.transform`` from ARKit camera axes (x right, y up, z back) to
OpenCV axes by right-multiplying with diag(1,-1,-1,1), and rotate ARKit's y-up
world to z-up. LiDAR ``sceneDepth`` (or depth rendered from a reconstruction)
supplies the depth channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Session:
    name: str
    K: np.ndarray                 # (3,3)
    rgb: np.ndarray               # (N,H,W,3) uint8
    depth: np.ndarray             # (N,H,W) float32 metres, 0 = invalid
    poses: np.ndarray             # (N,4,4) camera-to-session-world
    timestamps: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        self.poses = np.asarray(self.poses, dtype=np.float64)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.timestamps is None:
            self.timestamps = np.arange(len(self.poses), dtype=np.float64) * 0.5
        n = len(self.poses)
        assert self.rgb.shape[0] == n and self.depth.shape[0] == n, "frame count mismatch"
        assert self.rgb.shape[1:3] == self.depth.shape[1:3], "rgb/depth size mismatch"

    @property
    def height(self) -> int:
        return int(self.depth.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth.shape[2])

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def subset(self, idx) -> "Session":
        idx = np.asarray(idx)
        return Session(self.name, self.K.copy(), self.rgb[idx], self.depth[idx], self.poses[idx],
                       self.timestamps[idx], dict(self.meta))

    def with_poses(self, poses: np.ndarray) -> "Session":
        return Session(self.name, self.K, self.rgb, self.depth, poses, self.timestamps, dict(self.meta))

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        info = {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "poses": self.poses.reshape(len(self), 16).tolist(),
            "timestamps": np.asarray(self.timestamps).tolist(),
            "meta": self.meta,
        }
        (path / "session.json").write_text(json.dumps(info, indent=1))
        depth_mm = np.clip(np.round(self.depth * 1000.0), 0, 65535).astype(np.uint16)
        np.savez_compressed(path / "frames.npz", rgb=self.rgb.astype(np.uint8), depth=depth_mm)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        path = Path(path)
        info = json.loads((path / "session.json").read_text())
        data = np.load(path / "frames.npz")
        poses = np.asarray(info["poses"], dtype=np.float64).reshape(-1, 4, 4)
        return cls(
            name=info.get("name", path.name),
            K=np.asarray(info["K"]),
            rgb=data["rgb"],
            depth=data["depth"].astype(np.float32) / 1000.0,
            poses=poses,
            timestamps=np.asarray(info.get("timestamps", np.arange(len(poses)) * 0.5)),
            meta=info.get("meta", {}),
        )
