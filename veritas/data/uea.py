"""UEA multivariate archive datasets with a clear sensor structure (domain: activity / gesture).

- ``natops``: NATOPS aircraft-handling gestures (Ghouaiel et al.). 24 channels = x/y/z
  positions of 8 body points (hand tip, elbow, wrist and thumb, left and right), 51 steps,
  6 classes. Channel order verified from the data: consecutive triplets are one point, and
  the four left-side points (and the four right-side points) are strongly correlated.
- ``cricket``: Cricket umpire signals (Ko et al.). 6 channels = x/y/z acceleration of the
  left and right wrist, 1197 steps at 184 Hz, downsampled by 4 to 300 steps, 12 classes.
  Left/right order verified from the data (x correlated, y mirrored between wrists).

``standard`` is the archive's default train/test split.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .base import RAW_DIR, DatasetMeta, TSDataset, register

AXES = ["x", "y", "z"]


def _load(name: str):
    from aeon.datasets import load_classification
    path = RAW_DIR / "uea"
    Xtr, ytr = load_classification(name, split="train", extract_path=str(path))
    Xte, yte = load_classification(name, split="test", extract_path=str(path))
    labels = np.concatenate([ytr, yte])
    classes = sorted(np.unique(labels), key=lambda s: float(s))
    y = np.array([classes.index(v) for v in labels], dtype=np.int64)
    X = np.concatenate([Xtr, Xte]).astype(np.float32)
    n_tr = len(ytr)
    return X, y, np.arange(n_tr), np.arange(n_tr, len(y))


def _triplets(points: list[str]):
    channels = [f"{p}_{a}" for p in points for a in AXES]
    groups = {p: [3 * i, 3 * i + 1, 3 * i + 2] for i, p in enumerate(points)}
    edges = []
    for i in range(len(points)):  # x-y-z triangle within each point
        edges += [(3 * i, 3 * i + 1), (3 * i, 3 * i + 2), (3 * i + 1, 3 * i + 2)]
    return channels, groups, edges


@register("natops")
def build_natops() -> TSDataset:
    X, y, tr, te = _load("NATOPS")
    points = ["hand_left", "hand_right", "elbow_left", "elbow_right",
              "wrist_left", "wrist_right", "thumb_left", "thumb_right"]
    channels, groups, edges = _triplets(points)
    for side in ("left", "right"):  # same coordinate across the points of one arm
        idx = [points.index(f"{p}_{side}") for p in ("hand", "elbow", "wrist", "thumb")]
        for a in range(3):
            for i in idx:
                for j in idx:
                    if i < j:
                        edges.append((3 * i + a, 3 * j + a))
    meta = DatasetMeta(
        name="natops", domain="activity", fs=None, channel_names=channels,
        class_names=[f"gesture {c}" for c in "ABCDEF"],  # archive labels 1-6; neutral names
        channel_groups=groups, channel_edges=edges, splits=["standard"],
        description="NATOPS aircraft-handling gestures: x/y/z of hand tips, elbows, wrists and thumbs; 6 classes.",
        source="https://timeseriesclassification.com/description.php?Dataset=NATOPS",
        license="UEA archive terms (cite Bagnall et al. 2018; Ghouaiel et al. 2017)",
        notes="UEA default split (180/180). Edges: x-y-z within each point; same coordinate across the "
              "four points of each arm.",
    )
    return TSDataset(X=X, y=y, meta=meta, split_idx={"standard": {"train": tr, "test": te}})


@register("cricket")
def build_cricket() -> TSDataset:
    X, y, tr, te = _load("Cricket")
    X = resample_poly(X, up=1, down=4, axis=-1).astype(np.float32)  # 1197 -> 300 steps
    points = ["wrist_left", "wrist_right"]
    channels, groups, edges = _triplets(points)
    edges += [(a, 3 + a) for a in range(3)]  # same axis on both wrists
    meta = DatasetMeta(
        name="cricket", domain="activity", fs=184 / 4, channel_names=channels,
        class_names=[f"signal {c}" for c in "ABCDEFGHIJKL"],  # archive labels 1-12; neutral names
        channel_groups=groups, channel_edges=edges, splits=["standard"],
        description="Cricket umpire signals: x/y/z acceleration of both wrists; 12 classes.",
        source="https://timeseriesclassification.com/description.php?Dataset=Cricket",
        license="UEA archive terms (cite Bagnall et al. 2018; Ko et al. 2005)",
        notes="UEA default split (108/72); downsampled by 4 with an anti-aliasing filter. Classes are named by "
              "the archive's numeric labels (1-12 -> A-L) because the label-to-signal mapping is not verified.",
    )
    return TSDataset(X=X, y=y, meta=meta, split_idx={"standard": {"train": tr, "test": te}})
