"""Unified dataset format for VERITAS-TS.

Every dataset is converted once into ``data/processed/<name>.npz`` (arrays) plus
``<name>.json`` (metadata). The npz holds *all* series of the dataset together
with index arrays for each named split, so the same file serves the standard
split and the distribution-shift split.

npz layout
----------
X                  float32 (n, C, T)   all series, fixed length
y                  int64   (n,)        class index into meta.class_names
g_<field>          (n,)                one array per meta.group_fields entry
                                       (subject, device, condition, ...)
<split>_train      int64               row indices, one pair per meta.splits entry
<split>_test       int64

Experiments call :func:`load_dataset`, which applies a split, a stratified test
subsample (LLM agents are expensive) and an optional length cap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DOMAINS = ("healthcare", "activity", "industrial", "energy", "synthetic")


@dataclass
class DatasetMeta:
    name: str
    domain: str
    fs: float | None
    channel_names: list[str]
    class_names: list[str]
    # Evidence-grid channel groups: group name -> channel indices. Every channel
    # must belong to exactly one group. Univariate data has a single group.
    channel_groups: dict[str, list[int]]
    # Prior channel graph for the graph-aware diffusion model (undirected pairs).
    channel_edges: list[tuple[int, int]] = field(default_factory=list)
    group_fields: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    description: str = ""
    source: str = ""
    license: str = ""
    notes: str = ""


@dataclass
class TSDataset:
    X: np.ndarray
    y: np.ndarray
    meta: DatasetMeta
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    split_idx: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.X.shape)


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: DatasetMeta
    split: str
    groups_train: dict[str, np.ndarray]
    groups_test: dict[str, np.ndarray]
    test_rows: np.ndarray  # row indices into the full dataset, for traceability


# --------------------------------------------------------------------------- registry

_BUILDERS: dict[str, Callable[[], TSDataset]] = {}


def register(name: str):
    """Register a builder that downloads raw data (if needed) and returns a TSDataset."""

    def deco(fn: Callable[[], TSDataset]):
        if name in _BUILDERS:
            raise ValueError(f"dataset {name!r} registered twice")
        _BUILDERS[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(_BUILDERS)


def build(name: str, overwrite: bool = False) -> Path:
    """Build and save one dataset; returns the npz path."""
    out = PROCESSED_DIR / f"{name}.npz"
    if out.exists() and not overwrite:
        return out
    if name not in _BUILDERS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    ds = _BUILDERS[name]()
    validate(ds)
    save(ds)
    return out


# --------------------------------------------------------------------------- io

def save(ds: TSDataset) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {"X": ds.X.astype(np.float32), "y": ds.y.astype(np.int64)}
    for k, v in ds.groups.items():
        arrays[f"g_{k}"] = np.asarray(v)
    for s, parts in ds.split_idx.items():
        for part, idx in parts.items():
            arrays[f"{s}_{part}"] = np.asarray(idx, dtype=np.int64)
    np.savez_compressed(PROCESSED_DIR / f"{ds.meta.name}.npz", **arrays)
    meta = asdict(ds.meta)
    meta["channel_edges"] = [list(e) for e in ds.meta.channel_edges]
    meta["n"], meta["C"], meta["T"] = (int(d) for d in ds.X.shape)
    meta["class_counts"] = np.bincount(ds.y, minlength=len(ds.meta.class_names)).tolist()
    (PROCESSED_DIR / f"{ds.meta.name}.json").write_text(json.dumps(meta, indent=2))


def load_processed(name: str) -> TSDataset:
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/prepare_data.py {name}")
    meta_d = json.loads((PROCESSED_DIR / f"{name}.json").read_text())
    for k in ("n", "C", "T", "class_counts"):
        meta_d.pop(k, None)
    meta_d["channel_edges"] = [tuple(e) for e in meta_d["channel_edges"]]
    meta = DatasetMeta(**meta_d)
    with np.load(path, allow_pickle=False) as z:
        X, y = z["X"], z["y"]
        groups = {f: z[f"g_{f}"] for f in meta.group_fields}
        split_idx = {s: {p: z[f"{s}_{p}"] for p in ("train", "test")} for s in meta.splits}
    return TSDataset(X=X, y=y, meta=meta, groups=groups, split_idx=split_idx)


def load_dataset(
    name: str,
    split: str = "standard",
    max_test: int | None = 2000,
    max_len: int | None = None,
    seed: int = 0,
    max_train: int | None = None,
) -> SplitData:
    """Load one split, with stratified test (and optionally train) subsamples
    and an optional length cap."""
    ds = load_processed(name)
    if split not in ds.split_idx:
        raise KeyError(f"{name} has splits {list(ds.split_idx)}, not {split!r}")
    tr, te = ds.split_idx[split]["train"], ds.split_idx[split]["test"]
    if max_test is not None and len(te) > max_test:
        te = stratified_subsample(te, ds.y[te], max_test, seed)
    if max_train is not None and len(tr) > max_train:
        tr = stratified_subsample(tr, ds.y[tr], max_train, seed)
    X = ds.X
    if max_len is not None and X.shape[-1] > max_len:
        X = downsample_to(X, max_len)
    return SplitData(
        X_train=X[tr], y_train=ds.y[tr], X_test=X[te], y_test=ds.y[te],
        meta=ds.meta, split=split,
        groups_train={k: v[tr] for k, v in ds.groups.items()},
        groups_test={k: v[te] for k, v in ds.groups.items()},
        test_rows=te,
    )


# --------------------------------------------------------------------------- helpers

def stratified_subsample(idx: np.ndarray, labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Pick k of idx, keeping class proportions (every present class keeps >= 1)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(counts / counts.sum() * k).astype(int))
    while quota.sum() > k:
        quota[np.argmax(quota)] -= 1
    short = k - quota.sum()
    order = np.argsort(-(counts / counts.sum() * k - quota))
    for i in order[:short]:
        quota[i] = min(quota[i] + 1, counts[i])
    picked = [rng.choice(idx[labels == c], size=q, replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(picked))


def downsample_to(X: np.ndarray, max_len: int) -> np.ndarray:
    """Anti-aliased integer-factor downsampling so that T <= max_len."""
    from scipy.signal import resample_poly

    factor = int(np.ceil(X.shape[-1] / max_len))
    return resample_poly(X, up=1, down=factor, axis=-1).astype(np.float32)


def fixed_length(x: np.ndarray, length: int, mode: str = "center") -> np.ndarray:
    """Crop or zero-pad the last axis to `length` (center crop by default)."""
    T = x.shape[-1]
    if T >= length:
        start = (T - length) // 2 if mode == "center" else 0
        return x[..., start:start + length]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, length - T)]
    return np.pad(x, pad)


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)


def group_split(groups: np.ndarray, test_groups) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose group is in test_groups go to test; the rest to train."""
    is_test = np.isin(groups, np.asarray(list(test_groups)))
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def validate(ds: TSDataset) -> None:
    X, y, m = ds.X, ds.y, ds.meta
    n, C, T = X.shape
    assert X.ndim == 3, "X must be (n, C, T)"
    assert m.domain in DOMAINS, f"domain {m.domain!r} not in {DOMAINS}"
    assert np.isfinite(X).all(), "X contains NaN/inf"
    assert y.shape == (n,) and y.min() >= 0 and y.max() < len(m.class_names)
    assert len(m.channel_names) == C, "channel_names length != C"
    covered = sorted(i for g in m.channel_groups.values() for i in g)
    assert covered == list(range(C)), "channel_groups must partition the channels"
    assert all(0 <= a < C and 0 <= b < C and a != b for a, b in m.channel_edges)
    assert set(ds.groups) == set(m.group_fields)
    assert all(len(v) == n for v in ds.groups.values())
    assert list(ds.split_idx) == m.splits and "standard" in m.splits
    for s, parts in ds.split_idx.items():
        tr, te = parts["train"], parts["test"]
        assert len(tr) and len(te), f"split {s} empty"
        assert not np.intersect1d(tr, te).size, f"split {s}: train/test overlap"
        assert tr.max() < n and te.max() < n
        missing = set(np.unique(y[te])) - set(np.unique(y[tr]))
        assert not missing, f"split {s}: test classes {missing} absent from train"
