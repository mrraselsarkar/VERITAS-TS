"""Paderborn University (KAt) bearing data, preprocessed "FD" set (domain: industrial).

The FD windows (TS-TCC / TF-C / AdaTime / InstructTime) come in two public releases:
- TF-C (Zhang et al. 2022, figshare, CC BY 4.0): FD-A and FD-B, *all* windows
  of two operating conditions (13,640 each);
- AdaTime (Ragab et al. 2023, DR-NTU doi:10.21979/N9/PU85XN, folder ``FD``,
  CC BY-NC 4.0): all four conditions (domains 0-3), each a random ~20 % subset
  of the same windows (2,729 per domain; AdaTime domain 0 / 1 windows are exact,
  label-identical subsets of FD-A / FD-B, verified).
We use TF-C's FD-A / FD-B for conditions 0 / 1 and AdaTime's domains 2 / 3 for
the other two conditions (TF-C does not publish them, and DR-NTU could not
deliver AdaTime's 75 MB train_0 / train_1 files: it throttles concurrent
connections, ignores Range requests and closes connections after ~30 min).
FD-A / FD-B are reduced to a seeded class-stratified 20 % subset so that all four
conditions have AdaTime's size and the same sparse-window structure.

Test rig (Lessmeier et al. 2016): PMSM drive motor, torque shaft, test module
with a 6203 deep-groove ball bearing, flywheel, load motor; accelerometer on the
test-module housing sampled at 64 kHz; 20 recordings of 4 s per bearing and
operating condition. Each FD sample is a 5,120-sample window (80 ms) of the
``vibration_1`` signal, cut at a stride of 4,096 samples (62 per recording).

Verified here against the raw KAt archives (exact sample matching of windows to
``<condition>_<bearing>_<file>.mat``) and bearing physics:
- condition index (= AdaTime domain index; NOT the order of Lessmeier's table):
  0 = N09_M07_F10 (900 rpm, 0.7 Nm, 1000 N) = TF-C FD-A,
  1 = N15_M01_F10 (1500 rpm, 0.1 Nm, 1000 N) = TF-C FD-B,
  2 = N15_M07_F04 (1500 rpm, 0.7 Nm, 400 N),
  3 = N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N).
- labels and bearings (all windows of AdaTime domains 2 and 3 located in raw files):
  0 = healthy: the single bearing K001;
  1 = OUTER race damage: KA04, KA15, KA16, KA22, KA30 (real damage from
      accelerated lifetime tests); envelope spectra peak at the outer-race
      defect frequency BPFO = 3.05 x shaft rate (76 Hz at 1500 rpm);
  2 = INNER race damage: KI14, KI16, KI17, KI18, KI21 (real damage); envelope
      peak at BPFI = 4.95 x shaft rate (123 Hz) with shaft-rate sidebands.
  AdaTime names the classes Healthy/D1/D2; TF-C's prose order "undamaged,
  inner, outer" does NOT match the labels.

Preprocessing
- exact duplicate windows are dropped (FD-A re-uses one KA04 recording).
- neighbouring windows of one recording overlap by 1,024 samples (stride 4,096
  < length 5,120). Overlapping windows are linked into ``segment_id`` groups
  (exact sample matching) so that splits never put two overlapping windows on
  different sides. AdaTime's own split does; ``source_part`` keeps the source's
  train/val/test assignment for reference only.
- downsampled 5,120 -> 1,024 samples (x5, polyphase anti-aliasing filter),
  i.e. 12.8 kHz, 80 ms per window; values in source units, no normalisation.

Splits
- ``standard``: stratified by condition x class, 70/30, whole segments to one
  side (seed 0). Windows of the same 4 s recording / same bearing still occur
  on both sides (as in TF-C/AdaTime), and the healthy class is a single
  bearing, so this split is optimistic.
- ``shift``: train on the three 1500 rpm conditions (1, 2, 3), test on the
  900 rpm condition (0 = FD-A). A speed change scales every fault frequency by
  0.6. It is also the only hold-out that is a real shift: a random forest on
  band powers trained on three conditions scored 0.72 on 900 rpm vs 0.95-1.00
  on each 1500 rpm hold-out (AdaTime test files).

If DR-NTU downloads keep failing, fetch train_2.pt, test_2.pt, train_3.pt and
test_3.pt manually from https://doi.org/10.21979/N9/PU85XN (folder FD) into
``data/raw/paderborn/``.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

FIGSHARE = "https://ndownloader.figshare.com/files/{fid}"
DRNTU = "https://researchdata.ntu.edu.sg/api/access/datafile/{fid}"
# condition -> list of (local file, url, size in bytes, source part)
SOURCES = {
    0: [(f"tfc_FD-A_{p}.pt", FIGSHARE.format(fid=fid), size, f"tfc_{p}") for p, fid, size in (
        ("train", 35469257, 335283203), ("val", 35469260, 111761795), ("test", 35469251, 111761795))],
    1: [(f"tfc_FD-B_{p}.pt", FIGSHARE.format(fid=fid), size, f"tfc_{p}") for p, fid, size in (
        ("train", 35454872, 2459139), ("val", 35454875, 861379), ("test", 35454869, 555486211))],
    2: [("train_2.pt", DRNTU.format(fid=108810), 74890471, "adatime_train"),
        ("test_2.pt", DRNTU.format(fid=108805), 36913127, "adatime_test")],
    3: [("train_3.pt", DRNTU.format(fid=108807), 74890471, "adatime_train"),
        ("test_3.pt", DRNTU.format(fid=108812), 36913127, "adatime_test")],
}
TFC_FRAC = 0.2         # AdaTime keeps ~20 % of the windows of each condition
FS_RAW, DOWN, L_RAW = 64_000, 5, 5120
STRIDE = 4096          # stride of the source windows inside a recording
TEST_FRAC = 0.3
SHIFT_TEST_CONDITION = 0  # 900 rpm
# condition index -> (code, rpm, load torque Nm, radial force N); verified, see docstring
CONDITIONS = {
    0: ("N09_M07_F10", 900, 0.7, 1000),
    1: ("N15_M01_F10", 1500, 0.1, 1000),
    2: ("N15_M07_F04", 1500, 0.7, 400),
    3: ("N15_M07_F10", 1500, 0.7, 1000),
}
CLASS_NAMES = [
    "healthy bearing (no damage)",
    "outer race damage (real damage from accelerated lifetime test)",
    "inner race damage (real damage from accelerated lifetime test)",
]


def _download(url: str, dest: Path, size: int, attempts: int = 8) -> None:
    """Whole-file, size-checked retries (DR-NTU drops long connections and
    ignores Range requests). Files are fetched one at a time on purpose."""
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (veritas-ts data loader)"})
    for k in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size == size:
                tmp.replace(dest)
                return
        except OSError:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not download {url} -> {dest} ({attempts} attempts)")


def _load_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=True)
    X = d["samples"].numpy().reshape(len(d["labels"]), -1).astype(np.float64)
    return X, d["labels"].numpy().astype(np.int64)


def _key(a: np.ndarray) -> np.ndarray:
    """64-bit hash of every length-4 run along the last axis (exact float bits)."""
    K = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5],
                 dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(np.ascontiguousarray(a).view(np.uint64), 4, axis=-1)
    with np.errstate(over="ignore"):
        return (w * K).sum(-1)


def _segments(X: np.ndarray) -> np.ndarray:
    """Component id per window: windows i, j are linked when j's tail equals i's
    head, i.e. j starts STRIDE samples before i in the same recording."""
    n, L = X.shape
    head = _key(X[:, :4])[:, 0]
    at_stride = _key(X[:, STRIDE:STRIDE + 4])[:, 0]
    order = np.argsort(head)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    lo = np.searchsorted(head[order], at_stride, "left")
    hi = np.searchsorted(head[order], at_stride, "right")
    for j in np.flatnonzero(hi > lo):
        for i in order[lo[j]:hi[j]]:
            if i != j and np.array_equal(X[j, STRIDE:], X[i, :L - STRIDE]):
                parent[find(i)] = find(j)
    return np.array([find(a) for a in range(n)])


def _group_stratified_split(strata: np.ndarray, groups: np.ndarray, frac: float, seed: int):
    """Per stratum, move whole groups to test until >= frac of its rows."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        g, cnt = np.unique(groups[rows], return_counts=True)
        perm = rng.permutation(len(g))
        take = g[perm][np.cumsum(cnt[perm]) <= np.ceil(frac * len(rows))]
        is_test[rows[np.isin(groups[rows], take)]] = True
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def load_paderborn() -> TSDataset:
    from scipy.signal import resample_poly

    raw = RAW_DIR / "paderborn"
    for files in SOURCES.values():
        for fname, url, size, _ in files:
            _download(url, raw / fname, size)

    rng = np.random.default_rng(0)
    Xs, ys, conds, parts, segs = [], [], [], [], []
    n_dup, n_src = {}, {}
    seg_offset = 0
    for c, files in SOURCES.items():
        loaded = [(*_load_pt(raw / fname), part) for fname, _, _, part in files]
        X = np.concatenate([a for a, _, _ in loaded])
        y = np.concatenate([b for _, b, _ in loaded])
        part = np.concatenate([np.full(len(b), p) for _, b, p in loaded])
        assert X.shape[1] == L_RAW
        _, first = np.unique(X, axis=0, return_index=True)   # drop exact duplicates
        keep = np.sort(first)
        n_dup[c] = len(X) - len(keep)
        n_src[c] = len(keep)
        if part[0].startswith("tfc"):                         # AdaTime-sized subset
            keep = np.sort(np.concatenate([
                rng.choice(keep[y[keep] == k], int(round(TFC_FRAC * (y[keep] == k).sum())), replace=False)
                for k in np.unique(y[keep])]))
        X, y, part = X[keep], y[keep], part[keep]
        _, seg = np.unique(_segments(X), return_inverse=True)
        Xs.append(resample_poly(X, up=1, down=DOWN, axis=-1).astype(np.float32))
        ys.append(y)
        conds.append(np.full(len(y), c))
        parts.append(part)
        segs.append(seg + seg_offset)
        seg_offset += seg.max() + 1
    X = np.concatenate(Xs)[:, None, :]
    y = np.concatenate(ys)
    cond = np.concatenate(conds).astype(np.int64)
    part = np.concatenate(parts)
    segment_id = np.concatenate(segs).astype(np.int64)
    condition = np.array([CONDITIONS[c][0] for c in cond])
    n_linked = int(len(y) - len(np.unique(segment_id)))

    std_tr, std_te = _group_stratified_split(cond * 10 + y, segment_id, TEST_FRAC, seed=0)
    sh_tr = np.flatnonzero(cond != SHIFT_TEST_CONDITION)
    sh_te = np.flatnonzero(cond == SHIFT_TEST_CONDITION)

    cond_txt = "; ".join(f"{i}={c} ({r} rpm, {m} Nm, {f} N)" for i, (c, r, m, f) in CONDITIONS.items())
    meta = DatasetMeta(
        name="paderborn",
        domain="industrial",
        fs=FS_RAW / DOWN,
        channel_names=["vibration (accelerometer on bearing test-module housing)"],
        class_names=CLASS_NAMES,
        channel_groups={"vibration": [0]},
        channel_edges=[],
        group_fields=["condition", "condition_id", "segment_id", "source_part"],
        splits=["standard", "shift"],
        description=(
            "Paderborn University (KAt) bearing test rig, 6203 ball bearing driven by an "
            "electric motor; housing vibration, 80 ms windows (1,024 samples at 12.8 kHz). "
            "3 classes: healthy, outer race damage, inner race damage; 4 operating conditions "
            "varying shaft speed (900/1500 rpm), load torque (0.1/0.7 Nm) and radial force "
            "(400/1000 N). Preprocessed 'FD' windows as used by TS-TCC, TF-C, AdaTime and "
            "InstructTime."
        ),
        source="TF-C FD-A/FD-B: https://figshare.com/articles/dataset/19930205 and /19930226; "
               "AdaTime FD domains 2-3: https://doi.org/10.21979/N9/PU85XN; raw data: "
               "https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter",
        license="CC BY-NC 4.0 (AdaTime part; TF-C part CC BY 4.0); cite Lessmeier et al. "
                "(2016), Zhang et al. (2022), Ragab et al. (2023)",
        notes=(
            f"Conditions {cond_txt}; condition_id equals the AdaTime domain index (mapping "
            "verified by matching windows to raw KAt files; differs from Lessmeier's table "
            "order). Conditions 0/1 from TF-C FD-A/FD-B (all windows; seeded class-stratified "
            f"{TFC_FRAC:.0%} subset kept to match AdaTime's per-domain size), conditions 2/3 = "
            "AdaTime domains 2/3 (AdaTime train_0/train_1 could not be downloaded from DR-NTU; "
            "AdaTime domains 0/1 are verified subsets of FD-A/FD-B). Windows per condition "
            f"after de-duplication, before subsetting: {n_src}; exact duplicates dropped: "
            f"{n_dup}. Labels and bearings verified by exact matching to raw KAt files: "
            "0 healthy = K001 only (healthy vs damaged is confounded with bearing identity), "
            "1 outer race = KA04/KA15/KA16/KA22/KA30, 2 inner race = KI14/KI16/KI17/KI18/KI21 "
            "(all real damage); envelope spectra peak at BPFO / BPFI respectively. TF-C's "
            "prose order (inner before outer) is wrong for the labels. Source windows 5,120 "
            "samples at 64 kHz, stride 4,096 within 4 s recordings; downsampled x5 "
            "(resample_poly) to 1,024 at 12.8 kHz, so content above 6.4 kHz is removed. "
            f"Overlapping neighbours (1,024 raw samples shared) linked into segment_id "
            f"({n_linked} links); kept, but never split across train/test. source_part = the "
            "source's own train/val/test assignment (AdaTime's splits overlapping windows; do "
            "not use it as a split). standard: 70/30 stratified by condition x class, grouped "
            "by segment, seed 0 (same recordings/bearings on both sides: optimistic). shift: "
            "train conditions 1-3 (1500 rpm), test condition 0 (900 rpm); a quick RF "
            "leave-one-condition-out baseline gives 0.72 for 900 rpm vs 0.95-1.00 for the "
            "1500 rpm hold-outs. Values in source units (raw KAt vibration_1; unit not stated "
            "in the source files), no normalisation."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"condition": condition, "condition_id": cond, "segment_id": segment_id,
                "source_part": part},
        split_idx={"standard": {"train": std_tr, "test": std_te},
                   "shift": {"train": sh_tr, "test": sh_te}},
    )


@register("paderborn")
def build_paderborn() -> TSDataset:
    return load_paderborn()
