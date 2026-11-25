"""UCI HAR: Human Activity Recognition Using Smartphones (domain: activity).

30 volunteers (19-48 y) wore a Samsung Galaxy S II on the waist and performed six
activities. We use the raw "Inertial Signals" shipped with the dataset: 9 channels
(body_acc xyz, body_gyro xyz, total_acc xyz), 128 samples per window at 50 Hz
(2.56 s). The provider already median-filtered the signals, applied a 20 Hz
low-pass, and split total acceleration into gravity + body motion (0.3 Hz
Butterworth); body_acc = total_acc - gravity. Units: g for acceleration, rad/s
for the gyroscope.

- ``standard``: stratified random split of all 10,299 windows, with the same test
  fraction as the original split. The provider's windows overlap by 50%, so
  neighbouring windows of one recording share half their samples and can land on
  both sides of this split: accuracy on it is optimistic (leakage).
- ``shift``:    the original subject-disjoint split (21 train / 9 test subjects).
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register, stratified_split

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_NAME = "human_activity_recognition_using_smartphones.zip"
INNER_ZIP = "UCI HAR Dataset.zip"
ROOT_DIR = "UCI HAR Dataset"

SIGNALS = ["body_acc", "body_gyro", "total_acc"]
AXES = ["x", "y", "z"]
CHANNELS = [f"{s}_{a}" for s in SIGNALS for a in AXES]
# activity_labels.txt ids 1..6 -> readable names (class index = id - 1)
CLASS_NAMES = ["walking", "walking upstairs", "walking downstairs", "sitting", "standing", "lying down"]
FS = 50.0
T = 128


def _download(url: str, dest: Path) -> Path:
    """Download url to dest unless it already exists (atomic via a .part file)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    return dest


def _fetch_raw() -> Path:
    """Download the UCI archive and extract the needed files; returns the dataset root."""
    raw = RAW_DIR / "ucihar"
    outer = _download(URL, raw / ZIP_NAME)
    root = raw / ROOT_DIR
    needed = [f"{ROOT_DIR}/activity_labels.txt", f"{ROOT_DIR}/README.txt"]
    for part in ("train", "test"):
        needed += [f"{ROOT_DIR}/{part}/y_{part}.txt", f"{ROOT_DIR}/{part}/subject_{part}.txt"]
        needed += [f"{ROOT_DIR}/{part}/Inertial Signals/{c}_{part}.txt" for c in CHANNELS]
    if all((raw / m).exists() for m in needed):
        return root
    with zipfile.ZipFile(outer) as zo, zipfile.ZipFile(io.BytesIO(zo.read(INNER_ZIP))) as zi:
        for m in needed:
            target = raw / m
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zi.read(m))
    return root


def _load_part(root: Path, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = root / part / "Inertial Signals"
    X = np.stack([np.loadtxt(sig / f"{c}_{part}.txt", dtype=np.float64) for c in CHANNELS], axis=1)
    y = np.loadtxt(root / part / f"y_{part}.txt", dtype=np.int64) - 1
    subj = np.loadtxt(root / part / f"subject_{part}.txt", dtype=np.int64)
    assert X.shape[1:] == (len(CHANNELS), T) and len(y) == len(subj) == len(X)
    return X.astype(np.float32), y, subj


def _edges() -> list[tuple[int, int]]:
    idx = {c: i for i, c in enumerate(CHANNELS)}
    edges = []
    for s in SIGNALS:  # x-y-z triangle within each signal
        edges += [(idx[f"{s}_x"], idx[f"{s}_y"]), (idx[f"{s}_x"], idx[f"{s}_z"]), (idx[f"{s}_y"], idx[f"{s}_z"])]
    for a in AXES:  # same axis across signals (one phone, one body frame)
        edges += [(idx[f"body_acc_{a}"], idx[f"total_acc_{a}"]), (idx[f"body_acc_{a}"], idx[f"body_gyro_{a}"])]
    return edges


@register("ucihar")
def build_ucihar() -> TSDataset:
    root = _fetch_raw()
    Xtr, ytr, str_ = _load_part(root, "train")
    Xte, yte, ste = _load_part(root, "test")
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    subject = np.concatenate([str_, ste])

    n_tr = len(ytr)
    orig_tr, orig_te = np.arange(n_tr), np.arange(n_tr, len(y))
    assert not np.intersect1d(subject[orig_tr], subject[orig_te]).size
    std_tr, std_te = stratified_split(y, test_frac=len(yte) / len(y), seed=0)

    test_subjects = sorted(int(s) for s in np.unique(ste))
    meta = DatasetMeta(
        name="ucihar",
        domain="activity",
        fs=FS,
        channel_names=CHANNELS,
        class_names=CLASS_NAMES,
        channel_groups={s: [CHANNELS.index(f"{s}_{a}") for a in AXES] for s in SIGNALS},
        channel_edges=_edges(),
        group_fields=["subject"],
        splits=["standard", "shift"],
        description=(
            "Waist-worn smartphone (Samsung Galaxy S II) inertial signals from 30 subjects; "
            "9 channels (body acceleration, gyroscope, total acceleration; x/y/z), "
            "2.56 s windows at 50 Hz; 6 activities of daily living."
        ),
        source="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        license="CC BY 4.0 (cite Anguita et al., ESANN 2013)",
        notes=(
            "Raw 'Inertial Signals' as provided (no resampling, no normalisation). Units: body_acc and "
            "total_acc in g (total includes gravity; body = total minus 0.3 Hz low-pass gravity), "
            "body_gyro in rad/s. Provider preprocessing: median filter + 20 Hz 3rd-order Butterworth "
            "low-pass. Windows are the provider's fixed-width 2.56 s windows with 50% OVERLAP, kept as "
            "provided, so consecutive windows share 64 samples. 'standard' = stratified random split "
            f"(seed 0, test fraction {len(yte) / len(y):.3f}); overlapping windows and same-subject "
            "windows fall on both sides, so this split leaks and overestimates accuracy. 'shift' = "
            f"original subject-disjoint split, test subjects {test_subjects}. Class 'lying down' is "
            "LAYING in the source labels. Edges: x-y-z within each signal; same-axis body_acc-total_acc "
            "and body_acc-body_gyro."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta, groups={"subject": subject},
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": orig_tr, "test": orig_te},
        },
    )
