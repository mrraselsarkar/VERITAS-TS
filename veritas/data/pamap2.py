"""PAMAP2 Physical Activity Monitoring (domain: activity).

9 subjects (101-109) wore three Colibri IMUs (dominant wrist "hand", chest,
dominant-side ankle; 100 Hz) and a chest heart-rate monitor (~9 Hz) while
following a protocol of 12 activities. We use the Protocol recordings only.

Channels (19): for each IMU the +-16 g accelerometer (m/s^2, x/y/z) and the
gyroscope (rad/s, x/y/z), plus heart rate (bpm) linearly interpolated to 100 Hz.
The +-6 g accelerometer (saturates during running/jumping), magnetometer,
temperature and the documented-invalid orientation columns are not used.

Windows: non-overlapping 2 s windows (200 samples) cut from runs of one protocol
activity (transient label 0 and windows that straddle an activity change are
dropped). Short dropouts are linearly interpolated; windows touching longer
gaps are dropped (see ``MAX_GAP_IMU`` / ``MAX_GAP_HR``).

- ``standard``: stratified random split over windows (leaks: windows of one
  subject and one activity bout land on both sides).
- ``shift``:    leave-subjects-out, ``SHIFT_TEST_SUBJECTS`` as test.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register, stratified_split, group_split

URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
ZIP_NAME = "pamap2_physical_activity_monitoring.zip"
INNER_ZIP = "PAMAP2_Dataset.zip"
SUBJECTS = list(range(101, 110))

FS = 100.0
T = 200  # 2 s
MAX_GAP_IMU = 10   # samples (0.1 s): longer IMU dropouts invalidate the window
MAX_GAP_HR = 300   # samples (3 s) without a heart-rate reading
# 105 and 108 performed all 12 activities (incl. rope jumping and running), so every class
# appears in test while the train subjects keep most rope-jumping windows. 109 recorded only
# ~1 min of rope jumping (31 windows); it is kept, always on the train side.
SHIFT_TEST_SUBJECTS = (105, 108)

# protocol activity id -> readable name (class index = position)
ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling",
    7: "Nordic walking", 12: "ascending stairs", 13: "descending stairs",
    16: "vacuum cleaning", 17: "ironing", 24: "rope jumping",
}

# column layout of subjectXXX.dat (0-based): 0 time [s], 1 activity id, 2 heart rate,
# then 17 columns per IMU starting at 3 (hand), 20 (chest), 37 (ankle):
# +0 temperature, +1..3 acc 16 g, +4..6 acc 6 g, +7..9 gyro, +10..12 mag, +13..16 orientation
IMUS = {"hand": 3, "chest": 20, "ankle": 37}
AXES = ["x", "y", "z"]
CHANNELS, COLUMNS = [], []
for _imu, _off in IMUS.items():
    for _kind, _k in (("acc", 1), ("gyro", 7)):
        for _j, _a in enumerate(AXES):
            CHANNELS.append(f"{_imu}_{_kind}_{_a}")
            COLUMNS.append(_off + _k + _j)
CHANNELS.append("heart_rate")
COLUMNS.append(2)


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
    """Download the UCI archive and extract the Protocol recordings; returns their folder."""
    raw = RAW_DIR / "pamap2"
    proto = raw / "PAMAP2_Dataset" / "Protocol"
    if all((proto / f"subject{s}.dat").exists() for s in SUBJECTS):
        return proto
    outer = _download(URL, raw / ZIP_NAME)
    # the inner zip is stored uncompressed in the outer one, so it can be read in place
    with zipfile.ZipFile(outer) as zo, zipfile.ZipFile(zo.open(INNER_ZIP)) as zi:
        for m in zi.namelist():
            is_doc = m.startswith("PAMAP2_Dataset/") and m.lower().endswith((".pdf", ".txt"))
            if (m.startswith("PAMAP2_Dataset/Protocol/") and m.endswith(".dat")) or is_doc:
                target = raw / m
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zi.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
    return proto


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap that have valid samples on both sides."""
    x = x.copy()
    nan = np.isnan(x)
    if not nan.any():
        return x
    d = np.diff(np.r_[0, nan.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # run = [start, end)
    for s, e in zip(starts, ends):
        if s == 0 or e == len(x) or e - s > max_gap:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def _windows_one_subject(path: Path) -> tuple[list[np.ndarray], list[int], dict]:
    import pandas as pd

    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1] + COLUMNS, dtype=np.float64,
                     engine="c")
    t = df[0].to_numpy()
    act = df[1].to_numpy().astype(np.int64)
    sig = df[COLUMNS].to_numpy()  # (N, 19), channel order = CHANNELS

    # contiguous-time blocks (no missing timestamps); interpolation never crosses a block border
    step = np.diff(t)
    breaks = np.flatnonzero(np.abs(step - 1 / FS) > 0.5 / FS) + 1
    bounds = np.r_[0, breaks, len(t)]
    clean = np.empty_like(sig)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        for c in range(sig.shape[1]):
            gap = MAX_GAP_HR if CHANNELS[c] == "heart_rate" else MAX_GAP_IMU
            clean[b0:b1, c] = _fill_short_gaps(sig[b0:b1, c], gap)

    # runs of constant protocol activity inside one time block
    block_id = np.zeros(len(t), np.int64)
    block_id[breaks] = 1
    block_id = np.cumsum(block_id)
    change = np.flatnonzero((np.diff(act) != 0) | (np.diff(block_id) != 0)) + 1
    run_b = np.r_[0, change, len(t)]
    wins, labels = [], []
    stats = {"n_samples": len(t), "time_breaks": len(breaks), "dropped_nan": 0}
    for r0, r1 in zip(run_b[:-1], run_b[1:]):
        a = act[r0]
        if a not in ACTIVITIES:
            continue
        for w0 in range(r0, r1 - T + 1, T):
            w = clean[w0:w0 + T]
            if not np.isfinite(w).all():
                stats["dropped_nan"] += 1
                continue
            wins.append(w.T.astype(np.float32))
            labels.append(a)
    return wins, labels, stats


def _edges() -> list[tuple[int, int]]:
    idx = {c: i for i, c in enumerate(CHANNELS)}
    edges = []
    for imu in IMUS:
        for kind in ("acc", "gyro"):  # x-y-z triangle within each sensor
            ch = [idx[f"{imu}_{kind}_{a}"] for a in AXES]
            edges += [(ch[0], ch[1]), (ch[0], ch[2]), (ch[1], ch[2])]
        for a in AXES:  # accelerometer-gyroscope of the same IMU, same axis
            edges.append((idx[f"{imu}_acc_{a}"], idx[f"{imu}_gyro_{a}"]))
    for a in AXES:  # body chain hand - chest - ankle (same-axis accelerometers)
        edges += [(idx[f"hand_acc_{a}"], idx[f"chest_acc_{a}"]), (idx[f"chest_acc_{a}"], idx[f"ankle_acc_{a}"])]
    for a in AXES:  # heart-rate strap sits on the chest
        edges.append((idx["heart_rate"], idx[f"chest_acc_{a}"]))
    return edges


@register("pamap2")
def build_pamap2() -> TSDataset:
    proto = _fetch_raw()
    X_list, act_list, subj_list, info = [], [], [], {}
    for s in SUBJECTS:
        wins, labels, stats = _windows_one_subject(proto / f"subject{s}.dat")
        info[s] = stats | {"windows": len(wins)}
        X_list += wins
        act_list += labels
        subj_list += [s] * len(wins)
    X = np.stack(X_list).astype(np.float32)
    act_ids = sorted(ACTIVITIES)
    lookup = {a: i for i, a in enumerate(act_ids)}
    y = np.array([lookup[a] for a in act_list], dtype=np.int64)
    subject = np.array(subj_list, dtype=np.int64)

    std_tr, std_te = stratified_split(y, test_frac=0.25, seed=0)
    sh_tr, sh_te = group_split(subject, SHIFT_TEST_SUBJECTS)
    missing = set(range(len(act_ids))) - set(np.unique(y[sh_te]))

    per_subject = {s: int(v["windows"]) for s, v in info.items()}
    dropped = sum(v["dropped_nan"] for v in info.values())
    meta = DatasetMeta(
        name="pamap2",
        domain="activity",
        fs=FS,
        channel_names=CHANNELS,
        class_names=[ACTIVITIES[a] for a in act_ids],
        channel_groups={
            f"{imu}_{kind}": [CHANNELS.index(f"{imu}_{kind}_{a}") for a in AXES]
            for imu in IMUS for kind in ("acc", "gyro")
        } | {"heart_rate": [CHANNELS.index("heart_rate")]},
        channel_edges=_edges(),
        group_fields=["subject"],
        splits=["standard", "shift"],
        description=(
            "Three body-worn IMUs (wrist, chest, ankle; 3-axis accelerometer + gyroscope each) and a "
            "chest heart-rate monitor, 9 subjects, 2 s windows at 100 Hz; 12 protocol activities of "
            "daily living, household chores and sports."
        ),
        source="https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring",
        license="CC BY 4.0 (cite Reiss & Stricker, ISWC 2012)",
        notes=(
            "Protocol files only. Units: acceleration m/s^2 (+-16 g sensor), angular rate rad/s, heart "
            "rate bpm; no normalisation. Heart rate (~9 Hz) linearly interpolated to 100 Hz; IMU "
            f"dropouts up to {MAX_GAP_IMU} samples and heart-rate gaps up to {MAX_GAP_HR} samples "
            f"interpolated, windows with longer gaps dropped ({dropped} dropped). Non-overlapping "
            f"{T}-sample windows starting at the onset of each activity run; label 0 (transient) and "
            "mixed windows dropped. Class ids in source: "
            + ", ".join(f"{ACTIVITIES[a]}={a}" for a in act_ids)
            + f". Windows per subject: {per_subject}. 'standard' = stratified random split (seed 0, "
            "25% test); windows of the same subject and activity bout appear on both sides, so it "
            "leaks and overestimates accuracy. 'shift' = leave-subjects-out, test subjects "
            f"{list(SHIFT_TEST_SUBJECTS)}"
            + (f"; classes missing from test: {sorted(missing)}" if missing else "; all 12 classes in test")
            + ". Subject 109 has only ~1.4 min of protocol data (rope jumping only) and stays in "
            "train. Not every subject did every activity: 103 has no running, cycling, Nordic walking "
            "or rope jumping; 104 no running or rope jumping; 107 no rope jumping; 106 one rope-jumping "
            "window. "
            "The +-16 g accelerometer still clips near +-157 m/s^2 (mostly ankle; also hand/chest in "
            "running and rope jumping). "
            "Sensor axes are each IMU's own frame (not aligned across IMUs), so cross-IMU edges "
            "are a body-chain prior rather than a physical axis correspondence."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta, groups={"subject": subject},
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": sh_tr, "test": sh_te},
        },
    )
