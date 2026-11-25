"""HHAR: Heterogeneity Human Activity Recognition, smartphone part (domain: activity).

9 users carried 8 smartphones of 4 models (LG Nexus 4, Samsung Galaxy S3,
Galaxy S3 mini, Galaxy S+; two units each) in a waist pouch while biking,
sitting, standing, walking and taking stairs up/down. Each phone logged its
accelerometer (m/s^2) and gyroscope (rad/s) at the highest rate the device
allowed (irregularly spaced; acc ~200 Hz Nexus 4, ~100 Hz S3 / S3 mini,
~50 Hz S+; gyro ~200 Hz Nexus 4 / S3, ~100 Hz S3 mini). We use the
"Activity recognition exp" phone files; smartwatch and "Still exp" data are not
used.

The Galaxy S+ ("samsungold") phones have NO gyroscope rows in the release, so
this 6-channel (acc + gyro) version contains 3 models / 6 phones; the S+ data
are dropped rather than zero-filled. s3mini_2 contributes only a few minutes per
user, and s3_1 has no data for user f.

Preprocessing per (user, device):
1. Sort each stream by its sensor timestamp (``Creation_Time``, ns), drop
   duplicate timestamps. Accelerometer and gyroscope of one phone share that
   clock, which is what makes the merge possible.
2. Split each stream where consecutive samples are more than ``MAX_GAP_S``
   apart; intersect the acc and gyro intervals.
3. Inside each common interval, linearly interpolate both streams onto a
   200 Hz grid, low-pass (4th-order Butterworth, 20 Hz, zero phase) against
   aliasing and keep every 4th sample -> common 50 Hz grid.
4. Label each grid point with the activity of the nearest accelerometer sample;
   drop points where the nearest gyroscope sample carries another label (an
   alignment check) and "null" points.
5. Cut non-overlapping 128-sample windows (2.56 s) from runs of one label.

- ``standard``: stratified random split over windows (leaks: same user, device
  and activity bout on both sides).
- ``shift``:    cross-device-model split, ``SHIFT_TEST_MODELS`` held out.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, group_split, register, stratified_split

URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
ZIP_NAME = "heterogeneity_activity_recognition.zip"
INNER_ZIP = "Activity recognition exp.zip"
INNER_DIR = "Activity recognition exp"
FILES = {"acc": "Phones_accelerometer.csv", "gyro": "Phones_gyroscope.csv"}

USERS = list("abcdefghi")
DEVICES = ["nexus4_1", "nexus4_2", "s3_1", "s3_2", "s3mini_1", "s3mini_2", "samsungold_1", "samsungold_2"]
MODEL_NAMES = {
    "nexus4": "LG Nexus 4", "s3": "Samsung Galaxy S3", "s3mini": "Samsung Galaxy S3 mini",
    "samsungold": "Samsung Galaxy S+",
}
# source gt label -> readable class name (class index = position); "null" is dropped
LABELS = {"bike": "biking", "sit": "sitting", "stand": "standing", "walk": "walking",
          "stairsup": "walking upstairs", "stairsdown": "walking downstairs"}
GT_CODES = list(LABELS)

FS = 50.0
FS_HI = 200.0      # intermediate grid for anti-aliased resampling
T = 128            # 2.56 s
MAX_GAP_S = 0.1    # larger gaps between consecutive samples split a stream
LOWPASS_HZ = 20.0
SHIFT_TEST_MODELS = ("s3mini",)  # held-out model; train = Nexus 4 + Galaxy S3
CHUNK_ROWS = 2_000_000

AXES = ["x", "y", "z"]
CHANNELS = [f"acc_{a}" for a in AXES] + [f"gyro_{a}" for a in AXES]


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
    """Download the UCI archive and extract the two phone CSVs; returns their folder."""
    raw = RAW_DIR / "hhar"
    folder = raw / INNER_DIR
    if all((folder / f).exists() for f in list(FILES.values()) + ["readme.txt"]):
        return folder
    outer = _download(URL, raw / ZIP_NAME)
    # the inner zip is stored uncompressed in the outer one, so it can be read in place
    with zipfile.ZipFile(outer) as zo, zipfile.ZipFile(zo.open(INNER_ZIP)) as zi:
        for f in list(FILES.values()) + ["readme.txt"]:
            target = folder / f
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with zi.open(f"{INNER_DIR}/{f}") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
    return folder


def _codes(values, vocab: list[str]) -> np.ndarray:
    """Map strings to positions in vocab; anything else (incl. NaN/'null') -> -1."""
    import pandas as pd

    cat = pd.Categorical(values, categories=vocab)
    return np.asarray(cat.codes, dtype=np.int8)


def _read_stream(path: Path) -> dict[str, np.ndarray]:
    """Read one phone CSV in chunks into compact arrays."""
    import pandas as pd

    cols = ["Creation_Time", "x", "y", "z", "User", "Device", "gt"]
    dtypes = {"Creation_Time": np.int64, "x": np.float32, "y": np.float32, "z": np.float32,
              "User": str, "Device": str, "gt": str}
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("t", "v", "user", "device", "gt")}
    for ch in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=CHUNK_ROWS):
        parts["t"].append(ch["Creation_Time"].to_numpy())
        parts["v"].append(ch[["x", "y", "z"]].to_numpy(np.float32))
        parts["user"].append(_codes(ch["User"], USERS))
        parts["device"].append(_codes(ch["Device"], DEVICES))
        parts["gt"].append(_codes(ch["gt"], GT_CODES))
    return {k: np.concatenate(v) for k, v in parts.items()}


def _intervals(t: np.ndarray) -> np.ndarray:
    """[start, end] times of maximal runs with consecutive spacing <= MAX_GAP_S."""
    br = np.flatnonzero(np.diff(t) > MAX_GAP_S)
    starts = np.r_[t[0], t[br + 1]]
    ends = np.r_[t[br], t[-1]]
    return np.stack([starts, ends], axis=1)


def _intersect(a: np.ndarray, b: np.ndarray) -> list[tuple[float, float]]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append((lo, hi))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return out


def _interp(t: np.ndarray, v: np.ndarray, lab: np.ndarray, grid: np.ndarray):
    """Linear interpolation of v (N, 3) at grid times; label of the nearest sample."""
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    w = ((grid - t[i]) / (t[i + 1] - t[i]))[:, None]
    out = v[i] * (1 - w) + v[i + 1] * w
    nearest = np.where(w[:, 0] < 0.5, lab[i], lab[i + 1])
    return out, nearest


def _stream(d: dict, mask: np.ndarray, t0: int):
    t_ns, order = np.unique(d["t"][mask], return_index=True)  # sorted, duplicates dropped
    return (t_ns - t0) / 1e9, d["v"][mask][order].astype(np.float64), d["gt"][mask][order]


def _windows_one(acc: dict, gyro: dict, ma: np.ndarray, mg: np.ndarray, sos) -> tuple[list, list, dict]:
    from scipy.signal import sosfiltfilt

    t0 = min(acc["t"][ma].min(), gyro["t"][mg].min())
    ta, va, la = _stream(acc, ma, t0)
    tg, vg, lg = _stream(gyro, mg, t0)
    step, up = 1 / FS, int(FS_HI / FS)
    wins, labels = [], []
    st = {"common_s": 0.0, "label_mismatch": 0, "grid_points": 0}
    for lo, hi in _intersect(_intervals(ta), _intervals(tg)):
        n50 = int(np.floor((hi - lo) / step)) + 1
        if n50 < T:
            continue
        st["common_s"] += hi - lo
        grid = lo + np.arange((n50 - 1) * up + 1) / FS_HI
        xa, ya = _interp(ta, va, la, grid)
        xg, yg = _interp(tg, vg, lg, grid)
        x = sosfiltfilt(sos, np.concatenate([xa, xg], axis=1), axis=0)[::up]
        ya, yg = ya[::up], yg[::up]
        st["grid_points"] += len(ya)
        st["label_mismatch"] += int((ya != yg).sum())
        lab = np.where(ya == yg, ya, -1)
        change = np.flatnonzero(np.diff(lab) != 0) + 1
        bounds = np.r_[0, change, len(lab)]
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            if lab[r0] < 0:
                continue
            for w0 in range(r0, r1 - T + 1, T):
                wins.append(x[w0:w0 + T].T.astype(np.float32))
                labels.append(int(lab[r0]))
    return wins, labels, st


def _edges() -> list[tuple[int, int]]:
    edges = []
    for off in (0, 3):  # x-y-z triangle within each sensor
        edges += [(off, off + 1), (off, off + 2), (off + 1, off + 2)]
    edges += [(a, a + 3) for a in range(3)]  # same-axis acc-gyro (one phone frame)
    return edges


@register("hhar")
def build_hhar() -> TSDataset:
    from scipy.signal import butter

    folder = _fetch_raw()
    acc = _read_stream(folder / FILES["acc"])
    gyro = _read_stream(folder / FILES["gyro"])
    sos = butter(4, LOWPASS_HZ, btype="low", fs=FS_HI, output="sos")

    X_list, y_list, g_user, g_dev, stats, no_gyro = [], [], [], [], {}, set()
    for u in range(len(USERS)):
        acc_u, gyro_u = acc["user"] == u, gyro["user"] == u
        for dv in range(len(DEVICES)):
            ma, mg = acc_u & (acc["device"] == dv), gyro_u & (gyro["device"] == dv)
            if ma.sum() >= 2 and mg.sum() < 2:
                no_gyro.add(DEVICES[dv])
            if ma.sum() < 2 or mg.sum() < 2:
                continue
            wins, labels, st = _windows_one(acc, gyro, ma, mg, sos)
            stats[(USERS[u], DEVICES[dv])] = st | {"windows": len(wins)}
            X_list += wins
            y_list += labels
            g_user += [USERS[u]] * len(wins)
            g_dev += [DEVICES[dv]] * len(wins)
    del acc, gyro

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    user = np.array(g_user)
    device = np.array(g_dev)
    model = np.array([d.rsplit("_", 1)[0] for d in g_dev])

    std_tr, std_te = stratified_split(y, test_frac=0.25, seed=0)
    sh_tr, sh_te = group_split(model, SHIFT_TEST_MODELS)

    mismatch = sum(s["label_mismatch"] for s in stats.values()) / max(1, sum(s["grid_points"] for s in stats.values()))
    used = [m for m in MODEL_NAMES if (model == m).any()]
    per_model = {m: int((model == m).sum()) for m in used}
    per_device = {d: int((device == d).sum()) for d in DEVICES if (device == d).any()}
    assert set(SHIFT_TEST_MODELS) <= set(used)
    meta = DatasetMeta(
        name="hhar",
        domain="activity",
        fs=FS,
        channel_names=CHANNELS,
        class_names=list(LABELS.values()),
        channel_groups={"acc": [0, 1, 2], "gyro": [3, 4, 5]},
        channel_edges=_edges(),
        group_fields=["user", "device_model", "device"],
        splits=["standard", "shift"],
        description=(
            f"Waist-carried smartphones ({len(used)} models, {len(per_device)} phones), 9 users; 3-axis "
            "accelerometer and gyroscope resampled to 50 Hz, 2.56 s windows; 6 activities (biking, "
            "sitting, standing, walking, stairs up, stairs down)."
        ),
        source="https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition",
        license="CC BY 4.0 (cite Stisen et al., SenSys 2015)",
        notes=(
            "Phone accelerometer (m/s^2, includes gravity) and gyroscope (rad/s), no normalisation. "
            "Acc and gyro merged per (user, device) on the shared sensor clock (Creation_Time): "
            f"streams split at gaps > {MAX_GAP_S} s, intersected, linearly interpolated to {FS_HI:.0f} Hz, "
            f"{LOWPASS_HZ:.0f} Hz zero-phase Butterworth low-pass, decimated to {FS:.0f} Hz. Label = "
            "nearest accelerometer sample; points where the nearest gyroscope sample has a different "
            f"label ({100 * mismatch:.3f}% of grid points) and 'null' points dropped. Alignment check "
            "(done once, offline): in walking windows the cross-correlation of |d|acc|/dt| and |gyro| "
            "peaks at lag 0 to -1 sample (<= 20 ms) on every phone. Non-overlapping "
            f"{T}-sample windows from runs of one label. Phones without gyroscope data, dropped: "
            f"{sorted(no_gyro)}. Windows per model: {per_model}; per phone: {per_device}. Native "
            "rates differ by model (acc ~200 Hz Nexus 4, ~100 Hz S3 and S3 mini; gyro ~200 Hz Nexus 4 "
            "and S3, ~100 Hz S3 mini; s3mini_2 ~70-90 Hz and only minutes per user). 'standard' = "
            "stratified random split (seed 0, 25% test); same user/device/bout on both sides, so it "
            "leaks and overestimates accuracy. 'shift' = cross-device-model: train on "
            f"{[m for m in used if m not in SHIFT_TEST_MODELS]}, test on {list(SHIFT_TEST_MODELS)} "
            "(users overlap; devices and models do not). "
            "Model codes: " + ", ".join(f"{k}={v}" for k, v in MODEL_NAMES.items()) + "."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"user": user, "device_model": model, "device": device},
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": sh_tr, "test": sh_te},
        },
    )
