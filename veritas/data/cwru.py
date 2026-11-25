"""CWRU bearing data (domain: industrial), 12 kHz drive-end fault set, 10 classes.

Case Western Reserve University Bearing Data Center: a 2 hp motor with
accelerometers on the drive-end (DE) and fan-end (FE) bearing housings. Single
point faults were seeded by EDM into the drive-end bearing (SKF 6205) at the
inner race (IR), a ball (B) or the outer race at 6 o'clock (OR@6, the load
zone), with diameters 0.007, 0.014 and 0.021 in, and the motor ran at 0-3 hp
load (about 1797-1730 rpm).

Files used (one per class x load; ``https://engineering.case.edu/sites/default/files/<id>.mat``)::

    load hp   normal  IR007 B007 OR007  IR014 B014 OR014  IR021 B021 OR021
    0          97     105   118  130    169   185  197    209   222  234
    1          98     106   119  131    170   186  198    210   223  235
    2          99     107   120  132    171   187  199    211   224  236
    3         100     108   121  133    172   188  200    212   225  237

Preprocessing
- channels DE and FE; BA (base accelerometer) is skipped because the normal
  baseline files do not have it.
- the normal baseline files are recorded at 48 kHz (Smith & Randall 2015) and
  are resampled to 12 kHz with an anti-aliased polyphase filter (factor 4).
- non-overlapping windows of 1,024 samples (85 ms, ~2.4 shaft revolutions);
  the tail of each record that does not fill a window is dropped.
- values are kept in the source units (acceleration, presumably g; the
  Data Center does not state the unit). No normalisation.

Splits
- ``standard``: every file is cut in time into a contiguous train part (first
  70 % of its windows) and test part (last 30 %). This is stratified by class
  and load by construction and avoids neighbouring windows of one record landing
  on both sides; train and test still come from the same recording, so it is an
  optimistic, within-run split (as most CWRU papers use).
- ``shift``: train on loads 0, 1, 2 hp, test on 3 hp (new speed/load).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

URL = "https://engineering.case.edu/sites/default/files/{fid}.mat"
FS = 12_000
T = 1024
TRAIN_FRAC = 0.7
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# class name -> file ids for loads 0, 1, 2, 3 hp
CLASSES: dict[str, list[int]] = {
    "normal (healthy bearing)": [97, 98, 99, 100],
    "inner race fault, 0.007 in": [105, 106, 107, 108],
    "inner race fault, 0.014 in": [169, 170, 171, 172],
    "inner race fault, 0.021 in": [209, 210, 211, 212],
    "ball fault, 0.007 in": [118, 119, 120, 121],
    "ball fault, 0.014 in": [185, 186, 187, 188],
    "ball fault, 0.021 in": [222, 223, 224, 225],
    "outer race fault at 6 o'clock (load zone), 0.007 in": [130, 131, 132, 133],
    "outer race fault at 6 o'clock (load zone), 0.014 in": [197, 198, 199, 200],
    "outer race fault at 6 o'clock (load zone), 0.021 in": [234, 235, 236, 237],
}
NORMAL_IDS = {97, 98, 99, 100}  # 48 kHz recordings


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (veritas-ts data loader)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def _read(path: Path, fid: int) -> np.ndarray:
    """(2, L) array [DE, FE] at 12 kHz. Picks the variables of *this* file:
    some .mat files also carry another record (99.mat holds X098_* too)."""
    from scipy.io import loadmat
    from scipy.signal import resample_poly

    d = loadmat(path)
    tag = f"X{fid:03d}"
    x = np.stack([d[f"{tag}_DE_time"].ravel(), d[f"{tag}_FE_time"].ravel()]).astype(np.float64)
    if fid in NORMAL_IDS:
        x = resample_poly(x, up=1, down=4, axis=-1)
    return x


def load_cwru() -> TSDataset:
    raw = RAW_DIR / "cwru"
    Xs, ys, loads, fids, starts, is_test = [], [], [], [], [], []
    for c, (name, ids) in enumerate(CLASSES.items()):
        for load, fid in enumerate(ids):
            path = raw / f"{fid}.mat"
            _download(URL.format(fid=fid), path)
            x = _read(path, fid)
            nw = x.shape[1] // T
            w = x[:, :nw * T].reshape(2, nw, T).transpose(1, 0, 2)
            n_tr = int(round(TRAIN_FRAC * nw))
            Xs.append(w.astype(np.float32))
            ys.append(np.full(nw, c))
            loads.append(np.full(nw, load))
            fids.append(np.full(nw, fid))
            starts.append(np.arange(nw) * T)
            is_test.append(np.arange(nw) >= n_tr)
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.int64)
    load_hp = np.concatenate(loads).astype(np.int64)
    file_id = np.concatenate(fids).astype(np.int64)
    window_start = np.concatenate(starts).astype(np.int64)
    is_test = np.concatenate(is_test)

    std_tr, std_te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    sh_tr, sh_te = np.flatnonzero(load_hp != 3), np.flatnonzero(load_hp == 3)

    meta = DatasetMeta(
        name="cwru",
        domain="industrial",
        fs=float(FS),
        channel_names=[
            "drive-end accelerometer (DE bearing housing)",
            "fan-end accelerometer (FE bearing housing)",
        ],
        class_names=list(CLASSES),
        channel_groups={"DE": [0], "FE": [1]},
        channel_edges=[(0, 1)],  # both sensors on the same motor shaft / housing
        group_fields=["load_hp", "file_id", "window_start"],
        splits=["standard", "shift"],
        description=(
            "CWRU bearing vibration, 2 hp motor, faults seeded in the drive-end "
            "bearing (SKF 6205). 12 kHz DE+FE accelerometers, windows of 1,024 "
            "samples (85 ms). 10 classes: normal, inner race / ball / outer race "
            "(6 o'clock) faults of 0.007, 0.014 and 0.021 in, motor loads 0-3 hp "
            f"(nominal shaft speed {NOMINAL_RPM} rpm)."
        ),
        source="https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
        license="Free for research use per the CWRU Bearing Data Center; cite the "
                "Case Western Reserve University Bearing Data Center (and Smith & Randall 2015).",
        notes=(
            "12k drive-end files (ids in module docstring); normal baseline files 97-100 "
            "are 48 kHz and were resampled to 12 kHz (resample_poly down=4). Channels DE, FE "
            "(BA skipped: absent in normal files). Non-overlapping 1,024-sample windows; "
            "record tails dropped; values in source units (acceleration, presumably g), "
            "no normalisation. window_start is the sample offset at 12 kHz within the file. "
            "standard split: first 70% of each file's windows -> train, last 30% -> test "
            "(contiguous, so stratified by class and load; same recordings on both sides, "
            "i.e. within-run). shift split: train loads 0-2 hp, test load 3 hp. "
            "File 99.mat also contains record 98; only X099_* is used. Caveat: CWRU fault "
            "classes are separable from very few cues and each class x load is one "
            "recording, so accuracies are optimistic (Smith & Randall 2015). Sanity baseline "
            "(random forest on band powers + summary stats): accuracy 1.000 on both standard "
            "and shift, i.e. the load shift barely changes the signatures."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"load_hp": load_hp, "file_id": file_id, "window_start": window_start},
        split_idx={"standard": {"train": std_tr, "test": std_te},
                   "shift": {"train": sh_tr, "test": sh_te}},
    )


@register("cwru")
def build_cwru() -> TSDataset:
    return load_cwru()
