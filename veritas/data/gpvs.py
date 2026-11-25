"""GPVS-Faults: fault diagnosis in a grid-connected PV system (domain: energy).

Source: Bakdi, Guichi, Mekhilef, Bounoua (2020), "GPVS-Faults: Experimental Data
for fault scenarios in grid-connected PV systems under MPPT and IPPT modes",
Mendeley Data v1, doi:10.17632/n76t439f65.1, CC BY 4.0. Experiments and fault
scenarios: Bakdi, Bounoua, Guichi, Mekhilef (2021), Int. J. Electr. Power Energy
Syst. 125, 106457, doi:10.1016/j.ijepes.2020.106457.

Raw data (downloaded automatically, public, no login; ~121 MB in data/raw/gpvs/)
------------------------------------------------------------------------------
``CSV_Files.zip`` holds 16 CSVs ``F<x><y>.csv``: x = condition 0..7, y = operating
mode (M = MPPT, maximum power; L = IPPT, limited / intermediate power), one
lab experiment each, 7-14.4 s long. ``ReadMe.pdf`` is the dataset description.
The 16 ``.mat`` files on the same page hold the same tables (MATLAB v7.3 table
objects, awkward to read outside MATLAB) and are not downloaded. If the automatic
download ever fails, fetch "CSV_Files.zip" by hand from
https://data.mendeley.com/datasets/n76t439f65/1 into data/raw/gpvs/.

Columns: Time, Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc, Vf. The
ReadMe states an average sampling time of 9.9989 us; the Time column actually
advances by 99.99 us (fs ~= 10 kHz, as also used by later papers), with jitter
below half a sample, so rows are treated as uniformly sampled at 10 kHz.

Conditions (Bakdi et al. 2021, Table of injected faults)
--------------------------------------------------------
F0 fault-free; F1 inverter fault: complete failure in one of the six IGBTs;
F2 feedback sensor fault: one phase (current) sensor fault, 20%; F3 grid anomaly:
intermittent voltage sags; F4 PV array mismatch: 10-20% nonhomogeneous partial
shading; F5 PV array mismatch: 15% open circuit in the PV array; F6 MPPT/IPPT
controller fault: -20% gain of the PI controller of the boost converter; F7 boost
converter controller fault: +20% time constant of that PI controller.

Preprocessing
-------------
* Channels kept: the 9 measured signals Ipv [A], Vpv [V] (PV array), Vdc [V]
  (DC link), ia/ib/ic [A] (inverter output currents), va/vb/vc [V] (grid phase
  voltages). Dropped: Iabc, If, Vabc, Vf, the positive-sequence magnitude and
  frequency estimates of the PMU/PLL block (computed from the phase signals and
  updated only every ~3 samples). Note: vc is not ADC-quantised like va/vb
  (it is apparently computed in the acquisition system) but it is not an exact
  function of va, vb (vc + va + vb has a ~3 V std), so it is kept.
* Anti-aliased decimation 10 kHz -> 2.5 kHz (resample_poly, factor 4) on the
  whole recording, then non-overlapping 0.2 s windows (T = 500 samples = 10 grid
  cycles of 50 Hz, 50 samples per cycle; Nyquist 1.25 kHz keeps the 25th harmonic).
  Windows that short are needed because some post-fault segments are very short
  (F3M has 0.26 s of data after its fault). The MPPT/IPPT perturbation cycle of
  ~1.06 s is therefore not covered by one window. Physical units, float32, no
  normalisation.

Fault onsets (not documented beyond "introduced manually halfway"; detected here)
-------------------------------------------------------------------------------
Only windows after the onset carry the fault label; windows before
``onset - guard`` are healthy (label F0, segment "pre_fault"), windows in the
guard band are dropped. Onsets were read off the raw 10 kHz signals:
F1M 8.464 s  DC link and PV voltage collapse, grid currents jump to ~6 A peak
F1L 8.650 s  DC offsets appear in ib/ic, 50 Hz oscillation on Vdc
F2M 8.708 s  ib reading freezes at a constant (-0.21 A)
F2L 8.890 s  ic reading freezes at a constant (-0.35 A)
F3M 6.735 s  inverter trips (grid currents -> 0, Vdc rises to ~250 V); the
F3L 8.915 s  measured grid voltages show no visible sag, so the trip is the first
             observable effect; guard 0.5 s
F5M 7.388 s  PV current collapses to 0 (PV array disconnected), Vpv decays
F5L 7.147 s  same
F6M 11.58 s  DC-link voltage step +2.3 V; F6L 11.18 s step +1.5 V
F7M 11.18 s  DC-link voltage step -1.7 V; F7L 9.28 s step -1.0 V
             (F6/F7: least-squares fit of a step + linear drift + harmonics of the
             1.06 s perturbation cycle on one-cycle means of Vdc; F statistic
             2300-9400 vs <= 25 on F0/F4 recordings; guard 1.0 s because the
             parameter change may precede the visible step)
F4M, F4L     no change point in any channel (same fit and a per-feature step scan
             are at noise level), but both recordings sit at operating points that
             no other recording in the same mode has (F4M: Ipv ~1.83 A vs 2.2-2.4 A,
             i.e. ~20% less power; F4L: Vpv ~75 V vs 101-102 V). The shading was
             apparently in place for the whole recording, so all F4 windows are
             labelled F4.
Abrupt faults (F1, F2, F5) use a 0.2 s guard. After onset a "post_trip" segment
flag marks windows where the inverter has stopped (mean three-phase AC current rms
< 0.05 A; F1M after ~10.4 s, F3 entirely, F5M after ~13 s).

Balancing: at most CAP = 60 windows per (condition, mode); healthy windows are
spread evenly over their 7 source recordings per mode, and windows are thinned by
evenly spaced selection, which keeps time order.

Splits
------
* ``standard``: within each (recording, label) segment, the first 70% of windows
  in time go to train and the last 30% to test (a single-window segment goes to
  train). Leakage: train and test share recordings, so slow operating-point drift
  (irradiance, temperature) is shared and the windows on either side of a cut are
  adjacent in time. Because fault recordings also supply healthy windows, the
  recording identity alone does not give the label (except for F4).
  Late sub-phases can appear only in test (e.g. F5M's final inverter trip after
  ~13 s: its 7 post-trip windows are all in test).
* ``shift``: train on all MPPT windows, test on all IPPT windows (all 8 classes
  occur in both modes; F3 has a single MPPT window).
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

DATASET_URL = "https://data.mendeley.com/datasets/n76t439f65/1"
_FILE_URL = "https://data.mendeley.com/public-files/datasets/n76t439f65/files/{}/file_downloaded"
RAW_FILES = {  # name -> (Mendeley file id, sha256)
    "CSV_Files.zip": ("c4525461-eef2-4df4-9209-e4d79e06765d",
                      "88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63"),
    "ReadMe.pdf": ("ab22981b-c08b-4315-b219-0748b78425bd",
                   "d7325d6b90bfc9886c245082f6b6e6c66f984d1b6114e31413fa30788d5e5bde"),
}

CSV_COLUMNS = ["Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc",
               "Iabc", "If", "Vabc", "Vf"]
CHANNELS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc"]
CHANNEL_GROUPS = {"pv_dc_side": [0, 1, 2], "grid_currents": [3, 4, 5], "grid_voltages": [6, 7, 8]}
CHANNEL_EDGES = [
    (0, 1),                  # PV array current - voltage (I-V curve)
    (0, 2), (1, 2),          # PV array -> boost converter -> DC link (power balance)
    (2, 3), (2, 4), (2, 5),  # DC link -> inverter -> phase currents
    (3, 4), (4, 5), (3, 5),  # three-phase currents
    (6, 7), (7, 8), (6, 8),  # three-phase voltages
    (3, 6), (4, 7), (5, 8),  # current - voltage of the same phase
]
CLASS_NAMES = [
    "fault-free",
    "inverter fault (IGBT failure)",
    "feedback sensor fault (phase current sensor)",
    "grid anomaly (intermittent voltage sags)",
    "PV array mismatch (partial shading)",
    "PV array mismatch (open circuit)",
    "MPPT/IPPT controller fault (PI gain -20%)",
    "boost converter controller fault (PI time constant +20%)",
]
MODES = {"M": "MPPT", "L": "IPPT"}

FS_RAW = 10_000.0
DOWN = 4
FS = FS_RAW / DOWN          # 2500 Hz
WIN_S = 0.2
T = int(round(WIN_S * FS))  # 500 samples
EDGE = int(round(0.01 * FS))  # skip filter edge effects at both ends
TRAIN_FRAC = 0.7
CAP = 60                    # max windows per (condition, mode)
TRIP_IRMS = 0.05            # A; mean three-phase AC rms below this = inverter stopped

# experiment -> (fault onset [s on the file's Time axis], guard before onset [s]).
# onset None = fault-free recording; onset 0.0 = fault present for the whole file.
ONSETS: dict[str, tuple[float | None, float]] = {
    "F0M": (None, 0.0), "F0L": (None, 0.0),
    "F1M": (8.464, 0.2), "F1L": (8.650, 0.2),
    "F2M": (8.708, 0.2), "F2L": (8.890, 0.2),
    "F3M": (6.735, 0.5), "F3L": (8.915, 0.5),
    "F4M": (0.0, 0.0), "F4L": (0.0, 0.0),
    "F5M": (7.388, 0.2), "F5L": (7.147, 0.2),
    "F6M": (11.58, 1.0), "F6L": (11.18, 1.0),
    "F7M": (11.18, 1.0), "F7L": (9.28, 1.0),
}


# --------------------------------------------------------------------------- raw data

def _download(dest_dir) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, (file_id, sha) in RAW_FILES.items():
        path = dest_dir / name
        if path.exists():
            continue
        print(f"[gpvs] downloading {name} ...")
        req = urllib.request.Request(_FILE_URL.format(file_id),
                                     headers={"User-Agent": "veritas-ts data loader"})
        tmp = path.with_suffix(path.suffix + ".part")
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha:
            tmp.unlink()
            raise IOError(f"{name}: sha256 mismatch; download by hand from {DATASET_URL}")
        tmp.replace(path)


def _read_recordings(zip_path) -> dict[str, np.ndarray]:
    """experiment id -> (n, 14) float64 array with the CSV columns."""
    import pandas as pd

    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for exp in ONSETS:
            df = pd.read_csv(z.open(f"CSV_Files/{exp}.csv"))
            assert list(df.columns) == CSV_COLUMNS, f"{exp}: unexpected columns {list(df.columns)}"
            a = df.to_numpy(np.float64)
            assert np.isfinite(a).all(), f"{exp}: non-finite values"
            out[exp] = a
    return out


# --------------------------------------------------------------------------- windowing

def _windows(exp: str, a: np.ndarray, cls: int):
    """Cut one recording into labelled windows. Returns a list of dicts."""
    from scipy.signal import resample_poly

    t_raw, sig = a[:, 0], a[:, 1:1 + len(CHANNELS)]
    x = resample_poly(sig, up=1, down=DOWN, axis=0)          # (N, C) at 2.5 kHz
    t = t_raw[::DOWN][: len(x)]
    n = len(x)
    onset, guard = ONSETS[exp]

    def cut(starts, label, segment):
        rows = []
        for s in starts:
            w = x[s:s + T].T.astype(np.float32)                # (C, T)
            seg = segment
            if label != 0:
                ac = w[3:6] - w[3:6].mean(axis=1, keepdims=True)
                if np.sqrt((ac ** 2).mean(axis=1)).mean() < TRIP_IRMS:
                    seg = "post_trip"
            rows.append(dict(x=w, y=label, exp=exp, t0=float(t[s] - t[0]), segment=seg))
        return rows

    last = n - EDGE  # exclusive end for window ends
    if onset is None:                                           # fault-free run
        return cut(range(EDGE, last - T + 1, T), 0, "healthy_run")
    if onset == 0.0:                                            # fault present throughout
        return cut(range(EDGE, last - T + 1, T), cls, "fault")
    n_on = int(np.searchsorted(t, onset))
    rows = cut(range(n_on, last - T + 1, T), cls, "fault")
    n_end = n_on - int(round(guard * FS))                       # healthy windows end here
    pre = list(range(n_end - T, EDGE - 1, -T))[::-1]
    return cut(pre, 0, "pre_fault") + rows


def _evenly(n: int, k: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int)) if k < n else np.arange(n)


def _quota(counts: list[int], cap: int) -> list[int]:
    """Split `cap` over sources as evenly as their counts allow (water filling)."""
    q, left = [0] * len(counts), cap
    order = np.argsort(counts)
    for i, r in enumerate(order):
        q[r] = min(counts[r], left // (len(order) - i))
        left -= q[r]
    return q


def _balance(rows: list[dict]) -> list[dict]:
    keep = []
    for cls in range(len(CLASS_NAMES)):
        for m in MODES:
            sel = [r for r in rows if r["y"] == cls and r["exp"][-1] == m]
            exps = sorted({r["exp"] for r in sel})
            per = [[r for r in sel if r["exp"] == e] for e in exps]  # time-ordered
            for part, k in zip(per, _quota([len(p) for p in per], CAP)):
                keep += [part[i] for i in _evenly(len(part), k)]
    return keep


def _standard_split(exp: np.ndarray, y: np.ndarray, t0: np.ndarray):
    tr, te = [], []
    for e in np.unique(exp):
        for c in np.unique(y[exp == e]):
            idx = np.flatnonzero((exp == e) & (y == c))
            idx = idx[np.argsort(t0[idx])]
            k = len(idx)
            n_tr = 1 if k == 1 else min(max(1, int(round(TRAIN_FRAC * k))), k - 1)
            tr.append(idx[:n_tr])
            te.append(idx[n_tr:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


# --------------------------------------------------------------------------- builder

@register("gpvs")
def build_gpvs() -> TSDataset:
    raw = RAW_DIR / "gpvs"
    _download(raw)
    recs = _read_recordings(raw / "CSV_Files.zip")

    rows = []
    for exp, a in recs.items():
        rows += _windows(exp, a, cls=int(exp[1]))
    rows = _balance(rows)
    rows.sort(key=lambda r: (r["exp"][-1] == "L", r["exp"], r["t0"]))  # MPPT first

    X = np.stack([r["x"] for r in rows]).astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    exp = np.array([r["exp"] for r in rows])
    mode = np.array([MODES[e[-1]] for e in exp])
    t0 = np.array([r["t0"] for r in rows], dtype=np.float32)
    segment = np.array([r["segment"] for r in rows])

    std_tr, std_te = _standard_split(exp, y, t0)
    shift_tr, shift_te = np.flatnonzero(mode == "MPPT"), np.flatnonzero(mode == "IPPT")

    onset_txt = ", ".join(
        f"{e}: {'none' if o is None else ('whole file' if o == 0.0 else f'{o:.3f} s')}"
        for e, (o, _) in ONSETS.items())
    meta = DatasetMeta(
        name="gpvs",
        domain="energy",
        fs=FS,
        channel_names=CHANNELS,
        class_names=CLASS_NAMES,
        channel_groups=CHANNEL_GROUPS,
        channel_edges=CHANNEL_EDGES,
        group_fields=["mode", "experiment_id", "window_start_s", "segment"],
        splits=["standard", "shift"],
        description=(
            "GPVS-Faults: lab grid-connected PV system (PV array -> boost converter -> DC "
            "link -> three-phase inverter -> 50 Hz grid), 16 experiments = 8 conditions "
            "(fault-free + 7 faults) x 2 operating modes (MPPT, IPPT). Windows of 0.2 s "
            f"(T={T}) at {FS:g} Hz of PV current/voltage, DC-link voltage, three-phase "
            "currents and voltages, in A and V."
        ),
        source=f"{DATASET_URL} (doi:10.17632/n76t439f65.1); Bakdi et al., IJEPES 125 (2021) "
               "106457, doi:10.1016/j.ijepes.2020.106457",
        license="CC BY 4.0",
        notes=(
            "Fault definitions (Bakdi et al. 2021): F1 complete failure in one of the six "
            "inverter IGBTs; F2 one phase current feedback sensor fault 20% (the reading "
            "freezes at a constant: ib in F2M, ic in F2L); F3 intermittent grid voltage "
            "sags (not visible in the measured voltages; the data show the inverter trip "
            "and DC-link rise); F4 10-20% nonhomogeneous partial shading; F5 15% open "
            "circuit in the PV array (PV current drops to 0); F6 -20% gain of the boost "
            "converter's MPPT/IPPT PI controller; F7 +20% time constant of that PI "
            "controller (F6/F7 show only a +-1-2 V DC-link step; hard to separate from "
            "fault-free). Fault onsets were detected from the signals (see module "
            f"docstring): {onset_txt}. Only windows after onset carry the fault label; "
            "healthy windows before onset-guard (0.2 s abrupt faults, 0.5 s F3, 1.0 s "
            "F6/F7) are labelled fault-free (segment 'pre_fault'); F4 shows no onset, so "
            "both F4 files are labelled F4 throughout (both run at unusual operating "
            "points). segment 'post_trip' = fault windows after the inverter stopped "
            "(AC current ~0). Dropped derived channels Iabc, If, Vabc, Vf (PMU estimates). "
            f"Raw fs 10 kHz (the ReadMe's 9.9989 us is a typo for 99.99 us), decimated x{DOWN} "
            f"with anti-aliasing; at most {CAP} windows per (condition, mode). "
            "standard split = first 70% / last 30% in time of each (recording, label) "
            "segment; shift split = train MPPT, test IPPT. F3 is tiny (F3M has 0.26 s "
            "after its trip -> 1 window). Sensor calibration and units are as provided "
            "(A, V); grid ~110 V rms phase-to-neutral."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"mode": mode, "experiment_id": exp, "window_start_s": t0, "segment": segment},
        split_idx={"standard": {"train": std_tr, "test": std_te},
                   "shift": {"train": shift_tr, "test": shift_te}},
    )
