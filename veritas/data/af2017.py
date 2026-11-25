"""PhysioNet/CinC Challenge 2017 single-lead AF classification (domain: healthcare).

Source: training set of the 2017 Challenge (8,528 AliveCor single-lead ECGs,
300 Hz, 9-61 s), labels from REFERENCE-v3.csv (N, A, O, ~).

Preprocessing mirrors TimerBed / VL-Time (Liu et al., arXiv 2411.06018), whose
ECG task (43,673 series x 1,500 samples, 300 Hz, 4 classes) has exactly the
statistics of the TF-C ECG set (Zhang et al., NeurIPS 2022), which in turn used
the CLOCS code (Kiyasseh et al., ICML 2021): every recording is cut into
non-overlapping 1,500-sample (5 s) windows, the remainder is dropped, and each
window inherits the record label. That yields 54,593 windows; the published
43,673 is ~80% of it (one 80% part), consistent with this windowing. We then
decimate x2 (anti-aliased, resample_poly) to 150 Hz, giving T = 750 and keeping
the raw mV scale.

Split ``standard``: stratified 80/20 split of *records* (seed 0); all windows of
a record go to the same side. No shift split.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/",
    "https://physionet.org/files/challenge-2017/1.0.0/",
)
FILES = ("training2017.zip", "REFERENCE-v3.csv")
FS_RAW, WIN, DOWN = 300, 1500, 2
LABELS = ["N", "A", "O", "~"]
CLASS_NAMES = [
    "normal sinus rhythm",
    "atrial fibrillation",
    "other rhythm",
    "too noisy to classify",
]


def _fetch(rel: str, dest) -> None:
    """Download one file (relative to the dataset root) unless already present."""
    out = dest / rel
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    err = None
    for base in MIRRORS:
        for _ in range(3):
            try:
                with urllib.request.urlopen(base + rel, timeout=120) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                tmp.replace(out)
                return
            except Exception as e:  # noqa: BLE001 - retry, then try the next mirror
                err = e
    raise RuntimeError(f"could not download {rel}: {err}")


def _gain(header: str) -> float:
    """ADC gain (adu/mV) from a WFDB header, e.g. 'A00001.mat 16+24 1000/mV 16 0 ...'."""
    sig_line = header.strip().splitlines()[1].split()
    return float(sig_line[2].split("/")[0].split("(")[0])


def _load_af2017() -> TSDataset:
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    from sklearn.model_selection import train_test_split

    raw = RAW_DIR / "af2017"
    for rel in FILES:
        _fetch(rel, raw)

    ref = [ln.split(",") for ln in (raw / "REFERENCE-v3.csv").read_text().split()]
    label_of = {rec: LABELS.index(lab) for rec, lab in ref}

    windows, y, rec_id, orig_len, win_idx = [], [], [], [], []
    with zipfile.ZipFile(raw / "training2017.zip") as z:
        names = set(z.namelist())
        for rec in sorted(label_of):
            sig = loadmat(io.BytesIO(z.read(f"training2017/{rec}.mat")))["val"].ravel()
            hea = z.read(f"training2017/{rec}.hea").decode()
            mv = sig.astype(np.float64) / _gain(hea)
            n_win = len(mv) // WIN
            for k in range(n_win):
                windows.append(mv[k * WIN:(k + 1) * WIN])
                y.append(label_of[rec])
                rec_id.append(rec)
                orig_len.append(len(mv) / FS_RAW)
                win_idx.append(k)
        assert len(label_of) == sum(n.endswith(".mat") for n in names), "record count mismatch"

    X = resample_poly(np.stack(windows), up=1, down=DOWN, axis=-1)[:, None, :].astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    rec_id = np.asarray(rec_id)
    n_rec = len(label_of)

    # stratified 80/20 split of records, then expand to windows
    recs = np.asarray(sorted(label_of))
    rec_y = np.asarray([label_of[r] for r in recs])
    tr_rec, te_rec = train_test_split(recs, test_size=0.2, stratify=rec_y, random_state=0)
    is_te = np.isin(rec_id, te_rec)
    tr, te = np.flatnonzero(~is_te), np.flatnonzero(is_te)

    rec_counts = np.bincount(rec_y, minlength=4).tolist()
    meta = DatasetMeta(
        name="af2017",
        domain="healthcare",
        fs=FS_RAW / DOWN,
        channel_names=["ECG (single lead, AliveCor)"],
        class_names=CLASS_NAMES,
        channel_groups={"ecg": [0]},
        channel_edges=[],
        group_fields=["record_id", "original_length_s", "window_index"],
        splits=["standard"],
        description=(
            "Single-lead hand-held (AliveCor) ECG, 5-s windows; classify the rhythm of the "
            "recording as normal sinus rhythm, atrial fibrillation, other rhythm, or too noisy."
        ),
        source="https://physionet.org/content/challenge-2017/1.0.0/",
        license="Open Data Commons Attribution License v1.0 (ODC-By 1.0); cite Clifford et al. 2017",
        notes=(
            f"{n_rec} training records (record labels N/A/O/~ = {rec_counts}) from REFERENCE-v3.csv. "
            "Windowing mirrors TimerBed/VL-Time and TF-C/CLOCS: non-overlapping 1,500-sample (5 s @ 300 Hz) "
            "windows, remainder dropped, label inherited from the record. This gives 54,593 windows; TF-C/TimerBed "
            "report 43,673, i.e. ~80% of that (our 80% record split has 43,667 train windows), so they apparently "
            "used one 80% part. The CLOCS 2017 loader itself is not public, so this match is inferred, not "
            "verified against code. "
            "Then anti-aliased x2 decimation (scipy resample_poly) to 150 Hz -> T=750. Units mV "
            "(raw adu / header gain 1000), no filtering or normalisation. Window labels are record-level: "
            "a window of an AF or 'other' record may itself look normal, and '~' windows may contain clean beats. "
            "standard split = stratified 80/20 by record (seed 0); windows of one record never straddle. "
            "Patients are not identified in the Challenge data, so a patient could own several records."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={
            "record_id": rec_id,
            "original_length_s": np.asarray(orig_len, dtype=np.float32),
            "window_index": np.asarray(win_idx, dtype=np.int64),
        },
        split_idx={"standard": {"train": tr, "test": te}},
    )


@register("af2017")
def build_af2017() -> TSDataset:
    return _load_af2017()
