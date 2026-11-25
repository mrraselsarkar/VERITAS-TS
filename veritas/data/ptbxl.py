"""PTB-XL 12-lead ECG, diagnostic superclasses (domain: healthcare).

Source: PTB-XL v1.0.3 (Wagner et al., Sci Data 2020), 21,799 10-s 12-lead ECGs
from 18,869 patients, recorded 1984-2001 with Schiller devices. We use the
100 Hz version (records100, 12 x 1,000 samples, mV) and only download the
records we keep.

Labels follow the official PhysioNet example: every SCP code of a record that
is a diagnostic statement in scp_statements.csv contributes its
``diagnostic_class`` (NORM, MI, STTC, CD, HYP), regardless of likelihood. We
keep the 16,244 records whose set of superclasses has exactly one element.

Splits
- ``standard``: strat_fold 1-9 train, fold 10 test (the recommended split;
  folds are patient-disjoint).
- ``shift``: device generation / site shift. Train = Schiller CS-series
  recorders (CS-12, CS-12 E, CS100; sites 0 and 2), test = AT-series recorders
  (AT-6 variants, AT-60; mainly sites 1 and 3). Site and device are almost
  fully confounded in PTB-XL, so this is a joint site+device shift. CS records
  of the patients who also have an AT record are dropped from shift-train so
  that no patient straddles.
"""
from __future__ import annotations

import ast
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/",
    "https://physionet.org/files/ptb-xl/1.0.3/",
)
META_FILES = ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt")
FS = 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = [
    "normal ECG",
    "myocardial infarction",
    "ST/T change",
    "conduction disturbance",
    "hypertrophy",
]
CHANNEL_GROUPS = {
    "limb_bipolar (I, II, III)": [0, 1, 2],
    "augmented_limb (aVR, aVL, aVF)": [3, 4, 5],
    "septal_anterior_precordial (V1-V3)": [6, 7, 8],
    "lateral_precordial (V4-V6)": [9, 10, 11],
}


def _edges() -> list[tuple[int, int]]:
    i = {name: k for k, name in enumerate(LEADS)}
    pairs = [
        ("I", "II"), ("II", "III"), ("I", "III"),           # Einthoven: III = II - I
        ("aVR", "I"), ("aVR", "II"),                         # aVR = -(I + II) / 2
        ("aVL", "I"), ("aVL", "III"),                        # aVL = (I - III) / 2
        ("aVF", "II"), ("aVF", "III"),                       # aVF = (II + III) / 2
        ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6"),  # precordial chain
        ("I", "V5"), ("I", "V6"), ("aVL", "V5"), ("aVL", "V6"),                # lateral territory
        ("II", "aVF"), ("III", "aVF"), ("II", "III"),        # inferior territory
    ]
    out = []
    for a, b in pairs:
        e = tuple(sorted((i[a], i[b])))
        if e not in out:
            out.append(e)
    return out


def _fetch(rels: list[str], dest, workers: int = 32) -> None:
    """Download files (relative to the dataset root) into dest, skipping existing ones."""
    from tqdm import tqdm

    todo = [r for r in rels if not (dest / r).exists()]

    def one(rel: str) -> None:
        out = dest / rel
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

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(tqdm(ex.map(one, todo), total=len(todo), desc="ptbxl download"))


def _load_ptbxl() -> TSDataset:
    import pandas as pd
    import wfdb

    raw = RAW_DIR / "ptbxl"
    _fetch(list(META_FILES), raw, workers=3)

    db = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]["diagnostic_class"]
    n_all = len(db)
    db["superclasses"] = db.scp_codes.apply(
        lambda s: sorted({diag[c] for c in ast.literal_eval(s) if c in diag.index})
    )
    n_by_count = db.superclasses.str.len().value_counts().sort_index().to_dict()
    db = db[db.superclasses.str.len() == 1].copy()
    db["label"] = db.superclasses.str[0].map(SUPERCLASSES.index)

    _fetch([f"{f}.{ext}" for f in db.filename_lr for ext in ("hea", "dat")], raw)

    X = np.empty((len(db), len(LEADS), 10 * FS), dtype=np.float32)
    for k, rel in enumerate(db.filename_lr):
        sig, fields = wfdb.rdsamp(str(raw / rel))
        if k == 0:
            assert [s.upper() for s in fields["sig_name"]] == [s.upper() for s in LEADS], fields["sig_name"]
            assert fields["fs"] == FS and set(fields["units"]) == {"mV"}, fields
        X[k] = sig.T
    np.nan_to_num(X, copy=False)

    y = db.label.to_numpy(np.int64)
    patient = db.patient_id.to_numpy().astype(np.int64)
    fold = db.strat_fold.to_numpy(np.int64)
    site = db.site.fillna(-1).to_numpy().astype(np.int64)
    device = db.device.str.replace(r"\s+", " ", regex=True).str.strip().to_numpy().astype(str)

    std_tr, std_te = np.flatnonzero(fold <= 9), np.flatnonzero(fold == 10)
    assert not np.intersect1d(patient[std_tr], patient[std_te]).size

    is_at = np.char.startswith(device, "AT")
    sh_te = np.flatnonzero(is_at)
    sh_tr = np.flatnonzero(~is_at & ~np.isin(patient, patient[is_at]))
    n_dropped = int((~is_at).sum() - len(sh_tr))

    counts = np.bincount(y, minlength=5)
    meta = DatasetMeta(
        name="ptbxl",
        domain="healthcare",
        fs=float(FS),
        channel_names=LEADS,
        class_names=CLASS_NAMES,
        channel_groups=CHANNEL_GROUPS,
        channel_edges=_edges(),
        group_fields=["patient_id", "strat_fold", "site", "device"],
        splits=["standard", "shift"],
        description=(
            "Standard 12-lead resting ECG, 10 s; classify the single diagnostic superclass: normal ECG, "
            "myocardial infarction, ST/T change, conduction disturbance or hypertrophy."
        ),
        source="https://physionet.org/content/ptb-xl/1.0.3/",
        license="Creative Commons Attribution 4.0 International (CC BY 4.0); cite Wagner et al. 2020",
        notes=(
            f"records100 (100 Hz, mV, 12 x 1000), no filtering/normalisation. Of {n_all} records, number of "
            f"distinct diagnostic superclasses per record = {n_by_count}; only the {len(db)} records with "
            "exactly one superclass are kept (official-example aggregation, all likelihoods incl. 0 count). "
            f"Class counts {dict(zip(SUPERCLASSES, counts.tolist()))}. 'standard' = strat_fold 1-9 vs 10 "
            "(patient-disjoint). 'shift' = device generation/site: train CS-series (CS-12, CS-12 E, CS100; sites "
            "0, 2), test AT-series (AT-6 variants, AT-60; sites 1, 3+); site and device are nearly confounded. "
            f"{n_dropped} CS records of patients who also have AT records are left out of shift-train. "
            "Labels come from cardiologist reports (some auto-generated, see validated_by_human), so label noise "
            "exists. Lead order follows the WFDB header (I, II, III, aVR, aVL, aVF, V1-V6). Device strings have "
            "whitespace collapsed."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"patient_id": patient, "strat_fold": fold, "site": site, "device": device},
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": sh_tr, "test": sh_te},
        },
    )


@register("ptbxl")
def build_ptbxl() -> TSDataset:
    return _load_ptbxl()
