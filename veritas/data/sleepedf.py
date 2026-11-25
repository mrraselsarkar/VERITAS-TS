"""Sleep-EDF Expanded, 30-s sleep staging (domain: healthcare).

Source: Sleep-EDF Database Expanded v1.0.0 (Kemp et al. 2000; Goldberger et al.
2000). Subset to keep the download moderate (~1.6 GB):

- Sleep Cassette (SC, healthy, ambulatory at home, age 25-34 here): subjects
  0-9, both nights (SC40{0..9}{1,2}E0).
- Sleep Telemetry (ST, in hospital, mild sleep-onset difficulty, temazepam
  study): the first 10 subjects with both nights (1, 2, 4-11); one night is
  placebo, the other temazepam (from ST-subjects.xls).

Channels kept, all recorded at 100 Hz in both cohorts: EEG Fpz-Cz, EEG Pz-Oz,
horizontal EOG. Submental EMG is excluded because SC stores it as a 1 Hz
envelope while ST has it at 100 Hz. Signals stay in microvolts.

Epochs: 30-s epochs on the hypnogram grid. R&K stages are mapped to AASM-like
classes Wake, N1, N2, N3 (= S3 + S4), REM; movement time and unscored epochs are
dropped. Each night is trimmed to 30 min of wake before the first and after the
last sleep epoch. The whole night is decimated x3 (anti-aliased resample_poly,
100 -> 33.3 Hz) before cutting epochs, so each epoch is 3 x 1,000 samples.

Splits
- ``standard``: SC only, split by subject (7 train / 3 test subjects).
- ``shift``: train = all SC epochs, test = all ST epochs (home vs hospital,
  different recorder, older subjects with sleep complaints, placebo/temazepam).
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/",
    "https://physionet.org/files/sleep-edfx/1.0.0/",
)
META_FILES = ("SHA256SUMS.txt", "SC-subjects.xls", "ST-subjects.xls")
SC_SUBJECTS = list(range(10))
ST_SUBJECTS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
FS_RAW, DOWN, EPOCH_S = 100, 3, 30
T = FS_RAW * EPOCH_S // DOWN  # 1000
TRIM_EPOCHS = 60  # 30 min of wake kept on each side of the sleep period
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}  # "Sleep stage ?" and "Movement time" are dropped
CLASS_NAMES = [
    "wake",
    "N1 (light sleep, stage 1)",
    "N2 (stage 2 sleep, spindles and K-complexes)",
    "N3 (deep slow-wave sleep, R&K stages 3+4)",
    "REM sleep",
]


def _fetch(rels: list[str], dest, sha: dict[str, str] | None = None, workers: int = 8) -> None:
    """Download files (relative to the dataset root), skipping existing ones; check SHA-256."""
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
                    with urllib.request.urlopen(base + rel, timeout=300) as r, open(tmp, "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    if sha and rel in sha:
                        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
                        if digest != sha[rel]:
                            raise IOError(f"SHA-256 mismatch for {rel}")
                    tmp.replace(out)
                    return
                except Exception as e:  # noqa: BLE001 - retry, then try the next mirror
                    err = e
        raise RuntimeError(f"could not download {rel}: {err}")

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(tqdm(ex.map(one, todo), total=len(todo), desc="sleepedf download"))


def _night_list(raw, sha: dict[str, str]) -> list[dict]:
    """PSG/hypnogram file pairs plus subject metadata for the chosen subset."""
    import pandas as pd

    files = sorted(sha)
    sc_meta = pd.read_excel(raw / "SC-subjects.xls")  # subject, night, age, sex (F=1), LightsOff
    st_meta = pd.read_excel(raw / "ST-subjects.xls", header=None, skiprows=2).iloc[:, :6]
    st_meta.columns = ["subject", "age", "sex_m1_f2", "placebo_night", "placebo_lights_off", "temazepam_night"]

    nights = []
    for cohort, subjects in (("SC", SC_SUBJECTS), ("ST", ST_SUBJECTS)):
        for s in subjects:
            for night in (1, 2):
                prefix = f"sleep-cassette/SC4{s:02d}{night}E" if cohort == "SC" else \
                    f"sleep-telemetry/ST7{s:02d}{night}J"
                psg = [f for f in files if f.startswith(prefix) and f.endswith("-PSG.edf")]
                hyp = [f for f in files if f.startswith(prefix) and f.endswith("-Hypnogram.edf")]
                assert len(psg) == 1 and len(hyp) == 1, (prefix, psg, hyp)
                if cohort == "SC":
                    row = sc_meta[(sc_meta.subject == s) & (sc_meta.night == night)].iloc[0]
                    age, sex, drug = int(row.age), "F" if int(row.iloc[3]) == 1 else "M", "none"
                else:
                    row = st_meta[st_meta.subject == s].iloc[0]
                    age, sex = int(row.age), "M" if int(row.sex_m1_f2) == 1 else "F"
                    drug = "placebo" if int(row.placebo_night) == night else "temazepam"
                    assert drug == "placebo" or int(row.temazepam_night) == night
                nights.append(dict(psg=psg[0], hyp=hyp[0], cohort=cohort, subject=f"{cohort}{s:02d}",
                                   night=night, age=age, sex=sex, drug=drug))
    return nights


def _epochs(raw, night: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(epochs (n, 3, T) in uV, labels (n,), epoch index within the night)."""
    import mne
    from scipy.signal import resample_poly

    rec = mne.io.read_raw_edf(raw / night["psg"], include=CHANNELS, preload=True, verbose="error")
    assert rec.ch_names == CHANNELS and rec.info["sfreq"] == FS_RAW, (rec.ch_names, rec.info["sfreq"])
    sig = rec.get_data(units="uV")  # (3, N)
    ann = mne.read_annotations(raw / night["hyp"])

    n_ep = sig.shape[1] // (FS_RAW * EPOCH_S)
    lab = np.full(n_ep, -1, dtype=np.int64)
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc not in STAGE_MAP:
            continue
        assert onset % EPOCH_S == 0 and dur % EPOCH_S == 0, (night["hyp"], onset, dur)
        a = int(onset // EPOCH_S)
        lab[a:min(a + int(dur // EPOCH_S), n_ep)] = STAGE_MAP[desc]

    sleep = np.flatnonzero(lab >= 1)
    lo, hi = max(sleep[0] - TRIM_EPOCHS, 0), min(sleep[-1] + TRIM_EPOCHS, n_ep - 1)
    keep = np.arange(lo, hi + 1)
    keep = keep[lab[keep] >= 0]

    low = resample_poly(sig[:, : n_ep * FS_RAW * EPOCH_S], up=1, down=DOWN, axis=-1)
    ep = low.reshape(len(CHANNELS), n_ep, T).transpose(1, 0, 2)[keep]
    return ep.astype(np.float32), lab[keep], keep


def _load_sleepedf() -> TSDataset:
    raw = RAW_DIR / "sleepedf"
    _fetch(list(META_FILES), raw, workers=3)
    sha = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        sha[rel.strip()] = digest
    nights = _night_list(raw, sha)
    _fetch([f for n in nights for f in (n["psg"], n["hyp"])], raw, sha=sha)

    Xs, ys = [], []
    g = {k: [] for k in ("subject", "night", "cohort", "drug", "age", "sex", "epoch_index")}
    for night in nights:
        ep, lab, idx = _epochs(raw, night)
        Xs.append(ep)
        ys.append(lab)
        for k in ("subject", "night", "cohort", "drug", "age", "sex"):
            g[k].append(np.repeat(night[k], len(lab)))
        g["epoch_index"].append(idx)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    groups = {k: np.concatenate(v) for k, v in g.items()}
    for k in ("subject", "cohort", "drug", "sex"):
        groups[k] = groups[k].astype(str)
    for k in ("night", "age", "epoch_index"):
        groups[k] = groups[k].astype(np.int64)

    # standard: SC subjects, ~70/30 by subject; redraw until every class is on both sides
    sc_subj = np.array([f"SC{s:02d}" for s in SC_SUBJECTS])
    n_test = round(0.3 * len(sc_subj))
    for seed in range(100):
        te_subj = np.random.default_rng(seed).choice(sc_subj, size=n_test, replace=False)
        is_sc = groups["cohort"] == "SC"
        is_te = np.isin(groups["subject"], te_subj)
        std_tr, std_te = np.flatnonzero(is_sc & ~is_te), np.flatnonzero(is_sc & is_te)
        if len(np.unique(y[std_tr])) == 5 and len(np.unique(y[std_te])) == 5:
            break
    sh_tr = np.flatnonzero(groups["cohort"] == "SC")
    sh_te = np.flatnonzero(groups["cohort"] == "ST")

    counts = {c: np.bincount(y[groups["cohort"] == c], minlength=5).tolist() for c in ("SC", "ST")}
    meta = DatasetMeta(
        name="sleepedf",
        domain="healthcare",
        fs=FS_RAW / DOWN,
        channel_names=CHANNELS,
        class_names=CLASS_NAMES,
        channel_groups={
            "frontal EEG (Fpz-Cz)": [0],
            "parieto-occipital EEG (Pz-Oz)": [1],
            "horizontal EOG (eye movements)": [2],
        },
        channel_edges=[(0, 1), (0, 2)],
        group_fields=["subject", "night", "cohort", "drug", "age", "sex", "epoch_index"],
        splits=["standard", "shift"],
        description=(
            "30-s polysomnography epochs (two EEG derivations and horizontal EOG); classify the sleep stage: "
            "wake, N1, N2, N3 or REM."
        ),
        source="https://physionet.org/content/sleep-edfx/1.0.0/",
        license="Open Data Commons Attribution License v1.0 (ODC-By 1.0); cite Kemp et al. 2000",
        notes=(
            f"Subset: SC subjects {SC_SUBJECTS} and ST subjects {ST_SUBJECTS}, both nights each "
            f"({len(nights)} nights). Channels EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal (100 Hz in both cohorts); "
            "EMG excluded (1 Hz envelope in SC). Units uV, no filtering/normalisation. Whole night decimated "
            "x3 with scipy resample_poly (fs 33.3 Hz, so content above ~16 Hz, e.g. beta and fast spindles, is "
            "removed), then cut into 30-s epochs of 1,000 samples on the hypnogram grid. Labels: R&K W/1/2/3/4/R "
            "-> wake/N1/N2/N3(3+4)/REM; 'Movement time' and 'Sleep stage ?' dropped. Each night trimmed to 30 min "
            f"of wake before the first and after the last sleep epoch. Class counts SC {counts['SC']}, ST "
            f"{counts['ST']}. standard = SC only, subjects {sorted(str(s) for s in set(sc_subj) - set(te_subj))} "
            f"train / {sorted(str(s) for s in te_subj)} test (seed {seed}); shift = SC train / ST test. "
            "Amplitude ranges differ by cohort (part of the shift): SC EEG is digitised with 12 bits over about "
            "+-190 uV and saturates there (EOG about +-1000 uV), ST uses 14 bits over +-3000 uV; ST04 night 1 "
            "Pz-Oz is largely artefact (99th percentile ~3 mV). ST drug per night from "
            "ST-subjects.xls (placebo vs temazepam); SC drug = none. Sex coding differs in the source sheets "
            "(SC F=1, ST M=1) and is normalised to F/M here. Hypnograms are manual R&K scores; N1 is rare and "
            "the least reliable class. Raw files checked against SHA256SUMS.txt."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta, groups=groups,
        split_idx={
            "standard": {"train": std_tr, "test": std_te},
            "shift": {"train": sh_tr, "test": sh_te},
        },
    )


@register("sleepedf")
def build_sleepedf() -> TSDataset:
    return _load_sleepedf()
