"""Tennessee Eastman Process (domain: industrial), Rieth et al. (2017) extended simulation.

52 process variables (XMEAS 1-41 measurements, XMV 1-11 manipulated variables)
sampled every 3 min, 21 classes: normal operation (fault 0) and the 20
process disturbances IDV 1-20 of Downs & Vogel (1993). The data set holds 500
independent simulation runs per fault in each of four R files:

    TEP_FaultFree_Training.RData   fault 0,     500 runs x 500 samples (25 h)
    TEP_Faulty_Training.RData      faults 1-20, 500 runs x 500 samples, onset after sample 20 (1 h)
    TEP_FaultFree_Testing.RData    fault 0,     500 runs x 960 samples (48 h)
    TEP_Faulty_Testing.RData       faults 1-20, 500 runs x 960 samples, onset after sample 160 (8 h)

Preprocessing
- runs: the first 50 runs (``simulationRun`` 1-50) of every fault from the
  training files and the first 25 runs of every fault from the testing files.
- windows: non-overlapping windows of 100 samples (5 h) that start right after
  the fault onset, i.e. samples 21-420 of a training run (4 windows; samples
  421-500 dropped) and 161-960 of a testing run (8 windows). Fault-free runs are
  cut at the same positions. 5 h is long enough to contain the transient of
  most detectable faults and gives several windows per run; slow faults (e.g.
  IDV 13 drift, IDV 18) can still look normal in the first window, which is
  inherent label noise of after-onset labelling.
- values in engineering units (kscmh, kg/h, kPa gauge, %, deg C, m3/h, kW,
  mol %, valve %), no normalisation. Composition analysers are sampled every
  6 or 15 min with dead time in the TE model, so XMEAS 23-41 are piecewise
  constant within a window.

Splits
- ``standard``: training-file windows only, split by simulation run (70 % of
  the 50 run indices -> train, 30 % -> test, seed 0). A run index is kept on
  one side for *all* faults because the simulator reuses the random seed of a
  run index across faults (run k of every fault has identical pre-onset data,
  verified on faults 1 vs 2), so splitting per fault would leak noise
  realisations.
- ``shift``: train on all training-file windows (onset at 1 h, windows up to
  20 h after onset), test on testing-file windows (different seeds -- no run
  shares its initial state with a training run, verified -- 8 h of normal
  operation before onset, windows up to 40 h after onset). Chosen over
  early-vs-late windows because it keeps runs disjoint and follows the
  intended train/test protocol of the data set; windows 5-8 of each test run
  lie beyond the post-onset horizon seen in training (``onset_offset``).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import RAW_DIR, DatasetMeta, TSDataset, register

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{fid}"
FILES = {  # file name -> Harvard Dataverse file id (doi:10.7910/DVN/6C3JR1)
    "TEP_FaultFree_Training.RData": 3031241,
    "TEP_Faulty_Training.RData": 3031242,
    "TEP_FaultFree_Testing.RData": 3031240,
    "TEP_Faulty_Testing.RData": 3031243,
}
W = 100                 # window length in samples (3 min each -> 5 h)
N_RUNS_TRAINFILE = 50   # runs per fault taken from the training files
N_RUNS_TESTFILE = 25    # runs per fault taken from the testing files
TEST_RUN_FRAC = 0.3     # standard split: share of run indices held out
# (run length, fault onset = number of samples before the fault is switched on)
LAYOUT = {"training": (500, 20), "testing": (960, 160)}

# Downs & Vogel (1993), Tables 4-5; units as in the original paper.
XMEAS = [
    "A feed (stream 1), kscmh",
    "D feed (stream 2), kg/h",
    "E feed (stream 3), kg/h",
    "A and C feed (stream 4), kscmh",
    "recycle flow (stream 8), kscmh",
    "reactor feed rate (stream 6), kscmh",
    "reactor pressure, kPa gauge",
    "reactor level, %",
    "reactor temperature, deg C",
    "purge rate (stream 9), kscmh",
    "product separator temperature, deg C",
    "product separator level, %",
    "product separator pressure, kPa gauge",
    "product separator underflow (stream 10), m3/h",
    "stripper level, %",
    "stripper pressure, kPa gauge",
    "stripper underflow (stream 11), m3/h",
    "stripper temperature, deg C",
    "stripper steam flow, kg/h",
    "compressor work, kW",
    "reactor cooling water outlet temperature, deg C",
    "separator cooling water outlet temperature, deg C",
] + [f"reactor feed analysis (stream 6): component {c}, mol %" for c in "ABCDEF"] \
  + [f"purge gas analysis (stream 9): component {c}, mol %" for c in "ABCDEFGH"] \
  + [f"product analysis (stream 11): component {c}, mol %" for c in "DEFGH"]
XMV = [
    "D feed flow valve (stream 2), %",
    "E feed flow valve (stream 3), %",
    "A feed flow valve (stream 1), %",
    "A and C feed flow valve (stream 4), %",
    "compressor recycle valve, %",
    "purge valve (stream 9), %",
    "separator pot liquid flow valve (stream 10), %",
    "stripper liquid product flow valve (stream 11), %",
    "stripper steam valve, %",
    "reactor cooling water flow valve, %",
    "condenser cooling water flow valve, %",
]
assert len(XMEAS) == 41 and len(XMV) == 11
CHANNEL_NAMES = [f"XMEAS {i + 1}: {s}" for i, s in enumerate(XMEAS)] + \
                [f"XMV {i + 1}: {s}" for i, s in enumerate(XMV)]

FAULTS = [
    "normal operation (no fault)",
    "A/C feed ratio, B composition constant (stream 4) - step",
    "B composition, A/C ratio constant (stream 4) - step",
    "D feed temperature (stream 2) - step",
    "reactor cooling water inlet temperature - step",
    "condenser cooling water inlet temperature - step",
    "A feed loss (stream 1) - step",
    "C header pressure loss, reduced availability (stream 4) - step",
    "A, B, C feed composition (stream 4) - random variation",
    "D feed temperature (stream 2) - random variation",
    "C feed temperature (stream 4) - random variation",
    "reactor cooling water inlet temperature - random variation",
    "condenser cooling water inlet temperature - random variation",
    "reaction kinetics - slow drift",
    "reactor cooling water valve - sticking",
    "condenser cooling water valve - sticking",
] + ["unknown disturbance type (not disclosed by Downs & Vogel)"] * 5
CLASS_NAMES = ["fault 0: normal operation (no fault)"] + \
              [f"fault {k} (IDV {k}): {FAULTS[k]}" for k in range(1, 21)]


def _m(k: int) -> int:   # channel index of XMEAS k
    return k - 1


def _v(k: int) -> int:   # channel index of XMV k
    return 40 + k


CHANNEL_GROUPS = {
    "feeds": [_m(k) for k in (1, 2, 3, 4)],
    "reactor": [_m(k) for k in (6, 7, 8, 9, 21)],
    "separator": [_m(k) for k in (11, 12, 13, 14, 22)],
    "stripper": [_m(k) for k in (15, 16, 17, 18, 19)],
    "compressor_recycle_purge": [_m(k) for k in (5, 10, 20)],
    "analyzers": [_m(k) for k in range(23, 42)],
    "manipulated": [_v(k) for k in range(1, 12)],
}


def _edges() -> list[tuple[int, int]]:
    m, v = _m, _v
    e = [
        # fresh feeds, recycle and stripper overhead mix into the reactor feed (stream 6)
        (m(1), m(6)), (m(2), m(6)), (m(3), m(6)), (m(5), m(6)), (m(16), m(6)),
        (m(4), m(16)),                                    # stream 4 enters the stripper
        # reactor
        (m(6), m(7)), (m(7), m(8)), (m(7), m(9)), (m(8), m(9)), (m(9), m(21)),
        # reactor effluent -> condenser -> separator
        (m(7), m(13)), (m(9), m(11)),
        (m(11), m(12)), (m(11), m(13)), (m(12), m(13)), (m(12), m(14)), (m(11), m(22)),
        # separator vapour -> compressor -> recycle; purge
        (m(13), m(20)), (m(20), m(5)), (m(13), m(10)),
        # separator liquid (stream 10) -> stripper
        (m(14), m(15)),
        (m(15), m(16)), (m(15), m(17)), (m(16), m(18)), (m(18), m(19)),
    ]
    # analysers attached to the stream they sample
    e += [(m(6), m(k)) for k in range(23, 29)]           # reactor feed, stream 6
    e += [(m(10), m(k)) for k in range(29, 37)]          # purge gas, stream 9
    e += [(m(17), m(k)) for k in range(37, 42)]          # product, stream 11
    # neighbouring components within one analyser
    for lo, hi in ((23, 28), (29, 36), (37, 41)):
        e += [(m(k), m(k + 1)) for k in range(lo, hi)]
    # same component in linked streams: purge gas = recycle gas that feeds the reactor;
    # separator vapour/liquid equilibrium links purge and product
    e += [(m(23 + i), m(29 + i)) for i in range(6)]      # A-F reactor feed <-> purge
    e += [(m(32 + i), m(37 + i)) for i in range(5)]      # D-H purge <-> product
    # valves acting on the flow / temperature they set
    e += [(v(1), m(2)), (v(2), m(3)), (v(3), m(1)), (v(4), m(4)), (v(5), m(5)),
          (v(6), m(10)), (v(7), m(14)), (v(8), m(17)), (v(9), m(19)), (v(9), m(18)),
          (v(10), m(9)), (v(10), m(21)), (v(11), m(11)), (v(11), m(22))]
    # controller pairings of Ricker's (1996) decentralised control
    e += [(m(7), v(6)),     # reactor pressure -> purge
          (m(12), v(7)),    # separator level -> separator underflow
          (m(15), v(8)),    # stripper level -> stripper product flow
          (m(8), v(11)),    # reactor level -> separator temperature setpoint (condenser CW)
          (m(40), v(1)), (m(40), v(2)),   # % G in product -> D/E feed ratio
          (m(23), v(3)),    # A in reactor feed -> A feed
          (m(25), v(4))]    # A+C in reactor feed -> A and C feed
    e = sorted({tuple(sorted(p)) for p in e})
    return e


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (veritas-ts data loader)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.replace(dest)


def _windows(path: Path, part: str, n_runs: int) -> tuple[np.ndarray, ...]:
    """Windows of the first n_runs runs of every fault in one RData file.

    Returns X (n, 52, W) float32, fault, run, window_start (1-based sample),
    onset_offset (samples between onset and window start)."""
    import pyreadr

    length, onset = LAYOUT[part]
    df = next(iter(pyreadr.read_r(str(path)).values()))
    df = df[df["simulationRun"] <= n_runs].sort_values(["faultNumber", "simulationRun", "sample"])
    faults = np.sort(df["faultNumber"].unique()).astype(int)
    vals = df.iloc[:, 3:].to_numpy(np.float64)
    assert vals.shape == (len(faults) * n_runs * length, 52), vals.shape
    runs = vals.reshape(len(faults), n_runs, length, 52)
    nw = (length - onset) // W
    starts0 = onset + np.arange(nw) * W                       # 0-based start
    X = np.stack([runs[:, :, s:s + W] for s in starts0], axis=2)  # (F, R, nw, W, 52)
    X = X.reshape(-1, W, 52).transpose(0, 2, 1).astype(np.float32)
    F, R = np.meshgrid(faults, np.arange(1, n_runs + 1), indexing="ij")
    rep = lambda a: np.repeat(a.ravel(), nw)                   # noqa: E731
    fault, run = rep(F), rep(R)
    start = np.tile(starts0 + 1, len(faults) * n_runs)
    return X, fault, run, start, start - 1 - onset


def load_tep() -> TSDataset:
    raw = RAW_DIR / "tep"
    for fname, fid in FILES.items():
        _download(DATAVERSE.format(fid=fid), raw / fname)

    parts = []
    for part, n_runs in (("training", N_RUNS_TRAINFILE), ("testing", N_RUNS_TESTFILE)):
        for kind in ("FaultFree", "Faulty"):
            X, f, r, s, o = _windows(raw / f"TEP_{kind}_{part.capitalize()}.RData", part, n_runs)
            parts.append((X, f, r, s, o, np.full(len(f), part)))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts]).astype(np.int64)
    run_id = np.concatenate([p[2] for p in parts]).astype(np.int64)
    window_start = np.concatenate([p[3] for p in parts]).astype(np.int64)
    onset_offset = np.concatenate([p[4] for p in parts]).astype(np.int64)
    source_file = np.concatenate([p[5] for p in parts])

    is_trainfile = source_file == "training"
    rng = np.random.default_rng(0)
    test_runs = rng.permutation(np.arange(1, N_RUNS_TRAINFILE + 1))[: int(round(TEST_RUN_FRAC * N_RUNS_TRAINFILE))]
    held = np.isin(run_id, test_runs)
    std_tr = np.flatnonzero(is_trainfile & ~held)
    std_te = np.flatnonzero(is_trainfile & held)
    sh_tr, sh_te = np.flatnonzero(is_trainfile), np.flatnonzero(~is_trainfile)

    meta = DatasetMeta(
        name="tep",
        domain="industrial",
        fs=1 / 180,  # one sample every 3 min
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        channel_groups=CHANNEL_GROUPS,
        channel_edges=_edges(),
        group_fields=["run_id", "source_file", "window_start", "onset_offset"],
        splits=["standard", "shift"],
        description=(
            "Tennessee Eastman chemical process simulation (Rieth et al. 2017 extended set): "
            "reactor, condenser, vapour-liquid separator, recycle compressor and product "
            "stripper making products G and H from gaseous reactants A, C, D, E. 52 variables "
            "(41 measurements XMEAS, 11 manipulated valves XMV) every 3 min; 5 h windows taken "
            "after fault onset. 21 classes: normal operation and process faults IDV 1-20."
        ),
        source="https://doi.org/10.7910/DVN/6C3JR1",
        license="Public domain dedication (Dataverse terms by Pacific Science & Engineering "
                "Group: copy, modify, distribute for any lawful purpose, no endorsement implied); "
                "cite Rieth, Amsel, Tran & Cook (2017) and Downs & Vogel (1993).",
        notes=(
            f"Runs 1-{N_RUNS_TRAINFILE} of each fault from the training files (500 samples, "
            f"onset after sample 20) and runs 1-{N_RUNS_TESTFILE} from the testing files (960 "
            f"samples, onset after sample 160). Non-overlapping {W}-sample (5 h) windows starting "
            "at onset: 4 per training run (samples 21-420), 8 per testing run (161-960); "
            "fault-free runs cut at the same positions. window_start = 1-based first sample in "
            "the run; onset_offset = samples from onset to window start. Engineering units, no "
            "normalisation; analyser channels (XMEAS 23-41) are sample-and-hold signals. "
            "standard split: training-file windows, 35 run indices train / 15 test, the same "
            "run indices for every fault (the simulator reuses a run index's random seed across "
            "faults: identical pre-onset data). shift split: training-file runs -> "
            "testing-file runs (different seeds, longer pre-onset period, windows up to 40 h "
            "after onset vs 20 h in training). Faults 3, 9 and 15 are widely reported as "
            "practically undetectable (near-identical to normal operation), so expect "
            "confusion among classes 0, 3, 9, 15; IDV 16-20 are of unknown type. "
            "Channel edges: flowsheet adjacency (streams/units), analyser-to-stream, "
            "same-component links between analysers, valve-to-controlled-variable and "
            "Ricker (1996) decentralised-control pairings; the controller list is from the "
            "literature and was not re-verified against the simulation code. "
            "Class order = fault number. Sanity baseline (random forest on per-channel "
            "mean/std/min/max/drift): accuracy 0.875 standard, 0.872 shift (0.877 on test "
            "windows 0-20 h after onset, 0.867 on 20-40 h), so the shift is mild; errors "
            "concentrate in classes 0, 3, 9, 15."
        ),
    )
    return TSDataset(
        X=X, y=y, meta=meta,
        groups={"run_id": run_id, "source_file": source_file,
                "window_start": window_start, "onset_offset": onset_offset},
        split_idx={"standard": {"train": std_tr, "test": std_te},
                   "shift": {"train": sh_tr, "test": sh_te}},
    )


@register("tep")
def build_tep() -> TSDataset:
    return load_tep()
