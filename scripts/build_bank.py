"""Counterfactual bank: verification outcomes for every single-cell claim (proposal, Method part 4).

For each selected series and each (window, group) cell, m necessity edits
(the cell regenerated towards a random other class) and m sufficiency edits
(everything except the cell regenerated, unconditional) are made with the
chosen edit method and judged by the oracle ensemble. Diffusion therefore never
runs inside RL rollouts: the reward is a lookup into this bank.

Stored per series, claim and edit: the ensemble's predicted class and its
probability for the true class.

Stricter check (--crossfit --suf-donors other): every recording is judged by an
oracle that was not trained on it (train_oracles_xf.py), and the sufficiency
edits copy content of a class other than the recording's. --extra-judges also
judges the same edits with the full oracle and the CNN (<part>_full.npz,
<part>_judge2.npz).

Usage:
    python scripts/build_bank.py --dataset synth_planted --part train --n 1680
    python scripts/build_bank.py --dataset synth_planted --part test --n 600
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import load_dataset  # noqa: E402
from veritas.data.base import stratified_subsample  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import InpaintDiffusion  # noqa: E402
from veritas.verifier.edits import edit  # noqa: E402

ORACLES = ("multirocket_hydra", "quant")


class Ensemble:
    """Probability average of the frozen oracles fitted on the training split."""

    def __init__(self, ddir: Path, members=ORACLES):
        self.members = [pickle.loads((ddir / f"oracle_{m}.pkl").read_bytes()) for m in members]

    def predict_proba(self, X: np.ndarray, pos: np.ndarray | None = None) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


class CrossFit:
    """Fold oracles of train_oracles_xf.py: the recording at bank position ``pos``
    is judged by the oracle of its fold, which never saw it."""

    def __init__(self, ddir: Path, part: str, rows: np.ndarray):
        z = np.load(ddir / "xf" / "folds.npz")
        self.members = [pickle.loads((ddir / "xf" / f"oracle_fold{k}.pkl").read_bytes()) for k in range(int(z["F"]))]
        self.fold = (z["train_fold"] if part == "train" else z["test_fold"])[rows]

    def predict_proba(self, X: np.ndarray, pos: np.ndarray) -> np.ndarray:
        f = self.fold[pos]
        out = None
        for k in np.unique(f):
            sel = f == k
            P = self.members[k].predict_proba(X[sel])
            out = np.zeros((len(X), P.shape[1])) if out is None else out
            out[sel] = P
        return out


def bank_claims(grid: EvidenceGrid, max_groups: int):
    """Claims of one window and up to ``max_groups`` channel groups. With K=8 and
    max_groups=1 these are single cells; with K=1 they are whole sensors (or
    pairs of sensors) over the full series."""
    return [c for c in grid.claims() if len(c.windows) == 1 and len(c.groups) <= max_groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--part", choices=["train", "test"], required=True)
    ap.add_argument("--split", default="standard")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--m", type=int, default=2, help="edits per claim and test")
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=40, help="series per edit batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=8, help="time windows; K=1 gives whole-series claims")
    ap.add_argument("--max-groups", type=int, default=1)
    ap.add_argument("--tag", default="", help="bank directory suffix, e.g. _group")
    ap.add_argument("--oracles", nargs="+", default=list(ORACLES),
                    help="judges to average, e.g. cnn for the independent second judge")
    ap.add_argument("--diffusion-name", default="prior+learned", help="diffusion checkpoint name")
    ap.add_argument("--out-name", default=None, help="bank file name (default: <part>.npz)")
    ap.add_argument("--crossfit", action="store_true", help="judge with the cross-fitted fold oracles")
    ap.add_argument("--suf-donors", choices=["any", "other"], default="any",
                    help="class of the content copied in the sufficiency edits")
    ap.add_argument("--extra-judges", nargs="*", default=[], choices=["full", "cnn"],
                    help="also judge the same edits with the full oracle and/or the CNN")
    ap.add_argument("--only-class", type=int, default=None, help="keep the recordings of one class")
    args = ap.parse_args()

    ddir = ROOT / "results" / "diffusion" / args.dataset / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(args.dataset, split=args.split, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    X, y = (Xtr, data.y_train) if args.part == "train" else (Xte, data.y_test)
    rows = np.arange(len(y))
    if args.only_class is not None:  # e.g. every test recording of one diagnosis
        rows = rows[y == args.only_class]
    if args.n < len(rows):
        rows = stratified_subsample(rows, y, args.n, args.seed)

    grid = EvidenceGrid(T=X.shape[-1], channel_groups=data.meta.channel_groups, K=args.K,
                        max_groups=args.max_groups)
    claims = bank_claims(grid, args.max_groups)
    masks = np.stack([grid.mask(c) for c in claims])
    L, m, n_cls = len(claims), args.m, len(data.meta.class_names)
    out_name = args.out_name or f"{args.part}.npz"
    judges = {out_name: CrossFit(ddir, args.part, rows) if args.crossfit else Ensemble(ddir, args.oracles)}
    if "full" in args.extra_judges:
        judges[f"{args.part}_full.npz"] = Ensemble(ddir)
    if "cnn" in args.extra_judges:
        judges[f"{args.part}_judge2.npz"] = Ensemble(ddir, ["cnn"])
    diff = (InpaintDiffusion.load(ddir / f"{args.diffusion_name}.pt")
            if args.method in ("diffusion", "hybrid") else None)
    rng = np.random.default_rng(args.seed)

    pos_all = np.arange(len(rows))
    p0 = {name: j.predict_proba(X[rows], pos_all) for name, j in judges.items()}
    shape = (len(rows), L, m)
    outs = {name: {k: np.zeros(shape, dtype=np.float32 if k.endswith("ptrue") else np.int16)
                   for k in ("nec_pred", "nec_ptrue", "suf_pred", "suf_ptrue")} for name in judges}
    print(f"{args.dataset}/{args.part}: {len(rows)} series x {L} claims x {m} edits x 2 tests "
          f"= {len(rows) * L * m * 2} edits", flush=True)
    t_start = time.perf_counter()
    for s in range(0, len(rows), args.chunk):
        r = rows[s:s + args.chunk]
        B = len(r)
        job_series = np.repeat(np.arange(B), L * m)
        job_claim = np.tile(np.repeat(np.arange(L), m), B)
        Xj, yj, Mj = X[r][job_series], y[r][job_series], masks[job_claim]
        target = (yj + rng.integers(1, n_cls, size=len(yj))) % n_cls
        common = dict(pool=(Xtr, data.y_train), diffusion=diff, n_steps=args.steps,
                      strength=args.strength, batch_size=256)
        X_nec = edit(args.method, Xj, Mj, target, rng, guidance=args.guidance, seed=args.seed + s, **common)
        donor = (yj + rng.integers(1, n_cls, size=len(yj))) % n_cls if args.suf_donors == "other" else None
        X_suf = edit(args.method, Xj, ~Mj, None, rng, guidance=1.0, seed=args.seed + s + 7, donor=donor, **common)
        for jname, judge in judges.items():
            for name, Xe in (("nec", X_nec), ("suf", X_suf)):
                P = judge.predict_proba(Xe, s + job_series)
                outs[jname][f"{name}_pred"][s:s + B] = P.argmax(1).reshape(B, L, m)
                outs[jname][f"{name}_ptrue"][s:s + B] = P[np.arange(len(yj)), yj].reshape(B, L, m)
        done = s + B
        el = time.perf_counter() - t_start
        print(f"  {done}/{len(rows)} series  {el / 60:.1f} min  (eta {el / done * (len(rows) - done) / 60:.1f} min)",
              flush=True)

    dest = ROOT / "results" / "bank" / f"{args.dataset}{args.tag}"
    dest.mkdir(parents=True, exist_ok=True)
    names = {f"{args.part}_full.npz": "+".join(ORACLES), f"{args.part}_judge2.npz": "cnn"}
    names[out_name] = "crossfit" if args.crossfit else "+".join(args.oracles)
    for jname in judges:
        np.savez_compressed(dest / jname, rows=rows, y=y[rows], p0=p0[jname], oracles=np.array([names[jname]]),
                            claim_windows=np.array([c.windows[0] for c in claims]),
                            claim_groups=np.array(["+".join(c.groups) for c in claims]),
                            method=args.method, m=m, K=args.K, suf_donors=args.suf_donors, **outs[jname])
        print(f"-> {dest / jname}")
    print(f"({(time.perf_counter() - t_start) / 60:.1f} min, "
          f"{(time.perf_counter() - t_start) / len(rows):.2f} s per recording)")


if __name__ == "__main__":
    main()
