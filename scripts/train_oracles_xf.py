"""Cross-fitted oracles: no recording is judged by an oracle that was trained on it.

The training split is divided into F stratified folds. Oracle k (MultiRocket-Hydra + QUANT, like the
main oracle) is trained on the other folds, at most 5000 recordings. build_bank.py --crossfit judges a
training recording of fold k with oracle k, and each test recording with the oracle of a fold drawn
once at random, so training and test verdicts come from equally strong oracles.

Saved as results/diffusion/<dataset>/standard/xf/oracle_fold<k>.pkl, with the fold ids in folds.npz.

Usage:
    python scripts/train_oracles_xf.py --dataset natops
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import load_dataset  # noqa: E402
from veritas.data.base import stratified_subsample  # noqa: E402
from veritas.verifier.oracle import Oracle  # noqa: E402

MEMBERS = ("multirocket_hydra", "quant")


def stratified_folds(y: np.ndarray, F: int, seed: int) -> np.ndarray:
    """Fold id per recording, with every class spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    offset = 0
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = (np.arange(len(idx)) + offset) % F
        offset += len(idx)
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ddir = ROOT / "results" / "diffusion" / args.dataset / "standard"
    out = ddir / "xf"
    out.mkdir(exist_ok=True)
    st = np.load(ddir / "stats.npz")
    data = load_dataset(args.dataset, split="standard", max_test=2000)  # the rows build_bank.py indexes
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    F = args.folds
    train_fold = stratified_folds(data.y_train, F, args.seed)
    test_fold = np.random.default_rng(args.seed + 1).integers(F, size=len(data.y_test))
    np.savez(out / "folds.npz", F=F, train_fold=train_fold, test_fold=test_fold)
    n_cls = len(data.meta.class_names)
    log = {}
    for k in range(F):
        path = out / f"oracle_fold{k}.pkl"
        if path.exists():
            print(f"fold {k}: exists", flush=True)
            continue
        idx = np.flatnonzero(train_fold != k)
        if len(idx) > args.max_train:
            idx = stratified_subsample(idx, data.y_train[idx], args.max_train, args.seed)
        assert len(np.unique(data.y_train[idx])) == n_cls, f"fold {k} misses a class"
        t0 = time.perf_counter()
        model = Oracle(members=MEMBERS).fit(Xtr[idx], data.y_train[idx])
        held = train_fold == k
        acc_held = float((model.predict_proba(Xtr[held]).argmax(1) == data.y_train[held]).mean())
        acc_test = float((model.predict_proba(Xte).argmax(1) == data.y_test).mean())
        path.write_bytes(pickle.dumps(model))
        log[k] = {"n_train": int(len(idx)), "acc_held_out_fold": acc_held, "acc_test": acc_test}
        print(f"fold {k}: {len(idx)} recordings, accuracy on its own fold {acc_held:.3f}, "
              f"on the test split {acc_test:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if log:
        (out / "folds.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
