"""Train the judges for one dataset on its (standardised) training split.

  multirocket_hydra, quant   the main oracle (averaged in the bank)
  cnn                        the independent second judge (veritas.verifier.cnn_judge)

Each is saved as results/diffusion/<dataset>/standard/oracle_<name>.pkl, next to
the diffusion model and its standardisation statistics (stats.npz, written by
train_diffusion.py, which must run first). Existing judges are kept.

Usage:
    python scripts/train_oracles.py --dataset ucihar_subj --members multirocket_hydra quant cnn
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
from veritas.verifier.cnn_judge import CNNJudge  # noqa: E402
from veritas.verifier.oracle import Oracle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--members", nargs="+", default=["multirocket_hydra", "quant", "cnn"])
    ap.add_argument("--max-train", type=int, default=5000)
    args = ap.parse_args()
    ddir = ROOT / "results" / "diffusion" / args.dataset / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(args.dataset, split="standard", max_test=2000, max_train=args.max_train)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    for m in args.members:
        path = ddir / f"oracle_{m}.pkl"
        if path.exists():
            print(f"{m}: exists", flush=True)
            continue
        t0 = time.perf_counter()
        model = CNNJudge().fit(Xtr, data.y_train) if m == "cnn" else Oracle(members=(m,)).fit(Xtr, data.y_train)
        acc = (model.predict_proba(Xte).argmax(1) == data.y_test).mean()
        path.write_bytes(pickle.dumps(model))
        print(f"{m}: test accuracy {acc:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
