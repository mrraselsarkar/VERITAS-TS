"""Bank for the occlusion-reward ablation (review: "reward agreement with an attribution method").

From the zero-masking bank (which stores the oracle's probability of the true class after a
cell is set to zero), the occlusion attribution of every cell is the drop of that probability.
The occlusion bank marks exactly one claim per recording as passing, the cell with the largest
drop (the oracle must still predict the true class on the original recording, as in the check).
It has the same format as the other banks, so GRPO uses it through --bank-tag _occl.

Usage:
    python scripts/make_occlusion_bank.py --dataset synth_planted
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synth_planted")
    args = ap.parse_args()
    for part in ("train", "test"):
        z = dict(np.load(ROOT / "results" / "bank" / f"{args.dataset}_zero" / f"{part}.npz"))
        y = z["y"]
        drop = z["p0"][np.arange(len(y)), y][:, None] - z["nec_ptrue"].mean(-1)   # (n, claims)
        top = drop.argmax(1)
        n, L, m = z["nec_pred"].shape
        other = (y + 1) % z["p0"].shape[1]
        nec = np.repeat(y[:, None, None], L, 1).repeat(m, 2).astype(np.int16)
        nec[np.arange(n), top] = other[:, None]          # only the top cell "changes the class"
        suf = np.repeat(y[:, None, None], L, 1).repeat(m, 2).astype(np.int16)
        z.update(nec_pred=nec, suf_pred=suf, method=np.array("occlusion"))
        dest = ROOT / "results" / "bank" / f"{args.dataset}_occl"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / f"{part}.npz", **z)
        np.save(dest / f"{part}_top.npy", top)
        print(f"{part}: {n} recordings, occlusion top cell stored -> {dest}")


if __name__ == "__main__":
    main()
