"""Ablation views without the neighbour classes (review: is the class decision just the neighbour vote?).

Copies results/agent/<dataset> to results/agent/<dataset>_nonb with the line
"Nearest labelled neighbours ..." removed from every prompt and the neighbour
votes zeroed in features.npz, and copies the bank so that the reward and the
evaluation read the same verdicts as the original experiment.

Usage:
    python scripts/make_nonb_views.py --dataset synth_planted
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    src, dst = ROOT / "results" / "agent" / args.dataset, ROOT / "results" / "agent" / f"{args.dataset}_nonb"
    dst.mkdir(parents=True, exist_ok=True)
    mode = src / "mode.json"
    (dst / "mode.json").write_text(mode.read_text() if mode.exists()
                                   else json.dumps({"source": args.dataset, "claims": "cell"}))
    for part in ("train", "test"):
        lines = (src / f"{part}.jsonl").read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            r = json.loads(line)
            r["prompt"] = "\n".join(s for s in r["prompt"].split("\n") if not s.startswith("Nearest labelled neighbours"))
            out.append(json.dumps(r))
        (dst / f"{part}.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    z = dict(np.load(src / "features.npz"))
    for k in ("train_votes", "test_votes"):
        z[k] = np.zeros_like(z[k])
    np.savez_compressed(dst / "features.npz", **z)
    bsrc, bdst = ROOT / "results" / "bank" / args.dataset, ROOT / "results" / "bank" / f"{args.dataset}_nonb"
    bdst.mkdir(parents=True, exist_ok=True)
    for f in ("train.npz", "test.npz", "test_judge2.npz"):
        if (bsrc / f).exists():
            shutil.copy2(bsrc / f, bdst / f)
    print(f"-> {dst} and {bdst}")


if __name__ == "__main__":
    main()
