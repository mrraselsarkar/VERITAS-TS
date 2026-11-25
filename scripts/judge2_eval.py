"""Re-check the agents' evidence with an independent judge (reward-hacking test).

The agents were trained against the main oracle (MultiRocket-Hydra + QUANT).
Here the same answers are checked with the same kind of edits but judged by an
independent CNN (bank file test_judge2.npz). If the verification reward had only
taught claims that suit the main oracle, the gain would vanish under this judge.

Usage:
    python scripts/judge2_eval.py --dataset synth_planted --runs grpo_accuracy_div grpo_verified_div
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402


def verified(bank: Bank, P: list[dict]) -> np.ndarray:
    """Per answer: correct and claim verified for the true label."""
    return np.array([p["label"] == p["label_true"] and
                     bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"])
                     for p in P])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    adir = ROOT / "results" / "agent" / args.dataset
    main_bank = Bank(ROOT / "results" / "bank" / args.dataset / "test.npz")
    judge2 = Bank(ROOT / "results" / "bank" / args.dataset / "test_judge2.npz")
    print(f"judge 2 accuracy on the unedited test recordings: {(judge2.p0.argmax(1) == judge2.y).mean():.3f}")
    res, outs = {}, {}
    for run in args.runs:
        P = sorted(json.loads((adir / f"pred_{run}.json").read_text())["preds"], key=lambda p: p["bank_i"])
        correct = np.array([p["label"] == p["label_true"] for p in P])
        v1, v2 = verified(main_bank, P), verified(judge2, P)
        outs[run] = v2
        res[run] = {"accuracy": float(correct.mean()),
                    "verified_main_of_correct": float(v1[correct].mean()),
                    "verified_judge2_of_correct": float(v2[correct].mean())}
        print(f"{run:22s} acc {correct.mean():.3f}  verified (main) {v1[correct].mean():.3f}  "
              f"verified (judge 2) {v2[correct].mean():.3f}")
    if len(args.runs) >= 2:
        a, b = outs[args.runs[-1]], outs[args.runs[0]]
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        res["test_last_vs_first"] = {"diff_all": float(a.mean() - b.mean()), "p_mcnemar": p}
        print(f"{args.runs[-1]} vs {args.runs[0]} under judge 2: {a.mean() - b.mean():+.3f} of all recordings, "
              f"McNemar p = {p:.2g}")
    (adir / "judge2_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
