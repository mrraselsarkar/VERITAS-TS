"""Paired significance tests between agent runs on the same test series.

For a binary outcome per series (correct, verified evidence, planted hit), two
runs are compared with an exact McNemar test on the discordant pairs, and a
95% paired bootstrap interval for the difference in rates.

Usage:
    python scripts/agent_stats.py --dataset synth_planted \
        --pairs grpo_verified_div:grpo_accuracy_div policy_verified:grpo_verified_div
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
from veritas.data import load_dataset  # noqa: E402


def outcomes(dataset: str, run: str, bank: Bank, gt) -> dict[str, np.ndarray]:
    P = json.loads((ROOT / "results" / "agent" / dataset / f"pred_{run}.json").read_text())["preds"]
    P = sorted(P, key=lambda p: p["bank_i"])
    y = np.array([p["label_true"] for p in P])
    correct = np.array([p["label"] == p["label_true"] for p in P])
    ver = np.array([bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), int(yy))
                    for p, yy in zip(P, y)]) & correct
    out = {"correct": correct, "verified": ver}
    if gt is not None:
        w, g, names = gt
        out["planted"] = correct & np.array([p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]]
                                             for p in P])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="runA:runB")
    args = ap.parse_args()
    bank = Bank(ROOT / "results" / "bank" / args.dataset / "test.npz")
    gt = None
    if args.dataset.startswith("synth"):
        d = load_dataset(args.dataset, max_test=2000)
        rows = np.load(ROOT / "results" / "bank" / args.dataset / "test.npz")["rows"]
        gt = (d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows], sorted(d.meta.channel_groups))
    rng = np.random.default_rng(0)
    res = {}
    for pair in args.pairs:
        a, b = pair.split(":")
        A, B = outcomes(args.dataset, a, bank, gt), outcomes(args.dataset, b, bank, gt)
        for k in A:
            x, z = A[k].astype(int), B[k].astype(int)
            n01, n10 = int(((x == 1) & (z == 0)).sum()), int(((x == 0) & (z == 1)).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            idx = rng.integers(0, len(x), size=(2000, len(x)))
            diff = (x[idx].mean(1) - z[idx].mean(1))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            res[f"{a} vs {b} [{k}]"] = {"a": x.mean(), "b": z.mean(), "diff": x.mean() - z.mean(),
                                        "ci95": [lo, hi], "p_mcnemar": p}
            print(f"{a} vs {b} [{k}]: {x.mean():.3f} vs {z.mean():.3f}  diff {x.mean() - z.mean():+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  McNemar p = {p:.2g}")
    out = ROOT / "results" / "agent" / args.dataset / "stats.json"
    out.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
