"""Abstention under distribution shift (proposal RQ4, second half).

The abstention threshold is calibrated with Learn-then-Test on the in-distribution
test set (alpha, delta) and then applied unchanged to the shifted test set, where
the guarantee no longer formally holds. Reported per run and score: accuracy,
coverage and error among accepted answers, in distribution and under shift.

Usage:
    python scripts/shift_eval.py --dataset synth_planted --runs sft_diverse grpo_verified_div
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.selective import ltt_threshold  # noqa: E402


def scores(name: str, run: str):
    bank = Bank(ROOT / "results" / "bank" / name / "test.npz")
    P = json.loads((ROOT / "results" / "agent" / name / f"pred_{run}.json").read_text())["preds"]
    i = np.array([p["bank_i"] for p in P])
    y = np.array([p["label_true"] for p in P])
    yh = np.array([-1 if p["label"] is None else p["label"] for p in P])
    margin = np.array([bank.margin(ii, bank.claim_id(p["window"], p["group"]), None if h < 0 else int(h))
                       for ii, p, h in zip(i, P, yh)])
    conf = np.array([0.0 if h < 0 else bank.p0[ii, h] for ii, h in zip(i, yh)])
    oracle_acc = float((bank.p0.argmax(1) == bank.y).mean())
    # baseline: answer whenever the oracle agrees with the agent (condition 1 of the check alone)
    agree = (bank.p0[i].argmax(1) == yh).astype(float)
    return {"margin": margin, "oracle": conf, "agree": agree}, yh != y, oracle_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shift", default="shift")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()
    shifted = f"{args.dataset}_{args.shift}"
    res = {}
    lines = [f"# Abstention under shift: calibrated on {args.dataset}, applied to {shifted} "
             f"(alpha {args.alpha}, delta {args.delta})", "",
             "| Run | Score | Acc (in) | Cov (in) | Risk (in) | Acc (shift) | Cov (shift) | Risk (shift) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in args.runs:
        s_in, w_in, o_in = scores(args.dataset, run)
        s_sh, w_sh, o_sh = scores(shifted, run)
        res[run] = {"oracle_acc_in": o_in, "oracle_acc_shift": o_sh}
        for k in ("margin", "oracle", "agree"):
            lam = ltt_threshold(s_in[k], w_in, args.alpha, args.delta)
            acc_in = np.zeros_like(w_in) if lam is None else s_in[k] >= lam
            acc_sh = np.zeros_like(w_sh) if lam is None else s_sh[k] >= lam
            r = {"threshold": lam, "acc_in": float(1 - w_in.mean()), "acc_shift": float(1 - w_sh.mean()),
                 "cov_in": float(acc_in.mean()), "risk_in": float(w_in[acc_in].mean()) if acc_in.any() else 0.0,
                 "cov_shift": float(acc_sh.mean()), "risk_shift": float(w_sh[acc_sh].mean()) if acc_sh.any() else 0.0}
            res[run][k] = r
            lines.append(f"| {run} | {k} | {r['acc_in']:.3f} | {r['cov_in']:.3f} | {r['risk_in']:.3f} | "
                         f"{r['acc_shift']:.3f} | {r['cov_shift']:.3f} | {r['risk_shift']:.3f} |")
    lines.append("")
    lines.append(f"Oracle accuracy: in distribution {o_in:.3f}, under shift {o_sh:.3f}")
    out = ROOT / "results" / "agent" / shifted
    (out / "shift_summary.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "shift_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
