"""Score every agent run on a dataset and write results/agent/<dataset>/summary.{json,md}.

Per run:
  accuracy, macro-F1, well-formed answers
  verified-evidence rate  share of correct predictions whose cited cell passes
                          necessity and sufficiency (proposal's main metric)
  planted-cell hit        (synthetic only) share of correct predictions that
                          cite the true planted cell
  selective               Learn-then-Test gating at error level alpha: coverage,
                          selective risk and how often risk <= alpha, for two
                          scores: the verification margin and the neighbour-vote share

Usage:
    python scripts/analyze_agent.py --dataset synth_planted --alpha 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.selective import evaluate  # noqa: E402
from veritas.data import load_dataset  # noqa: E402

RUNS = ["base", "sft", "grpo_verified_t1", "sft_diverse", "grpo_accuracy_div", "grpo_verified_div",
        "sft_verified", "grpo_accuracy_ver", "grpo_verified_ver", "policy_heuristic", "policy_verified"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()
    adir = ROOT / "results" / "agent" / args.dataset
    bank = Bank(ROOT / "results" / "bank" / args.dataset / "test.npz")
    votes = np.load(adir / "features.npz")["test_votes"]
    names = None
    gt = None
    if args.dataset.startswith("synth"):
        data = load_dataset(args.dataset, max_test=2000)
        names = sorted(data.meta.channel_groups)
        rows = np.load(ROOT / "results" / "bank" / args.dataset / "test.npz")["rows"]
        gt = (data.groups_test["ev_window"][rows], data.groups_test["ev_group"][rows])

    # ceiling: share of series with at least one verified cell for the true label
    any_ver = np.mean([bank.verified_cells(i, int(bank.y[i])).any() for i in range(len(bank))])
    summary = {"dataset": args.dataset, "alpha": args.alpha, "delta": args.delta,
               "n_test": len(bank), "any_cell_verified": float(any_ver), "runs": {}}
    for run in RUNS:
        path = adir / f"pred_{run}.json"
        if not path.exists():
            continue
        P = json.loads(path.read_text())["preds"]
        i = np.array([p["bank_i"] for p in P])
        y = np.array([p["label_true"] for p in P])
        yhat = np.array([-1 if p["label"] is None else p["label"] for p in P])
        correct = yhat == y
        cid = [bank.claim_id(p["window"], p["group"]) for p in P]
        ver = np.array([bank.verified(ii, j, yy) for ii, j, yy in zip(i, cid, y)])
        margin = np.array([bank.margin(ii, j, None if yh < 0 else int(yh)) for ii, j, yh in zip(i, cid, yhat)])
        vote_share = np.array([0.0 if yh < 0 else votes[ii, yh] / votes[ii].sum() for ii, yh in zip(i, yhat)])
        # baseline gate: the oracle's own probability for the agent's label (agreement without evidence)
        oracle_conf = np.array([0.0 if yh < 0 else bank.p0[ii, yh] for ii, yh in zip(i, yhat)])
        res = {
            "accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(y, yhat, average="macro")),
            "well_formed": float(np.mean([p["label"] is not None and p["window"] is not None
                                          and p["group"] is not None for p in P])),
            "verified_rate": float(ver[correct].mean()) if correct.any() else 0.0,
            "verified_all": float((ver & correct).mean()),
        }
        if gt is not None:
            hit = np.array([p["window"] == gt[0][ii] and p["group"] == names[gt[1][ii]] for p, ii in zip(P, i)])
            res["planted_hit"] = float(hit[correct].mean()) if correct.any() else 0.0
        res["selective_verification"] = evaluate(margin, ~correct, args.alpha, args.delta)
        res["selective_votes"] = evaluate(vote_share, ~correct, args.alpha, args.delta)
        res["selective_oracle"] = evaluate(oracle_conf, ~correct, args.alpha, args.delta)
        # evidence on top of agreement: accept only oracle-agreeing answers, ranked by verification
        agree = np.array([yh >= 0 and bank.p0[ii].argmax() == yh for ii, yh in zip(i, yhat)])
        res["agree_coverage"] = float(agree.mean())
        res["agree_error"] = float((~correct)[agree].mean()) if agree.any() else 0.0
        res["verified_coverage"] = float((margin >= 1).mean())
        res["verified_error"] = float((~correct)[margin >= 1].mean()) if (margin >= 1).any() else 0.0
        summary["runs"][run] = res

    (adir / "summary.json").write_text(json.dumps(summary, indent=2))
    hdr = ["Run", "Acc", "F1", "Format", "Verified", "Planted hit",
           "Cov (verif)", "Risk (verif)", "Cov (oracle)", "Risk (oracle)", "Cov (votes)", "Risk (votes)"]
    lines = [f"# {args.dataset}: agent results (n = {len(bank)}, alpha = {args.alpha})",
             "", f"Series with at least one verified cell: {any_ver:.3f}", "",
             "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)]
    for run, r in summary["runs"].items():
        sv, so, sk = r["selective_verification"], r["selective_votes"], r["selective_oracle"]
        lines.append(f"| {run} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['well_formed']:.2f} | "
                     f"{r['verified_rate']:.3f} | {r.get('planted_hit', float('nan')):.3f} | "
                     f"{sv['coverage']:.3f} | {sv['selective_risk']:.3f} | {sk['coverage']:.3f} | "
                     f"{sk['selective_risk']:.3f} | {so['coverage']:.3f} | {so['selective_risk']:.3f} |")
    (adir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
