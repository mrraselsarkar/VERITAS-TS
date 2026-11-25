"""Analyses added after review, computed from existing banks and predictions.

  granularity   which claim type a simple rule picks per dataset: the most specific
                type (cells before sensors) that is verifiable on at least half of
                the TRAINING recordings
  tiers         three-tier output instead of plain abstention: checked answers;
                unchecked answers the oracle agrees with; disagreements
  retries       the number of attempts k chosen by Learn-then-Test on in-distribution
                calibration data, then applied to shifted data
  imitation     does the agent only copy the oracle? agreement with the oracle and
                accuracy where the two disagree

Writes results/review_analysis.json and prints a summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402

A, B = ROOT / "results" / "agent", ROOT / "results" / "bank"


def any_verified(path: Path) -> float:
    b = Bank(path)
    return float(np.mean([b.verified_cells(i, int(b.y[i])).any() for i in range(len(b))]))


def granularity() -> dict:
    out = {}
    for ds, cell, sensor in [("synthetic", "synth_planted", None), ("PAMAP2", "pamap2", "pamap2_group"),
                             ("PTB-XL", None, "ptbxl_group"), ("UCI-HAR", None, "ucihar_subj_group")]:
        r = {}
        for kind, name in (("cell", cell), ("sensor", sensor)):
            if name and (B / name / "train.npz").exists():
                r[kind] = any_verified(B / name / "train.npz")
        pick = next((k for k in ("cell", "sensor") if r.get(k, 0) >= 0.5), max(r, key=r.get) if r else None)
        out[ds] = {"verifiable_share": r, "chosen": pick}
        print(f"granularity {ds:9s} {r} -> {pick}")
    return out


def tiers(name: str, run: str) -> dict | None:
    pf = A / name / f"pred_{run}.json"
    if not pf.exists():
        return None
    bank = Bank(B / name / "test.npz")
    P = json.loads(pf.read_text())["preds"]
    t = {"checked": [], "oracle_agrees": [], "disagree": []}
    for p in P:
        i, lab = p["bank_i"], p["label"]
        wrong = lab != p["label_true"]
        if lab is not None and bank.verified(i, bank.claim_id(p["window"], p["group"]), lab):
            t["checked"].append(wrong)
        elif lab is not None and bank.p0[i].argmax() == lab:
            t["oracle_agrees"].append(wrong)
        else:
            t["disagree"].append(wrong)
    n = len(P)
    res = {k: {"share": len(v) / n, "error": float(np.mean(v)) if v else 0.0} for k, v in t.items()}
    print(f"tiers {name}/{run}: " + "; ".join(f"{k} {v['share']:.3f} (err {v['error']:.3f})" for k, v in res.items()))
    return res


def ltt_k(answers: np.ndarray, wrong: np.ndarray, alpha: float, delta: float) -> int:
    """Largest k (testing k = 1, 2, ... in order) whose error among answered is certified <= alpha."""
    best = 0
    for k in range(answers.shape[1]):
        acc = answers[:, k]
        n, e = int(acc.sum()), int((acc & wrong[:, k]).sum())
        if n and binom.cdf(e, n, alpha) <= delta:
            best = k + 1
        else:
            break
    return best


def retry_arrays(path: Path):
    R = json.loads(path.read_text())
    recs = [json.loads(line) for line in (path.parent / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    y = np.array([r["label"] for r in recs])
    K = R["k"]
    answered = np.zeros((len(y), K), bool)
    wrong = np.zeros((len(y), K), bool)
    for j, att in enumerate(R["attempts"]):
        hit = None
        for k in range(K):
            if hit is None and k < len(att) and att[k]["verified"]:
                hit = att[k]
            answered[j, k] = hit is not None
            wrong[j, k] = hit is not None and hit["label"] != y[j]
    return answered, wrong


def retries(in_name: str, shift_name: str, run: str, alpha: float, delta: float = 0.1) -> dict | None:
    pi, ps = A / in_name / f"retry_{run}.json", A / shift_name / f"retry_{run}.json"
    if not (pi.exists() and ps.exists()):
        return None
    ai, wi = retry_arrays(pi)
    a_s, w_s = retry_arrays(ps)
    k = ltt_k(ai, wi, alpha, delta)
    res = {"k_chosen": k}
    if k:
        res.update(in_coverage=float(ai[:, k - 1].mean()), in_error=float(wi[:, k - 1][ai[:, k - 1]].mean()),
                   shift_coverage=float(a_s[:, k - 1].mean()),
                   shift_error=float(w_s[:, k - 1][a_s[:, k - 1]].mean()) if a_s[:, k - 1].any() else 0.0)
    print(f"retries {in_name}/{run} alpha {alpha}: {res}")
    return res


def imitation(name: str, run: str) -> dict | None:
    pf = A / name / f"pred_{run}.json"
    if not pf.exists():
        return None
    bank = Bank(B / name / "test.npz")
    P = json.loads(pf.read_text())["preds"]
    o = np.array([bank.p0[p["bank_i"]].argmax() for p in P])
    lab = np.array([-1 if p["label"] is None else p["label"] for p in P])
    y = np.array([p["label_true"] for p in P])
    agree = lab == o
    res = {"agrees_with_oracle": float(agree.mean()), "oracle_accuracy": float((o == y).mean()),
           "agent_accuracy": float((lab == y).mean()),
           "agent_right_where_oracle_wrong": float((lab == y)[o != y].mean()) if (o != y).any() else 0.0}
    print(f"imitation {name}/{run}: {res}")
    return res


def main():
    out = {"granularity": granularity(), "tiers": {}, "retries": {}, "imitation": {}}
    for name, run in [("synth_planted", "grpo_verified_div"), ("pamap2_group", "grpo_verified_ver"),
                      ("ptbxl_group", "grpo_verified_ver"), ("ucihar_subj_group", "grpo_verified_ver"),
                      ("synth_planted", "policy_verified"), ("pamap2_group", "policy_verified")]:
        out["tiers"][f"{name}/{run}"] = tiers(name, run)
        out["imitation"][f"{name}/{run}"] = imitation(name, run)
    for in_name, sh, run, alpha in [("synth_planted", "synth_planted_shift", "grpo_verified_div", 0.05),
                                    ("synth_planted", "synth_planted_shift", "grpo_accuracy_div", 0.05),
                                    ("ucihar_subj_group", "ucihar_subj_group_shift", "grpo_verified_ver", 0.1)]:
        out["retries"][f"{in_name}/{run}"] = retries(in_name, sh, run, alpha)
    (ROOT / "results" / "review_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
