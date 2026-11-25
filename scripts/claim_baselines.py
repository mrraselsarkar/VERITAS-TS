"""Floor and ceiling for the checked rate (review C2).

For the recordings that an agent classifies correctly, compare its checked rate with
  random claim : expected checked rate of a claim drawn uniformly from the claim set
  best claim   : share of those recordings that have at least one passing claim
Both use the same bank (training oracle, hybrid edits) as the agents' checked rate.

Usage:
    python scripts/claim_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402

RUNS = {"synth_planted": ("grpo_accuracy_div", "grpo_verified_div"),
        "pamap2_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "ptbxl_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "ucihar_subj_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "natops_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "cricket_group": ("grpo_accuracy_ver", "grpo_verified_ver")}


def main():
    out = {}
    for ds, runs in RUNS.items():
        bank = Bank(ROOT / "results" / "bank" / ds / "test.npz")
        res = {"n_claims": int(len(bank.index))}
        for run in runs:
            P = json.loads((ROOT / "results" / "agent" / ds / f"pred_{run}.json").read_text())["preds"]
            ok = [p for p in P if p["label"] == p["label_true"]]
            agent = np.mean([bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"]) for p in ok])
            cells = [bank.verified_cells(p["bank_i"], p["label_true"]) for p in ok]
            res[run] = {"agent": float(agent), "random": float(np.mean([c.mean() for c in cells])),
                        "best": float(np.mean([c.any() for c in cells]))}
        out[ds] = res
        print(ds, res["n_claims"], "claims",
              {r: {k: round(100 * v, 1) for k, v in res[r].items()} for r in runs})
    (ROOT / "results" / "claim_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
