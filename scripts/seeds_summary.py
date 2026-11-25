"""Mean and standard deviation over three training seeds for the main comparisons.

Seed 0 is the original run; seeds 1 and 2 carry the suffix _s1 / _s2 (the warm
start is shared, so the spread reflects GRPO / REINFORCE training).

Usage:
    python scripts/seeds_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.data import load_dataset  # noqa: E402

GROUPS = {
    "synth_planted": {"GRPO accuracy reward": "grpo_accuracy_div", "GRPO check reward": "grpo_verified_div",
                      "GRPO masking-check reward": "grpo_verified_div_zero",
                      "GRPO occlusion reward": "grpo_verified_div_occl",
                      "non-LLM, heuristic claim": "policy_heuristic", "non-LLM, check reward": "policy_verified"},
    "pamap2_group": {"GRPO accuracy reward": "grpo_accuracy_ver", "GRPO check reward": "grpo_verified_ver",
                     "non-LLM, heuristic claim": "policy_heuristic", "non-LLM, check reward": "policy_verified"},
    "ptbxl_group": {"GRPO accuracy reward": "grpo_accuracy_ver", "GRPO check reward": "grpo_verified_ver",
                    "non-LLM, check reward": "policy_verified"},
    "ucihar_subj_group": {"GRPO accuracy reward": "grpo_accuracy_ver", "GRPO check reward": "grpo_verified_ver",
                          "non-LLM, check reward": "policy_verified"},
    "natops_group": {"GRPO accuracy reward": "grpo_accuracy_ver", "GRPO check reward": "grpo_verified_ver",
                     "non-LLM, heuristic claim": "policy_heuristic", "non-LLM, check reward": "policy_verified"},
    "cricket_group": {"GRPO accuracy reward": "grpo_accuracy_ver", "GRPO check reward": "grpo_verified_ver",
                      "non-LLM, heuristic claim": "policy_heuristic", "non-LLM, check reward": "policy_verified"},
}


def metrics(ds: str, run: str, bank: Bank, gt) -> dict | None:
    f = ROOT / "results" / "agent" / ds / f"pred_{run}.json"
    if not f.exists():
        return None
    P = json.loads(f.read_text())["preds"]
    ok = np.array([p["label"] == p["label_true"] for p in P])
    ver = np.array([bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"]) for p in P])
    m = {"accuracy": ok.mean(), "verified": ver[ok].mean()}
    if gt is not None:
        w, g, names = gt
        hit = np.array([p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]] for p in P])
        m["true_evidence"] = hit[ok].mean()
    return m


def main():
    out = {}
    for ds, runs in GROUPS.items():
        bank = Bank(ROOT / "results" / "bank" / ds / "test.npz")
        gt = None
        if ds.startswith("synth"):
            d = load_dataset(ds, max_test=2000)
            rows = np.load(ROOT / "results" / "bank" / ds / "test.npz")["rows"]
            gt = (d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows], sorted(d.meta.channel_groups))
        out[ds] = {}
        for label, run in runs.items():
            ms = [m for m in (metrics(ds, run + s, bank, gt) for s in ("", "_s1", "_s2")) if m]
            if not ms:
                continue
            agg = {k: {"mean": float(np.mean([m[k] for m in ms])), "sd": float(np.std([m[k] for m in ms], ddof=1))
                       if len(ms) > 1 else 0.0} for k in ms[0]}
            agg["n_seeds"] = len(ms)
            out[ds][label] = agg
            print(f"{ds:14s} {label:26s} n={len(ms)}  " +
                  "  ".join(f"{k} {v['mean'] * 100:.1f}±{v['sd'] * 100:.1f}" for k, v in agg.items() if k != "n_seeds"))
    (ROOT / "results" / "seeds_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
