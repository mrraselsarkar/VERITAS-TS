"""Benchmark table, average ranks and significance tests over the six datasets.

Methods (agents): warm start (SFT), LLM with accuracy reward, LLM with check reward, non-LLM with
rule-based claims, non-LLM with check reward. Metrics, computed among correct answers where noted:
  accuracy   share of correct classes
  checked    share of correct answers whose claim passes the check (training oracle)
  checked2   the same under the independent judge (CNN)
Values are means over seeds (three for every trained agent; the warm start is a single model).

Tests (Demsar, 2006): Friedman test over the methods per metric with the Iman-Davenport
correction and the Nemenyi critical distance; Wilcoxon signed-rank test of the LLM check reward
against the LLM accuracy reward over the six datasets; paired bootstrap 95% interval of the
difference in checked rate per dataset (seed 0, 1000 resamples of test recordings).

Usage:
    python scripts/benchmark_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402

DATASETS = [("Synthetic", "synth_planted", "div"), ("PAMAP2", "pamap2_group", "ver"), ("PTB-XL", "ptbxl_group", "ver"),
            ("UCI-HAR", "ucihar_subj_group", "ver"), ("NATOPS", "natops_group", "ver"), ("Cricket", "cricket_group", "ver")]
METHODS = ["Warm start (SFT)", "LLM, accuracy reward", "LLM, check reward", "Non-LLM, rule-based claim",
           "Non-LLM, check reward"]
METRICS = ["accuracy", "checked", "checked2"]
SEEDS = ["", "_s1", "_s2"]


def runs(tag):
    return {"Warm start (SFT)": ["sft_diverse" if tag == "div" else "sft_verified"],
            "LLM, accuracy reward": [f"grpo_accuracy_{tag}{s}" for s in SEEDS],
            "LLM, check reward": [f"grpo_verified_{tag}{s}" for s in SEEDS],
            "Non-LLM, rule-based claim": [f"policy_heuristic{s}" for s in SEEDS],
            "Non-LLM, check reward": [f"policy_verified{s}" for s in SEEDS]}


def per_record(ds, run, b1, b2):
    P = sorted(json.loads((ROOT / "results" / "agent" / ds / f"pred_{run}.json").read_text())["preds"],
               key=lambda p: p["bank_i"])
    ok = np.array([p["label"] == p["label_true"] for p in P])
    v1 = np.array([b1.verified(p["bank_i"], b1.claim_id(p["window"], p["group"]), p["label_true"]) for p in P])
    v2 = np.array([b2.verified(p["bank_i"], b2.claim_id(p["window"], p["group"]), p["label_true"]) for p in P])
    return ok, ok & v1, ok & v2


def main():
    rng = np.random.default_rng(0)
    table, boot = {}, {}
    for name, ds, tag in DATASETS:
        b1 = Bank(ROOT / "results" / "bank" / ds / "test.npz")
        b2 = Bank(ROOT / "results" / "bank" / ds / "test_judge2.npz")
        table[name] = {}
        for m, rs in runs(tag).items():
            vals = []
            for r in rs:
                if not (ROOT / "results" / "agent" / ds / f"pred_{r}.json").exists():
                    continue
                ok, c1, c2 = per_record(ds, r, b1, b2)
                vals.append([ok.mean(), c1.sum() / ok.sum(), c2.sum() / ok.sum()])
            vals = np.array(vals)
            table[name][m] = {k: {"mean": float(vals[:, j].mean()), "sd": float(vals[:, j].std(ddof=1)) if len(vals) > 1 else 0.0,
                                  "n": len(vals)} for j, k in enumerate(METRICS)}
        # paired bootstrap of the checked-rate difference (seed 0): check reward minus accuracy reward
        oka, ca, _ = per_record(ds, f"grpo_accuracy_{tag}", b1, b2)
        okv, cv, _ = per_record(ds, f"grpo_verified_{tag}", b1, b2)
        n, diffs = len(oka), []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            diffs.append(cv[i].sum() / max(okv[i].sum(), 1) - ca[i].sum() / max(oka[i].sum(), 1))
        d0 = cv.sum() / okv.sum() - ca.sum() / oka.sum()
        boot[name] = {"diff": float(d0), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}

    # average ranks (rank 1 = best) and Friedman / Iman-Davenport / Nemenyi per metric
    N, k = len(DATASETS), len(METHODS)
    ranks, tests = {}, {}
    for met in METRICS:
        M = np.array([[table[d][m][met]["mean"] for m in METHODS] for d, _, _ in DATASETS])
        R = np.array([rankdata(-row) for row in M])
        ranks[met] = dict(zip(METHODS, R.mean(0).tolist()))
        chi2, p = friedmanchisquare(*M.T)
        ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
        tests[met] = {"chi2": float(chi2), "p": float(p), "F_ID": float(ff),
                      "p_ID": float(1 - fdist.cdf(ff, k - 1, (k - 1) * (N - 1)))}
    q05 = {5: 2.728}[k]
    cd = q05 * np.sqrt(k * (k + 1) / (6 * N))
    mean_rank = {m: float(np.mean([ranks[met][m] for met in METRICS])) for m in METHODS}

    # Wilcoxon: LLM check reward vs LLM accuracy reward over the six datasets
    wil = {}
    for met in METRICS:
        a = np.array([table[d]["LLM, check reward"][met]["mean"] for d, _, _ in DATASETS])
        b = np.array([table[d]["LLM, accuracy reward"][met]["mean"] for d, _, _ in DATASETS])
        alt = "two-sided" if met == "accuracy" else "greater"
        res = wilcoxon(a, b, alternative=alt)
        wil[met] = {"W": float(res.statistic), "p": float(res.pvalue), "alternative": alt,
                    "wins": int((a > b).sum()), "mean_diff": float((a - b).mean())}

    out = {"table": table, "ranks": ranks, "mean_rank": mean_rank, "friedman": tests,
           "nemenyi_cd": float(cd), "wilcoxon_llm_check_vs_accuracy": wil, "bootstrap_checked_diff_seed0": boot}
    (ROOT / "results" / "benchmark_stats.json").write_text(json.dumps(out, indent=1))
    for d, _, _ in DATASETS:
        print(d, {m: tuple(round(100 * table[d][m][k_]["mean"], 1) for k_ in METRICS) for m in METHODS})
    print("ranks", json.dumps(ranks, indent=0))
    print("mean rank", mean_rank)
    print("friedman", tests, "CD", round(cd, 2))
    print("wilcoxon", wil)
    print("bootstrap", boot)


if __name__ == "__main__":
    main()
