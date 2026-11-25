"""Evaluate the review-round-5 experiments (missing runs are skipped, so it can be re-run as they finish).

  judges_agree   real data: do the oracle and the independent judge reach the same verdict on the same
                 claim? Cohen's kappa for realistic edits and for zero-masking edits
  masking        agents trained with a zero-masking check (UCI-HAR, PTB-XL), scored with the realistic check
  swap           agents trained on the CNN judge's verdicts, scored by the held-out oracle (and true evidence)
  no_neighbours  agents without the neighbour classes: non-LLM on all datasets, LLM on the synthetic data

Writes results/review5_eval.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402

A, B = ROOT / "results" / "agent", ROOT / "results" / "bank"


def preds(ds, run):
    f = A / ds / f"pred_{run}.json"
    return sorted(json.loads(f.read_text())["preds"], key=lambda p: p["bank_i"]) if f.exists() else None


def checked(bank, P):
    """Per answer: correct and the claim passes for the true class."""
    return np.array([p["label"] == p["label_true"] and p["group"] is not None and
                     bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"]) for p in P])


def score(ds, run, banks):
    P = preds(ds, run)
    if P is None:
        return None
    ok = np.array([p["label"] == p["label_true"] for p in P])
    out = {"accuracy": float(ok.mean())}
    for name, bank in banks.items():
        out[name] = float(checked(bank, P)[ok].mean())
    return out


def mean_runs(ds, runs, banks):
    res = [r for r in (score(ds, run, banks) for run in runs) if r]
    if not res:
        return None
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]} | {"n_seeds": len(res)}


def kappa(a, b):
    a, b = a.astype(float), b.astype(float)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def judges_agree():
    out = {}
    for ds in ("pamap2_group", "ptbxl_group", "ucihar_subj_group", "natops_group", "cricket_group"):
        files = {"real_oracle": B / ds / "test.npz", "real_judge": B / ds / "test_judge2.npz",
                 "mask_oracle": B / f"{ds}_zero" / "test.npz", "mask_judge": B / f"{ds}_zero" / "test_judge2.npz"}
        if not all(f.exists() for f in files.values()):
            continue
        b = {k: Bank(f) for k, f in files.items()}
        y = b["real_oracle"].y
        # recordings both classifiers get right, so that the verdict depends on the edits only
        ok = (b["real_oracle"].p0.argmax(1) == y) & (b["real_judge"].p0.argmax(1) == y)
        V = {k: np.stack([bk.verified_cells(i, int(y[i])) for i in np.flatnonzero(ok)]).ravel() for k, bk in b.items()}
        r = {"n_recordings": int(ok.sum()),
             "kappa_realistic": kappa(V["real_oracle"], V["real_judge"]),
             "kappa_masked": kappa(V["mask_oracle"], V["mask_judge"]),
             "agree_realistic": float((V["real_oracle"] == V["real_judge"]).mean()),
             "agree_masked": float((V["mask_oracle"] == V["mask_judge"]).mean()),
             "pass_realistic_oracle": float(V["real_oracle"].mean()), "pass_masked_oracle": float(V["mask_oracle"].mean()),
             "same_verdict_real_vs_mask_oracle": float((V["real_oracle"] == V["mask_oracle"]).mean())}
        out[ds] = r
        print(f"judges {ds:18s} kappa realistic {r['kappa_realistic']:.2f}  masked {r['kappa_masked']:.2f}  "
              f"(pass rate {r['pass_realistic_oracle']:.2f} / {r['pass_masked_oracle']:.2f}; "
              f"oracle verdict same for both edits {r['same_verdict_real_vs_mask_oracle']:.2f})")
    return out


def masking():
    out = {}
    for ds in ("ucihar_subj_group", "ptbxl_group"):
        banks = {"checked": Bank(B / ds / "test.npz"), "checked_2nd": Bank(B / ds / "test_judge2.npz")}
        if (B / f"{ds}_zero" / "test.npz").exists():
            banks["checked_masking"] = Bank(B / f"{ds}_zero" / "test.npz")
        seeds = lambda r: [r, f"{r}_s1", f"{r}_s2"]  # noqa: E731
        r = {"accuracy_reward": mean_runs(ds, seeds("grpo_accuracy_ver"), banks),
             "check_reward": mean_runs(ds, seeds("grpo_verified_ver"), banks),
             "masking_check_reward": mean_runs(ds, seeds("grpo_verified_ver_zero"), banks)}
        out[ds] = r
        for k, v in r.items():
            if v:
                print(f"masking {ds:18s} {k:22s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items() if m != "n_seeds")
                      + f"  ({v['n_seeds']} seeds)")
    return out


def true_evidence(P):
    from veritas.data import load_dataset
    data = load_dataset("synth_planted", max_test=2000)
    names = sorted(data.meta.channel_groups)
    rows = np.load(B / "synth_planted" / "test.npz")["rows"]
    w, g = data.groups_test["ev_window"][rows], data.groups_test["ev_group"][rows]
    ok = [p for p in P if p["label"] == p["label_true"]]
    return float(np.mean([p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]] for p in ok]))


def swap():
    out = {}
    for ds, runs in (("synth_planted", ("grpo_accuracy_div", "grpo_verified_div", "grpo_verified_div_cnn")),
                     ("ucihar_subj_group", ("grpo_accuracy_ver", "grpo_verified_ver", "grpo_verified_ver_cnn"))):
        banks = {"oracle": Bank(B / ds / "test.npz"), "judge": Bank(B / ds / "test_judge2.npz")}
        r = {}
        for run in runs:
            s = score(ds, run, banks)
            if s is None:
                continue
            if ds == "synth_planted":
                s["true_evidence"] = true_evidence(preds(ds, run))
            r[run] = s
        if runs[2] in r:  # held-out oracle: judge-trained agent vs accuracy reward, same recordings
            a, b = checked(banks["oracle"], preds(ds, runs[2])), checked(banks["oracle"], preds(ds, runs[0]))
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            r["mcnemar_heldout_oracle_vs_accuracy"] = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        out[ds] = r
        for k, v in r.items():
            print(f"swap {ds:18s} {k:24s} {v if isinstance(v, float) else {m: round(x, 3) for m, x in v.items()}}")
    return out


def no_neighbours():
    out = {}
    for ds in ("synth_planted", "pamap2_group", "ptbxl_group", "ucihar_subj_group", "natops_group", "cricket_group"):
        banks = {"checked": Bank(B / ds / "test.npz")}
        if (B / ds / "test_judge2.npz").exists():
            banks["checked_2nd"] = Bank(B / ds / "test_judge2.npz")
        r = {}
        for claim in ("heuristic", "verified"):
            r[f"policy_{claim}"] = mean_runs(ds, [f"policy_{claim}", f"policy_{claim}_s1", f"policy_{claim}_s2"], banks)
            r[f"policy_{claim}_novote"] = mean_runs(ds, [f"policy_{claim}_novote", f"policy_{claim}_novote_s1",
                                                         f"policy_{claim}_novote_s2"], banks)
        out[ds] = r
        for k, v in r.items():
            if v:
                print(f"novote {ds:18s} {k:26s} acc {v['accuracy']:.3f}  checked {v['checked']:.3f}  ({v['n_seeds']} seeds)")
    # LLM without neighbour classes (synthetic)
    llm = {}
    banks = {"checked": Bank(B / "synth_planted" / "test.npz")}
    for ds, run in (("synth_planted", "grpo_accuracy_div"), ("synth_planted", "grpo_verified_div"),
                    ("synth_planted_nonb", "grpo_accuracy_div"), ("synth_planted_nonb", "grpo_verified_div"),
                    ("synth_planted_nonb", "sft_diverse")):
        s = score(ds, run, banks)
        if s:
            s["true_evidence"] = true_evidence(preds(ds, run))
            llm[f"{ds}/{run}"] = s
            print(f"novote LLM {ds}/{run}: " + "  ".join(f"{m} {x:.3f}" for m, x in s.items()))
    out["llm_synthetic"] = llm
    return out


def main():
    out = {"judges_agree": judges_agree(), "masking": masking(), "swap": swap(), "no_neighbours": no_neighbours()}
    (ROOT / "results" / "review5_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
