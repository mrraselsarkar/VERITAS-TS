"""Evaluate the review-round-6 runs (missing runs are skipped, so it can be re-run as they finish).

  stricter        every agent scored with the stricter check (cross-fitted oracle, other-class content in
                  Test 2), with the same edits judged by the full oracle and by the CNN, and with a strict
                  pass (both edits of both tests succeed); LLM retrained with the stricter check
  subject         subject-disjoint test sets (new people) of UCI-HAR and PAMAP2, three seeds
  swap            oracle and judge swapped, three seeds
  no_neighbours   LLM without the neighbour classes, synthetic and UCI-HAR, three seeds
  mi              PTB-XL, all test recordings of myocardial infarction: anterior vs inferior location
  verifier_ci     Table III: AUC per edit, mean over three seeds with a bootstrap interval

Writes results/review6_eval.json.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from veritas.agent.reward import Bank  # noqa: E402
from review5_eval import checked, preds, true_evidence  # noqa: E402

A, B = ROOT / "results" / "agent", ROOT / "results" / "bank"
DATASETS = {"synth_planted": "div", "pamap2_group": "ver", "ptbxl_group": "ver", "ucihar_subj_group": "ver",
            "natops_group": "ver", "cricket_group": "ver"}
SEEDS = ("", "_s1", "_s2")


def strict(bank, P):
    """Correct, and every edit of both tests succeeds (instead of at least one of the two)."""
    out = []
    for p in P:
        j = bank.claim_id(p["window"], p["group"])
        y, i = p["label_true"], p["bank_i"]
        ok = p["label"] == y and j is not None and bank.p0[i].argmax() == y
        out.append(bool(ok and (bank.nec_pred[i, j] != y).all() and (bank.suf_pred[i, j] == y).all()))
    return np.array(out)


def score(ds, run, banks, strict_bank=None):
    P = preds(ds, run)
    if P is None:
        return None
    ok = np.array([p["label"] == p["label_true"] for p in P])
    out = {"accuracy": float(ok.mean())}
    for name, bank in banks.items():
        out[name] = float(checked(bank, P)[ok].mean())
    if strict_bank is not None:
        out["strict"] = float(strict(strict_bank, P)[ok].mean())
    if ds == "synth_planted":
        out["true_evidence"] = true_evidence(P)
    return out


def mean_sd(ds, runs, banks, strict_bank=None):
    res = [r for r in (score(ds, run, banks, strict_bank) for run in runs) if r]
    if not res:
        return None
    return ({k: float(np.mean([r[k] for r in res])) for k in res[0]}
            | {f"{k}_sd": float(np.std([r[k] for r in res])) for k in res[0]} | {"n_seeds": len(res)})


def passing_rates(bank):
    ok = bank.p0.argmax(1) == bank.y
    anyp = np.array([bank.verified_cells(i, int(bank.y[i])).any() for i in range(len(bank))])
    return {"oracle_acc": float(ok.mean()), "any_pass_oracle_correct": float(anyp[ok].mean())}


def stricter():
    out = {}
    for ds, r in DATASETS.items():
        X = B / f"{ds}_xf"
        if not (X / "test.npz").exists():
            continue
        xf = Bank(X / "test.npz")
        banks = {"checked": Bank(B / ds / "test.npz"), "checked_2nd": Bank(B / ds / "test_judge2.npz"),
                 "xf": xf, "xf_full": Bank(X / "test_full.npz"), "xf_2nd": Bank(X / "test_judge2.npz")}
        runs = {"llm_accuracy": [f"grpo_accuracy_{r}{s}" for s in SEEDS],
                "llm_check": [f"grpo_verified_{r}{s}" for s in SEEDS],
                "llm_check_xf": [f"grpo_verified_{r}_xf{s}" for s in SEEDS],
                "nonllm_rule": [f"policy_heuristic{s}" for s in SEEDS],
                "nonllm_check": [f"policy_verified{s}" for s in SEEDS],
                "nonllm_check_xf": [f"policy_verified_xf{s}" for s in SEEDS]}
        res = {k: mean_sd(ds, v, banks, xf) for k, v in runs.items()}
        res["passing"] = {"original_train": passing_rates(Bank(B / ds / "train.npz")),
                          "original_test": passing_rates(banks["checked"]),
                          "xf_test": passing_rates(xf)}
        if (X / "train.npz").exists():
            res["passing"]["xf_train"] = passing_rates(Bank(X / "train.npz"))
            res["passing"]["xf_train_full"] = passing_rates(Bank(X / "train_full.npz"))
        # McNemar, seed 0: check reward (retrained with the stricter check) vs accuracy reward, stricter check
        for k in ("llm_check_xf", "llm_check"):
            a, b = preds(ds, runs[k][0]), preds(ds, runs["llm_accuracy"][0])
            if a and b:
                ca, cb = checked(xf, a), checked(xf, b)
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                res[f"mcnemar_{k}_vs_accuracy"] = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        out[ds] = res
        for k, v in res.items():
            if isinstance(v, dict) and "accuracy" in v:
                print(f"stricter {ds:18s} {k:16s} acc {v['accuracy']:.3f}  orig {v['checked']:.3f}  xf {v['xf']:.3f}  "
                      f"xf-full {v['xf_full']:.3f}  xf-2nd {v['xf_2nd']:.3f}  strict {v['strict']:.3f}"
                      + (f"  true {v['true_evidence']:.3f}" if "true_evidence" in v else "") + f"  ({v['n_seeds']})")
        print(f"stricter {ds:18s} passing: " + "  ".join(f"{k} {v['any_pass_oracle_correct']:.3f} (acc {v['oracle_acc']:.3f})"
                                                      for k, v in res["passing"].items()))
    return out


def votes_share(name, P):
    z = np.load(A / name / "features.npz")
    v = z["test_votes"].argmax(1)
    return float(np.mean([p["label"] == v[p["bank_i"]] for p in P]))


def subject():
    out = {}
    for N in ("ucihar_subj_group", "pamap2_subj_group"):
        for name in (N, f"{N}_shift"):
            banks = {"checked": Bank(B / name / "test.npz")}
            if (B / name / "test_judge2.npz").exists():
                banks["checked_2nd"] = Bank(B / name / "test_judge2.npz")
            res = {}
            for k, runs in (("llm_accuracy", [f"grpo_accuracy_ver{s}" for s in SEEDS]),
                            ("llm_check", [f"grpo_verified_ver{s}" for s in SEEDS]),
                            ("nonllm_rule", [f"policy_heuristic{s}" for s in SEEDS]),
                            ("nonllm_check", [f"policy_verified{s}" for s in SEEDS])):
                r = mean_sd(name, runs, banks)
                if r:
                    r["follows_votes"] = float(np.mean([votes_share(name, preds(name, run)) for run in runs
                                                        if preds(name, run)]))
                res[k] = r
            out[name] = res
            for k, v in res.items():
                if v:
                    print(f"subject {name:24s} {k:13s} acc {v['accuracy']:.3f}  checked {v['checked']:.3f}"
                          + (f"  2nd {v['checked_2nd']:.3f}" if "checked_2nd" in v else "")
                          + f"  follows votes {v['follows_votes']:.3f}  ({v['n_seeds']})")
    return out


def swap():
    out = {}
    for ds, r, extra in (("synth_planted", "div", "_div_cnn"), ("ucihar_subj_group", "ver", "_ver_cnn")):
        banks = {"oracle": Bank(B / ds / "test.npz"), "judge": Bank(B / ds / "test_judge2.npz")}
        res = {"accuracy_reward": mean_sd(ds, [f"grpo_accuracy_{r}{s}" for s in SEEDS], banks),
               "oracle_reward": mean_sd(ds, [f"grpo_verified_{r}{s}" for s in SEEDS], banks),
               "judge_reward": mean_sd(ds, [f"grpo_verified{extra}{s}" for s in SEEDS], banks)}
        pv = []
        for s in SEEDS:
            a, b = preds(ds, f"grpo_verified{extra}{s}"), preds(ds, f"grpo_accuracy_{r}{s}")
            if a and b:
                ca, cb = checked(banks["oracle"], a), checked(banks["oracle"], b)
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                pv.append(binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
        res["mcnemar_heldout_oracle_per_seed"] = pv
        out[ds] = res
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"swap {ds:18s} {k:16s} acc {v['accuracy']:.3f}  oracle {v['oracle']:.3f}+-{v['oracle_sd']:.3f}  "
                      f"judge {v['judge']:.3f}" + (f"  true {v['true_evidence']:.3f}" if "true_evidence" in v else "")
                      + f"  ({v['n_seeds']})")
        print(f"swap {ds}: McNemar p per seed {['%.2g' % p for p in pv]}")
    return out


def no_neighbours():
    out = {}
    for base, nonb, r in (("synth_planted", "synth_planted_nonb", "div"), ("ucihar_subj_group", "ucihar_subj_group_nonb", "ver")):
        banks = {"checked": Bank(B / base / "test.npz"), "checked_2nd": Bank(B / base / "test_judge2.npz")}
        res = {}
        for name in (base, nonb):
            for rew in ("accuracy", "verified"):
                v = mean_sd(name, [f"grpo_{rew}_{r}{s}" for s in SEEDS], banks)
                if v and name.endswith("_nonb") and base == "synth_planted":
                    v["true_evidence"] = float(np.mean([true_evidence(preds(name, f"grpo_{rew}_{r}{s}"))
                                                        for s in SEEDS if preds(name, f"grpo_{rew}_{r}{s}")]))
                res[f"{name}/{rew}"] = v
        out[base] = res
        for k, v in res.items():
            if v:
                print(f"no-neighbours {k:36s} acc {v['accuracy']:.3f}  checked {v['checked']:.3f}  2nd {v['checked_2nd']:.3f}"
                      + (f"  true {v['true_evidence']:.3f}" if "true_evidence" in v else "") + f"  ({v['n_seeds']})")
    return out


def mi():
    import pandas as pd
    from veritas.data import load_dataset
    from veritas.data.base import RAW_DIR
    N = "ptbxl_group_mi"
    if not (B / N / "test.npz").exists():
        return None
    raw = RAW_DIR / "ptbxl"
    db = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    codes = db.scp_codes.apply(ast.literal_eval)
    sup = codes.apply(lambda c: sorted({diag.loc[k, "diagnostic_class"] for k in c if k in diag.index}))
    db = db[sup.str.len() == 1]
    sub = codes.loc[db.index].apply(lambda c: frozenset(diag.loc[k, "diagnostic_subclass"] for k in c
                                                         if k in diag.index and diag.loc[k, "diagnostic_class"] == "MI"))
    data = load_dataset("ptbxl", max_test=2000)
    z = np.load(B / N / "test.npz")
    sub_bank = sub.to_numpy()[data.test_rows[z["rows"]]]
    kind = np.array(["anterior" if s == frozenset({"AMI"}) else "inferior" if s == frozenset({"IMI"}) else ""
                     for s in sub_bank])
    out = {"n": {k: int((kind == k).sum()) for k in ("anterior", "inferior")}}
    mi_cls = 1
    for jname in ("test", "test_judge2"):
        b = Bank(B / N / f"{jname}.npz")
        names = {j: k for k, j in b.index.items()}
        v13 = np.array(["V1-V3" in names[j][1] for j in range(len(names))])
        lim = np.array(["limb" in names[j][1] for j in range(len(names))])
        r = {}
        for k in ("anterior", "inferior"):
            idx = np.flatnonzero(kind == k)
            V = np.stack([b.verified_cells(i, mi_cls) for i in idx])
            r[k] = {"recordings": int(len(idx)), "with_pass": int(V.any(1).sum()), "passing": int(V.sum()),
                    "with_v13": int(V[:, v13].sum()), "only_limb": int(V[:, lim & ~v13].sum())}
        tab = [[r["anterior"]["with_v13"], r["anterior"]["passing"] - r["anterior"]["with_v13"]],
               [r["inferior"]["with_v13"], r["inferior"]["passing"] - r["inferior"]["with_v13"]]]
        r["fisher_p_greater"] = float(fisher_exact(tab, alternative="greater")[1])
        out[jname] = r
        print(f"MI bank {jname:11s}: passing claims with V1-V3: anterior {r['anterior']['with_v13']}/{r['anterior']['passing']} "
              f"({r['anterior']['with_pass']}/{r['anterior']['recordings']} recordings) vs inferior "
              f"{r['inferior']['with_v13']}/{r['inferior']['passing']} ({r['inferior']['with_pass']}/{r['inferior']['recordings']}),"
              f" p={r['fisher_p_greater']:.2g}")
    for label, runs in (("llm_check", [f"grpo_verified_ver{s}" for s in SEEDS]),
                        ("llm_accuracy", [f"grpo_accuracy_ver{s}" for s in SEEDS]),
                        ("nonllm_check", [f"policy_verified{s}" for s in SEEDS]),
                        ("nonllm_rule", [f"policy_heuristic{s}" for s in SEEDS])):
        cnt = {"anterior": [0, 0], "inferior": [0, 0]}
        for run in runs:
            for p in preds(N, run) or []:
                k = kind[p["bank_i"]]
                if k and p["label"] == p["label_true"] and p["group"] is not None:
                    cnt[k][0] += "V1-V3" in p["group"]
                    cnt[k][1] += 1
        if not cnt["anterior"][1]:
            continue
        tab = [[cnt["anterior"][0], cnt["anterior"][1] - cnt["anterior"][0]],
               [cnt["inferior"][0], cnt["inferior"][1] - cnt["inferior"][0]]]
        out[label] = {"anterior": cnt["anterior"], "inferior": cnt["inferior"],
                      "fisher_p_greater": float(fisher_exact(tab, alternative="greater")[1])}
        print(f"MI agent {label:13s}: cites V1-V3 in {cnt['anterior'][0]}/{cnt['anterior'][1]} anterior vs "
              f"{cnt['inferior'][0]}/{cnt['inferior'][1]} inferior correct answers, p={out[label]['fisher_p_greater']:.2g}")
    return out


def verifier_ci():
    from sklearn.metrics import roc_auc_score
    files = sorted((ROOT / "results" / "verifier").glob("synth_planted_eval_seed*_ci.json"))
    if not files:
        return None
    runs = [json.loads(f.read_text()) for f in files]
    out = {}
    rng = np.random.default_rng(0)
    for method in runs[0]["methods"]:
        aucs, boots, det, corr = [], [], [], []
        for R in runs:
            m = R["methods"][method]
            s = m["scores"]
            row, kind, sc = np.array(s["row"]), np.array(s["kind"]), np.array(s["combined"])
            def auc(rows):  # noqa: E306 - planted cell against the three wrong cells, pooled
                sel = np.isin(row, rows)
                return roc_auc_score(kind[sel] == "true", sc[sel])
            urows = np.unique(row)
            aucs.append(auc(urows))
            boots.append([auc(rng.choice(urows, len(urows))) for _ in range(500)])
            det.append(m["discriminative_acc"])
            corr.append(m["group_corr_edited"])
        b = np.array(boots).mean(0)  # bootstrap of the three-seed mean
        out[method] = {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                       "auc_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                       "auc_per_seed": [float(a) for a in aucs], "detector": float(np.mean(det)),
                       "corr": float(np.nanmean(corr)), "n_seeds": len(runs)}
        print(f"verifier {method:24s} AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f} "
              f"[{out[method]['auc_ci95'][0]:.3f}, {out[method]['auc_ci95'][1]:.3f}]  detector {np.mean(det):.2f}  "
              f"corr {np.nanmean(corr):.2f}  ({len(runs)} seeds)")
    # paired differences: every method edits the same recordings within a seed
    out["paired"] = {}
    for a, b in (("hybrid:prior+learned", "hybrid:none"), ("diffusion:prior+learned", "diffusion:none"),
                 ("hybrid:prior+learned", "splice"), ("hybrid:prior+learned", "zero")):
        diffs, boots = [], []
        for R in runs:
            sa, sb = R["methods"][a]["scores"], R["methods"][b]["scores"]
            row, kind = np.array(sa["row"]), np.array(sa["kind"])
            assert sa["row"] == sb["row"] and sa["kind"] == sb["kind"]
            xa, xb = np.array(sa["combined"]), np.array(sb["combined"])
            urows = np.unique(row)

            def d(rows):
                sel = np.isin(row, rows)
                return roc_auc_score(kind[sel] == "true", xa[sel]) - roc_auc_score(kind[sel] == "true", xb[sel])
            diffs.append(d(urows))
            boots.append([d(rng.choice(urows, len(urows))) for _ in range(500)])
        bb = np.array(boots).mean(0)
        out["paired"][f"{a} - {b}"] = {"diff": float(np.mean(diffs)), "ci95": [float(np.percentile(bb, 2.5)),
                                                                             float(np.percentile(bb, 97.5))],
                                       "share_boot_le_0": float((bb <= 0).mean())}
        print(f"verifier paired {a} - {b}: {np.mean(diffs):+.3f} [{np.percentile(bb, 2.5):+.3f}, "
              f"{np.percentile(bb, 97.5):+.3f}]  (bootstrap share <= 0: {(bb <= 0).mean():.3f})")
    return out


def main():
    import argparse
    parts = {"stricter": stricter, "subject": subject, "swap": swap, "no_neighbours": no_neighbours, "mi": mi,
             "verifier_ci": verifier_ci}
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(parts), choices=list(parts))
    args = ap.parse_args()
    path = ROOT / "results" / "review6_eval.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for p in args.parts:
        out[p] = parts[p]()
    path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
