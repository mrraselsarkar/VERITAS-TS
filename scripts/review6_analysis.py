"""Review round 6 analyses from the stored banks and predictions (no new runs needed).

  train_test      oracle accuracy and share of recordings with a passing claim, training vs test bank
  donors          chance that a Test 2 edit copies content of the recording's own class (any-class donors)
  per_class       baseline claim: the claim that passes most often for the class on the training data
  equivalence     accuracy of the check reward against the accuracy reward: intervals and TOST margin
  ltt_heldout     abstention with the threshold chosen on one half of the in-distribution test set and
                  evaluated on the other half and on the shifted set (repeated random halves); scores:
                  check score, oracle confidence, oracle agreement, CNN confidence, agent's own confidence
  mi_location     PTB-XL: do passing claims and the agents cite V1-V3 more for anterior than for inferior MI?

Writes results/review6_analysis.json.
"""
import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.selective import ltt_threshold  # noqa: E402
from veritas.data import load_dataset  # noqa: E402
from review5_eval import checked, preds  # noqa: E402

A, B, D = ROOT / "results" / "agent", ROOT / "results" / "bank", ROOT / "results" / "diffusion"
DATASETS = {"synth_planted": "div", "pamap2_group": "ver", "ptbxl_group": "ver", "ucihar_subj_group": "ver",
            "natops_group": "ver", "cricket_group": "ver"}
SEEDS = ("", "_s1", "_s2")


def runs(ds, reward):
    return [f"grpo_{reward}_{DATASETS[ds]}{s}" for s in SEEDS]


def train_test():
    out = {}
    for ds in DATASETS:
        r = {}
        for part in ("train", "test"):
            b = Bank(B / ds / f"{part}.npz")
            ok = b.p0.argmax(1) == b.y
            anyp = np.array([b.verified_cells(i, int(b.y[i])).any() for i in range(len(b))])
            r[part] = {"oracle_acc": float(ok.mean()), "any_pass_all": float(anyp.mean()),
                       "any_pass_oracle_correct": float(anyp[ok].mean())}
        out[ds] = r
        print(f"train/test {ds:18s} oracle acc {r['train']['oracle_acc']:.3f}/{r['test']['oracle_acc']:.3f}  "
              f"passing (oracle correct) {r['train']['any_pass_oracle_correct']:.3f}/{r['test']['any_pass_oracle_correct']:.3f}")
    return out


def donors():
    out = {}
    for ds in DATASETS:
        src, _ = agent_mode(ds)
        data = load_dataset(src, max_test=2000)
        prior = np.bincount(data.y_train, minlength=len(data.meta.class_names)) / len(data.y_train)
        y = Bank(B / ds / "test.npz").y
        p = 1 - (1 - prior[y]) ** 2  # at least one of the m = 2 donors of the recording's class
        out[ds] = {"mean": float(p.mean()), "max_class": float(1 - (1 - prior.max()) ** 2)}
        print(f"donors {ds:18s} same-class donor in Test 2: {p.mean():.3f} (largest class {out[ds]['max_class']:.3f})")
    return out


def claim_names(bank):
    return {j: k for k, j in bank.index.items()}


def per_class():
    """For every class, the claim that passes most often on the training data; scored on the correct answers
    of each agent, against the agent's own claims."""
    out = {}
    for ds in DATASETS:
        tr, te, j2 = Bank(B / ds / "train.npz"), Bank(B / ds / "test.npz"), Bank(B / ds / "test_judge2.npz")
        n_cls = int(max(tr.y.max(), te.y.max())) + 1
        best = {}
        for c in range(n_cls):
            rec = np.flatnonzero(tr.y == c)
            rates = np.mean([tr.verified_cells(i, c) for i in rec], axis=0)
            best[c] = int(rates.argmax())
        names = claim_names(te)
        r = {"claim_per_class": {c: "/".join(map(str, names[j])) for c, j in best.items()}}
        for label, reward in (("llm_check", "verified"), ("llm_accuracy", "accuracy")):
            vals = []
            for run in runs(ds, reward):
                P = preds(ds, run)
                ok = np.array([p["label"] == p["label_true"] for p in P])
                agent, agent2 = checked(te, P)[ok], checked(j2, P)[ok]
                base = np.array([te.verified(p["bank_i"], best[p["label"]], p["label"]) for p in P])[ok]
                base2 = np.array([j2.verified(p["bank_i"], best[p["label"]], p["label"]) for p in P])[ok]
                vals.append([agent.mean(), base.mean(), agent2.mean(), base2.mean()])
            v = np.mean(vals, 0)
            r[label] = {"agent": float(v[0]), "per_class": float(v[1]), "agent_judge2": float(v[2]),
                        "per_class_judge2": float(v[3])}
        out[ds] = r
        print(f"per-class {ds:18s} check-reward LLM {r['llm_check']['agent']:.3f} vs per-class claim "
              f"{r['llm_check']['per_class']:.3f} | judge 2: {r['llm_check']['agent_judge2']:.3f} vs "
              f"{r['llm_check']['per_class_judge2']:.3f}")
    # synthetic: true-evidence rate of the per-class claim
    z = np.load(B / "synth_planted" / "test.npz")
    data = load_dataset("synth_planted", max_test=2000)
    gnames = sorted(data.meta.channel_groups)
    w, g = data.groups_test["ev_window"][z["rows"]], data.groups_test["ev_group"][z["rows"]]
    te = Bank(B / "synth_planted" / "test.npz")
    names = claim_names(te)
    tr = Bank(B / "synth_planted" / "train.npz")
    best = {c: int(np.mean([tr.verified_cells(i, c) for i in np.flatnonzero(tr.y == c)], 0).argmax())
            for c in range(4)}
    te_rates = []
    for run in runs("synth_planted", "verified"):
        P = [p for p in preds("synth_planted", run) if p["label"] == p["label_true"]]
        te_rates.append(np.mean([names[best[p["label"]]] == (int(w[p["bank_i"]]), gnames[g[p["bank_i"]]]) for p in P]))
    out["synth_planted"]["per_class_true_evidence"] = float(np.mean(te_rates))
    print(f"per-class synthetic: true-evidence rate of the per-class claim {np.mean(te_rates):.3f}")
    return out


def equivalence():
    out, diffs = {}, []
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        ok = {}
        for reward in ("accuracy", "verified"):
            Ps = [preds(ds, r) for r in runs(ds, reward)]
            ok[reward] = np.stack([[p["label"] == p["label_true"] for p in P] for P in Ps]).astype(float)
        d = ok["verified"].mean(0) - ok["accuracy"].mean(0)  # per recording, mean over seeds
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        out[ds] = {"diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        diffs.append(d.mean())
        print(f"equivalence {ds:18s} accuracy difference {100 * d.mean():+.1f} points, 95% CI "
              f"[{100 * out[ds]['ci95'][0]:+.1f}, {100 * out[ds]['ci95'][1]:+.1f}]")
    diffs = np.array(diffs)
    m, se = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs))
    q = student_t.ppf(0.95, len(diffs) - 1)
    lo, hi = m - q * se, m + q * se  # 90% interval = TOST at the 5% level
    out["across"] = {"mean": float(m), "ci90": [float(lo), float(hi)], "tost_margin": float(max(abs(lo), abs(hi)))}
    print(f"equivalence across datasets: mean {100 * m:+.2f} points, 90% CI [{100 * lo:+.2f}, {100 * hi:+.2f}] "
          f"-> equivalent within +-{100 * max(abs(lo), abs(hi)):.2f} points (TOST, 5%)")
    return out


def cnn_probs(src, split, rows):
    ddir = D / src / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(src, split=split, max_test=2000)
    X = ((data.X_test[rows] - st["mu"]) / st["sd"]).astype(np.float32)
    return pickle.loads((ddir / "oracle_cnn.pkl").read_bytes()).predict_proba(X)


def score_table(name, src, split, run):
    bank = Bank(B / name / "test.npz")
    z = np.load(B / name / "test.npz")
    P = sorted(json.loads((A / name / f"pred_{run}.json").read_text())["preds"], key=lambda p: p["bank_i"])
    i = np.array([p["bank_i"] for p in P])
    y = np.array([p["label_true"] for p in P])
    yh = np.array([-1 if p["label"] is None else p["label"] for p in P])
    s = {"check": np.array([bank.margin(ii, bank.claim_id(p["window"], p["group"]), None if h < 0 else int(h))
                            for ii, p, h in zip(i, P, yh)]),
         "oracle": np.array([0.0 if h < 0 else bank.p0[ii, h] for ii, h in zip(i, yh)]),
         "agree": (bank.p0[i].argmax(1) == yh).astype(float)}
    pc = cnn_probs(src, split, z["rows"][i])
    s["cnn"] = np.array([0.0 if h < 0 else pc[k, h] for k, h in enumerate(yh)])
    conf = A / name / f"conf_{run}.json"
    if conf.exists():
        c = json.loads(conf.read_text())
        pa = dict(zip(c["bank_i"], c["probs"]))
        s["agent"] = np.array([0.0 if h < 0 else pa[ii][h] for ii, h in zip(i, yh)])
    return s, yh != y


def ltt_heldout(n_splits=500, delta=0.1):
    out = {}
    for name, src, run, alpha in (("synth_planted", "synth_planted", "grpo_verified_div", 0.05),
                                  ("pamap2_subj_group", "pamap2_subj", "grpo_verified_ver", 0.1),
                                  ("ucihar_subj_group", "ucihar_subj", "grpo_verified_ver", 0.1)):
        s_in, w_in = score_table(name, src, "standard", run)
        s_sh, w_sh = score_table(f"{name}_shift", src, "shift", run)
        rng = np.random.default_rng(0)
        res = {}
        for k in s_in:
            if k not in s_sh:
                continue
            rows = []
            for _ in range(n_splits):
                p = rng.permutation(len(w_in))
                cal, ev = p[: len(p) // 2], p[len(p) // 2:]
                # continuous scores: thresholds at the 20%, 25%, ..., 100% coverage levels of the calibration
                # half, tested from strict to loose (a single-answer level could never be certified)
                grid = None if k in ("check", "agree") else np.quantile(s_in[k][cal], 1 - np.arange(0.2, 1.001, 0.05))
                lam = ltt_threshold(s_in[k][cal], w_in[cal], alpha, delta, grid=grid)
                a_ev = np.zeros(len(ev), bool) if lam is None else s_in[k][ev] >= lam
                a_sh = np.zeros(len(w_sh), bool) if lam is None else s_sh[k] >= lam
                # answered and wrong counts; a split without a certified threshold answers nothing
                rows.append([a_ev.sum(), w_in[ev][a_ev].sum(), a_sh.sum(), w_sh[a_sh].sum(),
                             (w_in[ev][a_ev].mean() if a_ev.any() else 0.0) <= alpha, lam is not None])
            r = np.array(rows, float)
            m = [r[:, 0].mean() / len(ev), r[:, 1].sum() / max(r[:, 0].sum(), 1),  # error pooled over answers
                 r[:, 2].mean() / len(w_sh), r[:, 3].sum() / max(r[:, 2].sum(), 1), r[:, 4].mean(), r[:, 5].mean()]
            res[k] = {"cov_in": float(m[0]), "err_in": float(m[1]), "cov_shift": float(m[2]), "err_shift": float(m[3]),
                      "share_in_within_alpha": float(m[4]), "share_certified": float(m[5])}
            print(f"ltt {name:18s} {k:7s} in: answered {100 * m[0]:5.1f}% error {100 * m[1]:4.1f}% "
                  f"(within target in {100 * m[4]:.0f}% of splits, certified in {100 * m[5]:.0f}%) | "
                  f"shifted: answered {100 * m[2]:5.1f}% error {100 * m[3]:4.1f}%")
        out[name] = res
    return out


def mi_location():
    import pandas as pd
    from veritas.data.base import RAW_DIR, load_processed
    raw = RAW_DIR / "ptbxl"
    db = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    codes = db.scp_codes.apply(ast.literal_eval)
    sup = codes.apply(lambda c: sorted({diag.loc[k, "diagnostic_class"] for k in c if k in diag.index}))
    db = db[sup.str.len() == 1].copy()
    ds = load_processed("ptbxl")
    assert len(db) == len(ds.y)
    sub = codes.loc[db.index].apply(lambda c: frozenset(diag.loc[k, "diagnostic_subclass"] for k in c
                                                         if k in diag.index and diag.loc[k, "diagnostic_class"] == "MI"))
    data = load_dataset("ptbxl", max_test=2000)
    bank = Bank(B / "ptbxl_group" / "test.npz")
    j2 = Bank(B / "ptbxl_group" / "test_judge2.npz")
    z = np.load(B / "ptbxl_group" / "test.npz")
    sub_bank = sub.to_numpy()[data.test_rows[z["rows"]]]
    kind = np.array(["anterior" if s == frozenset({"AMI"}) else "inferior" if s == frozenset({"IMI"}) else ""
                     for s in sub_bank])
    mi = 1
    names = claim_names(bank)
    v13 = np.array(["V1-V3" in names[j][1] for j in range(len(names))])
    out = {"n_anterior": int((kind == "anterior").sum()), "n_inferior": int((kind == "inferior").sum())}
    # (a) the check itself: share of passing claims that include V1-V3
    for jname, b in (("oracle", bank), ("judge2", j2)):
        r = {}
        for k in ("anterior", "inferior"):
            idx = np.flatnonzero((kind == k) & (b.y == mi))
            V = np.stack([b.verified_cells(i, mi) for i in idx])
            r[k] = {"passing": int(V.sum()), "with_v13": int(V[:, v13].sum()),
                    "share_v13": float(V[:, v13].sum() / max(V.sum(), 1)), "recordings_any_pass": int(V.any(1).sum())}
        tab = [[r["anterior"]["with_v13"], r["anterior"]["passing"] - r["anterior"]["with_v13"]],
               [r["inferior"]["with_v13"], r["inferior"]["passing"] - r["inferior"]["with_v13"]]]
        r["fisher_p_greater"] = float(fisher_exact(tab, alternative="greater")[1])
        out[f"check_{jname}"] = r
        print(f"MI {jname}: passing claims with V1-V3 anterior {r['anterior']['share_v13']:.2f} "
              f"({r['anterior']['passing']} claims) vs inferior {r['inferior']['share_v13']:.2f} "
              f"({r['inferior']['passing']}), one-sided Fisher p={r['fisher_p_greater']:.2g}")
    # (b) the agents: share of correct MI answers whose claim includes V1-V3 (three seeds pooled)
    for label, rs in (("llm_check", runs("ptbxl_group", "verified")), ("llm_accuracy", runs("ptbxl_group", "accuracy")),
                      ("nonllm_check", ["policy_verified", "policy_verified_s1", "policy_verified_s2"]),
                      ("nonllm_rule", ["policy_heuristic", "policy_heuristic_s1", "policy_heuristic_s2"])):
        cnt = {"anterior": [0, 0], "inferior": [0, 0]}
        for run in rs:
            for p in preds("ptbxl_group", run):
                k = kind[p["bank_i"]]
                if k and p["label"] == p["label_true"] == mi and p["group"] is not None:
                    cnt[k][0] += "V1-V3" in p["group"]
                    cnt[k][1] += 1
        tab = [[cnt["anterior"][0], cnt["anterior"][1] - cnt["anterior"][0]],
               [cnt["inferior"][0], cnt["inferior"][1] - cnt["inferior"][0]]]
        pval = float(fisher_exact(tab, alternative="greater")[1])
        out[label] = {"anterior": cnt["anterior"], "inferior": cnt["inferior"], "fisher_p_greater": pval}
        print(f"MI {label:13s}: cites V1-V3 in {cnt['anterior'][0]}/{cnt['anterior'][1]} anterior vs "
              f"{cnt['inferior'][0]}/{cnt['inferior'][1]} inferior answers, one-sided Fisher p={pval:.2g}")
    return out


def main():
    import argparse
    parts = {"train_test": train_test, "donors": donors, "per_class": per_class, "equivalence": equivalence,
             "mi_location": mi_location, "ltt_heldout": ltt_heldout}
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(parts), choices=list(parts))
    args = ap.parse_args()
    path = ROOT / "results" / "review6_analysis.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for p in args.parts:
        out[p] = parts[p]()
    path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
