"""Numbers and LaTeX rows for the classifier-hint section of the paper.

Reads results/hint_summary.json (scripts/hint_summary.py) and the prediction files, and prints
  1. the rows added to the accuracy table (five-fold InceptionTime, agents with the hint),
  2. the hint table: accuracy without and with the hint, agreement with the hint, checked rate and
     judge-2 checked rate for the accuracy and check rewards with the hint, and without the hint,
  3. how often the agents copy the hint (agreement, recordings fixed and broken), per reward,
  4. the composition analysis: checked and true-evidence rates on the recordings answered correctly
     with and without the hint, and on the recordings that only the hinted agent answers correctly,
  5. one-sided Wilcoxon tests over the six datasets (check vs accuracy reward with the hint).
Writes results/hint_tables.json.

Usage:
    python scripts/hint_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.data import load_dataset  # noqa: E402

DATASETS = ["synth_planted", "pamap2_group", "ptbxl_group", "ucihar_subj_group", "natops_group", "cricket_group"]
SHORT = {"synth_planted": "Synthetic", "pamap2_group": "PAMAP2", "ptbxl_group": "PTB-XL",
         "ucihar_subj_group": "UCI-HAR", "natops_group": "NATOPS", "cricket_group": "Cricket"}
SEEDS = ("", "_s1", "_s2")


def preds(name: str, run: str):
    f = ROOT / "results" / "agent" / name / f"pred_{run}.json"
    return {p["bank_i"]: p for p in json.loads(f.read_text())["preds"]} if f.exists() else None


def m(S, ds, key, metric="accuracy"):
    e = S[ds].get(key)
    return (e[metric]["mean"] * 100, e[metric]["sd"] * 100, e.get("n_seeds", 1)) if e and metric in e else None


def fmt(v, sd=False):
    if v is None:
        return "--"
    return f"{v[0]:.1f}" + (f"$\\pm${v[1]:.1f}" if sd and v[2] > 1 else "")


def main():
    S = json.loads((ROOT / "results" / "hint_summary.json").read_text())
    out = {"copy": {}, "composition": {}, "wilcoxon": {}}

    # 1. accuracy-table rows
    hint_acc = {ds: json.loads((ROOT / "results" / "hint" / f"{ds}.json").read_text())["acc_test"] * 100
                for ds in DATASETS}
    print("% rows for tab:acc")
    print("InceptionTime, five-fold mean (hint) & " + " & ".join(f"{hint_acc[ds]:.1f}" for ds in DATASETS) + r" \\")
    for lab, key in (("LLM agent + hint, check reward", "GRPO check reward + hint"),
                     ("Non-LLM agent + hint, check reward", "non-LLM, check reward + hint")):
        print(f"{lab} & " + " & ".join(fmt(m(S, ds, key)) for ds in DATASETS) + r" \\")
    print("% seeds: " + ", ".join(f"{SHORT[ds]} {m(S, ds, 'GRPO check reward + hint')[2]}/"
                                   f"{m(S, ds, 'GRPO accuracy reward + hint')[2]}" for ds in DATASETS))

    # 2. hint table (LLM agent)
    print("\n% rows for tab:hint (LLM): acc w/o, acc with, agrees, checked acc+hint, check+hint, check w/o,"
          " judge acc+hint, check+hint, check w/o")
    for ds in DATASETS:
        c = [m(S, ds, "GRPO check reward"), m(S, ds, "GRPO check reward + hint")]
        ch = [m(S, ds, "GRPO accuracy reward + hint", "checked"), m(S, ds, "GRPO check reward + hint", "checked"),
              m(S, ds, "GRPO check reward", "checked")]
        j2 = [m(S, ds, "GRPO accuracy reward + hint", "checked_judge2"),
              m(S, ds, "GRPO check reward + hint", "checked_judge2"), m(S, ds, "GRPO check reward", "checked_judge2")]
        agree = []
        hp = np.load(ROOT / "results" / "hint" / f"{ds}.npz")["test_prob"].argmax(1)
        tag = "_div" if ds.startswith("synth") else "_ver"
        for r in ("verified", "accuracy"):
            st = []
            for s in SEEDS:
                P = preds(f"{ds}_hint", f"grpo_{r}{tag}{s}")
                if P is None:
                    continue
                lab = np.array([p["label"] for p in P.values()])
                yt = np.array([p["label_true"] for p in P.values()])
                h = hp[np.array(list(P))]
                st.append([np.mean(lab == h) * 100, int(((lab == yt) & (h != yt)).sum()),
                           int(((lab != yt) & (h == yt)).sum()), len(lab)])
            out["copy"].setdefault(ds, {})[r] = st
            if r == "verified":
                agree = st
        ag = np.mean([a[0] for a in agree])
        print(f"{SHORT[ds]} & {fmt(c[0])} & {fmt(c[1])} & {ag:.1f} & "
              + " & ".join(fmt(v) for v in ch) + " & " + " & ".join(fmt(v) for v in j2) + r" \\")

    # 3. copy statistics
    print("\n% agreement with the hint (mean over seeds), recordings fixed / broken (summed over seeds)")
    for ds in DATASETS:
        for r, st in out["copy"][ds].items():
            print(f"%  {SHORT[ds]:9s} {r:8s} agrees {np.mean([a[0] for a in st]):5.1f}%  fixes {sum(a[1] for a in st)}"
                  f"  breaks {sum(a[2] for a in st)}  of {sum(a[3] for a in st)}  ({len(st)} seeds)")

    # 4. composition: same recordings, with and without the hint (seed by seed, then pooled)
    print("\n% composition (LLM, pooled over seeds present for both): checked on both-correct recordings"
          " without / with the hint; checked on recordings only the hinted agent gets right")
    for ds in DATASETS:
        bank = Bank(ROOT / "results" / "bank" / ds / "test.npz")
        gt = None
        if ds.startswith("synth"):
            d = load_dataset(ds, max_test=2000)
            rows = np.load(ROOT / "results" / "bank" / ds / "test.npz")["rows"]
            gt = (d.groups_test["ev_window"][rows], d.groups_test["ev_group"][rows], sorted(d.meta.channel_groups))
        tag = "_div" if ds.startswith("synth") else "_ver"
        for r in ("verified", "accuracy"):
            acc = {"both_base": [], "both_hint": [], "new": [], "te_both_base": [], "te_both_hint": [], "te_new": []}
            for s in SEEDS:
                A, B = preds(ds, f"grpo_{r}{tag}{s}"), preds(f"{ds}_hint", f"grpo_{r}{tag}{s}")
                if A is None or B is None:
                    continue

                def ok(p):
                    return p["label"] == p["label_true"]

                def chk(p):
                    return bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"])

                both = [i for i in A if ok(A[i]) and ok(B[i])]
                new = [i for i in B if ok(B[i]) and not ok(A[i])]
                acc["both_base"] += [chk(A[i]) for i in both]
                acc["both_hint"] += [chk(B[i]) for i in both]
                acc["new"] += [chk(B[i]) for i in new]
                if gt:
                    w, g, names = gt

                    def te(p):
                        return p["window"] == w[p["bank_i"]] and p["group"] == names[g[p["bank_i"]]]
                    acc["te_both_base"] += [te(A[i]) for i in both]
                    acc["te_both_hint"] += [te(B[i]) for i in both]
                    acc["te_new"] += [te(B[i]) for i in new]
            res = {k: (float(np.mean(v)) * 100 if v else None, len(v)) for k, v in acc.items()}
            out["composition"].setdefault(ds, {})[r] = res
            line = (f"%  {SHORT[ds]:9s} {r:8s} both-correct n={res['both_base'][1]:4d}: {res['both_base'][0]:5.1f} -> "
                    f"{res['both_hint'][0]:5.1f} | only-with-hint n={res['new'][1]:3d}: "
                    + (f"{res['new'][0]:5.1f}" if res["new"][0] is not None else "  --"))
            if gt:
                line += (f" | true ev. both {res['te_both_base'][0]:5.1f} -> {res['te_both_hint'][0]:5.1f},"
                         f" only-with-hint {res['te_new'][0]:5.1f}")
            print(line)

    # 5. Wilcoxon over datasets: check vs accuracy reward with the hint, and hint vs no hint (check reward)
    print()
    for metric in ("checked", "checked_judge2"):
        a = [m(S, ds, "GRPO check reward + hint", metric)[0] for ds in DATASETS]
        b = [m(S, ds, "GRPO accuracy reward + hint", metric)[0] for ds in DATASETS]
        d = np.round(np.array(a) - np.array(b), 6)
        p = wilcoxon(a, b, alternative="greater").pvalue if np.any(d != 0) else 1.0
        out["wilcoxon"][f"{metric}: check vs accuracy reward, with hint"] = {"diff": d.tolist(), "p": float(p)}
        print(f"% Wilcoxon {metric}: check vs accuracy reward with hint, mean gain {d.mean():.1f}, "
              f"positive on {(d > 0).sum()}/6, one-sided p={p:.3f}")
    for metric in ("checked", "checked_judge2"):
        a = [m(S, ds, "GRPO check reward + hint", metric)[0] for ds in DATASETS]
        b = [m(S, ds, "GRPO check reward", metric)[0] for ds in DATASETS]
        d = np.round(np.array(a) - np.array(b), 6)
        p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
        out["wilcoxon"][f"{metric}: check reward, hint vs no hint"] = {"diff": d.tolist(), "p": float(p)}
        print(f"% Wilcoxon {metric}: check reward with vs without hint, mean change {d.mean():.1f}, two-sided p={p:.3f}")
    (ROOT / "results" / "hint_tables.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
