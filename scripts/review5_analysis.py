"""Analyses for review round 5 that need no new training (existing banks and predictions).

  coverage     share of the recording (channels x time) covered by a passing claim, per dataset
  tiers_test   is the checked tier more accurate than unchecked answers the oracle agrees with? (Fisher)
  multi_pass   synthetic: how many cells pass per recording, and agent vs checker-only pick
  occl_oracle  synthetic: zero occlusion computed on the oracle itself (the model the check uses)
  floor_means  chance / ceiling table as the mean over three seeds
  oracle_acc   oracle accuracy on the bank test rows vs on the full test set (PTB-XL 75.7 vs 75.8)
  seed_spread  per-seed accuracy and prediction overlap (zero standard deviations)
  neighbours   how often the agent's class equals the neighbour-vote majority

Writes results/review5_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.reward import Bank  # noqa: E402
from veritas.data.base import load_processed  # noqa: E402

A, B = ROOT / "results" / "agent", ROOT / "results" / "bank"
DATA = {"synth_planted": ("grpo_accuracy_div", "grpo_verified_div"),
        "pamap2_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "ptbxl_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "ucihar_subj_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "natops_group": ("grpo_accuracy_ver", "grpo_verified_ver"),
        "cricket_group": ("grpo_accuracy_ver", "grpo_verified_ver")}


def preds(ds, run):
    f = A / ds / f"pred_{run}.json"
    return json.loads(f.read_text())["preds"] if f.exists() else None


def seed_runs(run):
    return [run, f"{run}_s1", f"{run}_s2"]


def coverage():
    out = {}
    for ds, (_, run) in DATA.items():
        source, claims = agent_mode(ds)
        meta = load_processed(source).meta
        C = sum(len(v) for v in meta.channel_groups.values())
        bank = Bank(B / ds / "test.npz")
        frac_t = 1.0 if claims == "group" else 1.0 / 8

        def size(group):
            return frac_t * sum(len(meta.channel_groups[g]) for g in group.split("+")) / C
        all_claims = [size(g) for (_, g) in sorted(bank.index, key=bank.index.get)]
        P = preds(ds, run)
        passing = [size(p["group"]) for p in P if p["label"] == p["label_true"] and p["group"]
                   and bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"])]
        out[ds] = {"claim_set_mean": float(np.mean(all_claims)), "passing_claim_mean": float(np.mean(passing)),
                   "passing_claim_max": float(np.max(passing)), "n_passing": len(passing)}
        print(f"coverage {ds:18s} claim set {np.mean(all_claims):.2f}  passing claims {np.mean(passing):.2f}")
    return out


def tiers_test():
    out = {}
    for ds in ("synth_planted", "pamap2_group", "ptbxl_group", "ucihar_subj_group"):
        bank = Bank(B / ds / "test.npz")
        P = preds(ds, DATA[ds][1])
        chk, agr = [], []
        for p in P:
            i, lab = p["bank_i"], p["label"]
            if lab is None:
                continue
            wrong = lab != p["label_true"]
            if bank.verified(i, bank.claim_id(p["window"], p["group"]), lab):
                chk.append(wrong)
            elif bank.p0[i].argmax() == lab:
                agr.append(wrong)
        t = [[sum(chk), len(chk) - sum(chk)], [sum(agr), len(agr) - sum(agr)]]
        _, p = fisher_exact(t, alternative="less")
        out[ds] = {"checked_err": float(np.mean(chk)), "agree_unchecked_err": float(np.mean(agr)),
                   "n_checked": len(chk), "n_agree": len(agr), "p_fisher_one_sided": float(p)}
        print(f"tiers {ds:18s} checked {np.mean(chk):.3f} (n {len(chk)})  agree-unchecked {np.mean(agr):.3f} "
              f"(n {len(agr)})  p = {p:.2g}")
    return out


def multi_pass():
    from veritas.data import load_dataset
    ds = "synth_planted"
    data = load_dataset(ds, max_test=2000)
    names = sorted(data.meta.channel_groups)
    G = len(names)
    bank = Bank(B / ds / "test.npz")
    z = np.load(B / ds / "test.npz")
    rows, y = z["rows"], z["y"]
    truth = data.groups_test["ev_window"][rows] * G + data.groups_test["ev_group"][rows]
    cell_ids = [bank.claim_id(w, names[g]) for w in range(8) for g in range(G)]
    p0 = z["p0"][np.arange(len(y)), y]
    chk = (np.clip(p0[:, None] - z["nec_ptrue"].mean(-1), 0, None) * z["suf_ptrue"].mean(-1))[:, cell_ids]
    top_chk = chk.argmax(1)
    n_pass = np.array([bank.verified_cells(i, int(y[i]))[cell_ids].sum() for i in range(len(y))])
    P = {p["bank_i"]: p for p in preds(ds, "grpo_verified_div")}
    ok = [i for i, p in P.items() if p["label"] == p["label_true"] and p["window"] is not None]
    multi = [i for i in ok if n_pass[i] >= 2]
    agent_true = np.mean([P[i]["window"] * G + names.index(P[i]["group"]) == truth[i] for i in multi])
    chk_true = np.mean([top_chk[i] == truth[i] for i in multi])
    res = {"share_recordings_multi_pass": float(np.mean(n_pass[ok] >= 2)), "mean_passing_cells": float(n_pass[ok].mean()),
           "multi_agent_true": float(agent_true), "multi_checker_true": float(chk_true), "n_multi": len(multi)}
    print(f"multi_pass: {res}")
    return res


def occl_oracle():
    from veritas.data import load_dataset
    ds = "synth_planted"
    data = load_dataset(ds, max_test=2000)
    G = len(data.meta.channel_groups)
    z = np.load(B / "synth_planted_occl" / "test.npz")
    top = np.load(B / "synth_planted_occl" / "test_top.npy")
    bank = Bank(B / "synth_planted_zero" / "test.npz")
    rows, y = z["rows"], z["y"]
    names = sorted(data.meta.channel_groups)
    keys = sorted(bank.index, key=bank.index.get)
    top_cell = np.array([keys[t][0] * G + names.index(keys[t][1]) for t in top])
    truth = data.groups_test["ev_window"][rows] * G + data.groups_test["ev_group"][rows]
    ok = z["p0"].argmax(1) == y
    res = {"true_evidence": float((top_cell == truth)[ok].mean()), "n_correct": int(ok.sum())}
    print(f"occlusion on the oracle: {res}")
    return res


def floor_means():
    out = {}
    for ds, runs in DATA.items():
        bank = Bank(B / ds / "test.npz")
        r = {}
        for run in runs:
            vals = []
            for sr in seed_runs(run):
                P = preds(ds, sr)
                if P is None:
                    continue
                ok = [p for p in P if p["label"] == p["label_true"]]
                agent = np.mean([bank.verified(p["bank_i"], bank.claim_id(p["window"], p["group"]), p["label_true"]) for p in ok])
                cells = [bank.verified_cells(p["bank_i"], p["label_true"]) for p in ok]
                vals.append((agent, np.mean([c.mean() for c in cells]), np.mean([c.any() for c in cells])))
            v = np.array(vals)
            r[run] = {"agent": float(v[:, 0].mean()), "random": float(v[:, 1].mean()), "best": float(v[:, 2].mean()),
                      "n_seeds": len(vals)}
        out[ds] = r
        print(f"floor {ds:18s} " + "  ".join(f"{k}: agent {v['agent']:.3f} rand {v['random']:.3f} best {v['best']:.3f} ({v['n_seeds']} seeds)"
                                                for k, v in r.items()))
    return out


def oracle_acc():
    out = {}
    for ds in DATA:
        bank = Bank(B / ds / "test.npz")
        out[ds] = float((bank.p0.argmax(1) == bank.y).mean())
    print(f"oracle accuracy on the bank test rows: { {k: round(100 * v, 2) for k, v in out.items()} }")
    return out


def seed_spread():
    out = {}
    for ds, run in (("ucihar_subj_group", "grpo_accuracy_ver"), ("ucihar_subj_group", "grpo_verified_ver"),
                    ("cricket_group", "policy_verified"), ("cricket_group", "policy_heuristic")):
        labs, accs = [], []
        for sr in seed_runs(run):
            P = preds(ds, sr)
            if P is None:
                continue
            P = sorted(P, key=lambda p: p["bank_i"])
            labs.append([p["label"] for p in P])
            accs.append(np.mean([p["label"] == p["label_true"] for p in P]))
        L = np.array(labs)
        out[f"{ds}/{run}"] = {"acc": [float(a) for a in accs], "identical_labels_across_seeds": float((L == L[0]).all(0).mean())}
        print(f"spread {ds}/{run}: acc {np.round(accs, 4)}  identical labels {(L == L[0]).all(0).mean():.3f}")
    # do the two LLM rewards give the same labels on UCI-HAR?
    a = sorted(preds("ucihar_subj_group", "grpo_accuracy_ver"), key=lambda p: p["bank_i"])
    v = sorted(preds("ucihar_subj_group", "grpo_verified_ver"), key=lambda p: p["bank_i"])
    s = sorted(preds("ucihar_subj_group", "sft_verified"), key=lambda p: p["bank_i"])
    out["ucihar_rewards_same_labels"] = float(np.mean([x["label"] == y_["label"] for x, y_ in zip(a, v)]))
    out["ucihar_same_as_warm_start"] = float(np.mean([x["label"] == y_["label"] for x, y_ in zip(v, s)]))
    print(f"UCI-HAR: accuracy- and check-reward labels identical on {out['ucihar_rewards_same_labels']:.3f}, "
          f"check reward vs warm start {out['ucihar_same_as_warm_start']:.3f}")
    return out


def neighbours():
    out = {}
    for ds, (ra, rv) in DATA.items():
        z = np.load(A / ds / "features.npz")
        votes, y = z["test_votes"], z["test_y"]
        maj = votes.argmax(1)
        r = {"neighbour_vote_acc": float((maj == y).mean())}
        for run in (ra, rv, "policy_verified"):
            P = preds(ds, run)
            if P is None:
                continue
            lab = np.array([-1 if p["label"] is None else p["label"] for p in sorted(P, key=lambda p: p["bank_i"])])
            r[run] = {"acc": float((lab == y).mean()), "equals_majority": float((lab == maj).mean()),
                      "right_where_majority_wrong": float((lab == y)[maj != y].mean()) if (maj != y).any() else None}
        out[ds] = r
        print(f"neighbours {ds:18s} vote acc {r['neighbour_vote_acc']:.3f}  check-reward LLM = majority on "
              f"{r[rv]['equals_majority']:.3f}, right where majority wrong {r[rv]['right_where_majority_wrong']}")
    return out


def main():
    out = {"coverage": coverage(), "tiers_test": tiers_test(), "multi_pass": multi_pass(), "occl_oracle": occl_oracle(),
           "floor_means": floor_means(), "oracle_acc": oracle_acc(), "seed_spread": seed_spread(),
           "neighbours": neighbours()}
    (ROOT / "results" / "review5_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
