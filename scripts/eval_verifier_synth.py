"""RQ2/RQ3 on synthetic data: can region edits tell the planted evidence from wrong claims?

Datasets: synth_planted (local evidence) and synth_context (evidence judged
relative to the rest of the series, where a context-blind splice should fail).

For test series the oracle classifies correctly, four claims of equal size
(1 window x 1 channel group) are tested:
  true      the planted (window, group)
  group     same window, another group
  window    same group, adjacent window
  random    any other (window, group)
For every edit method:
  necessity   m edits of the claimed region towards random other classes;
              score = drop in oracle p(true class)
  sufficiency m edits of everything except the claim (unconditional);
              score = oracle p(true class) afterwards
  AUC         true vs each false-claim type, per score and for their product
  realism     discriminative score of necessity edits: accuracy of a MiniRocket
              classifier separating edited from real series (0.5 = indistinguishable),
              and within-group channel correlation inside the edited region vs real
  validity    share of necessity edits of the true region that the oracle assigns
              to the requested target class

Usage:
    python scripts/eval_verifier_synth.py --dataset synth_context --n 100 --m 2 \
        --methods splice diffusion:prior+learned hybrid:prior+learned
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import load_dataset  # noqa: E402
from veritas.grid import Claim, EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import InpaintDiffusion  # noqa: E402
from veritas.verifier.edits import edit  # noqa: E402
from veritas.verifier.metrics import discriminative_score  # noqa: E402
from veritas.verifier.oracle import Oracle  # noqa: E402

SPLIT = "standard"


def claims_for(ev_w: int, ev_g: int, names: list[str], K: int, rng) -> dict[str, Claim]:
    g = names[ev_g]
    other_g = names[(ev_g + rng.integers(1, len(names))) % len(names)]
    adj_w = ev_w + 1 if ev_w + 1 < K else ev_w - 1
    while True:
        rw, rg = rng.integers(K), names[rng.integers(len(names))]
        if (rw, rg) != (ev_w, g):
            break
    return {"true": Claim((ev_w,), (g,)), "group": Claim((ev_w,), (other_g,)),
            "window": Claim((adj_w,), (g,)), "random": Claim((int(rw),), (rg,))}




def group_corr(X: np.ndarray, M: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """Mean within-group channel correlation inside each row's masked span."""
    out = []
    for b in range(len(X)):
        vals = []
        for chans in groups.values():
            if len(chans) < 2 or not M[b, chans].any():
                continue
            span = M[b, chans[0]]
            a, c = X[b, chans[0], span], X[b, chans[1], span]
            if a.std() > 1e-6 and c.std() > 1e-6:
                vals.append(np.corrcoef(a, c)[0, 1])
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def nanmean(a: np.ndarray) -> float:
    """Mean ignoring NaN; NaN (not a warning) when nothing is defined, e.g. zero edits."""
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synth_planted", choices=["synth_planted", "synth_context"])
    ap.add_argument("--n", type=int, default=150, help="test series")
    ap.add_argument("--m", type=int, default=3, help="edits per claim and test")
    ap.add_argument("--methods", nargs="+",
                    default=["zero", "noise", "interp", "splice", "diffusion:prior+learned", "diffusion:none"])
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=0.5, help="hybrid: SDEdit noise level")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default="", help="suffix for the results file, so extra runs do not overwrite")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    DDIR = ROOT / "results" / "diffusion" / args.dataset / SPLIT

    data = load_dataset(args.dataset, split=SPLIT, max_test=None)
    st = np.load(DDIR / "stats.npz")
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    names = sorted(data.meta.channel_groups)
    grid = EvidenceGrid(T=Xtr.shape[-1], channel_groups=data.meta.channel_groups, K=8)
    n_cls = len(data.meta.class_names)

    oracle_path = DDIR / "oracle_multirocket_hydra.pkl"
    if oracle_path.exists():
        oracle = pickle.loads(oracle_path.read_bytes())
    else:
        t0 = time.perf_counter()
        oracle = Oracle().fit(Xtr, data.y_train)
        oracle_path.write_bytes(pickle.dumps(oracle))
        print(f"oracle fitted in {time.perf_counter() - t0:.0f}s", flush=True)

    p0 = oracle.predict_proba(Xte)
    correct = np.flatnonzero(p0.argmax(1) == data.y_test)
    print(f"oracle test accuracy {len(correct) / len(Xte):.3f}", flush=True)
    rows = rng.choice(correct, size=min(args.n, len(correct)), replace=False)

    # build all (row, claim type, repeat) jobs once; every method edits the same jobs
    kinds = ["true", "group", "window", "random"]
    job_row, job_kind, job_mask = [], [], []
    for r in rows:
        cl = claims_for(int(data.groups_test["ev_window"][r]), int(data.groups_test["ev_group"][r]),
                        names, grid.K, rng)
        for k in kinds:
            for _ in range(args.m):
                job_row.append(r)
                job_kind.append(k)
                job_mask.append(grid.mask(cl[k]))
    job_row, job_kind, job_mask = np.array(job_row), np.array(job_kind), np.stack(job_mask)
    y_job = data.y_test[job_row]
    # necessity targets: a random class different from the true one
    target = (y_job + rng.integers(1, n_cls, size=len(y_job))) % n_cls
    X_job = Xte[job_row]
    p_orig = p0[job_row, y_job]

    results = {"n_series": int(len(rows)), "m": args.m, "oracle_test_acc": float(len(correct) / len(Xte)),
               "guidance": args.guidance, "steps": args.steps, "strength": args.strength,
               "dataset": args.dataset, "methods": {}}
    real_corr = nanmean(group_corr(Xte[rows], np.stack(
        [grid.mask(Claim((int(data.groups_test["ev_window"][r]),), (names[int(data.groups_test["ev_group"][r])],)))
         for r in rows]), data.meta.channel_groups))

    for spec in args.methods:
        method, _, variant = spec.partition(":")
        diff = None
        if method in ("diffusion", "hybrid"):
            path = DDIR / f"{variant}.pt"
            if not path.exists():
                print(f"skip {spec}: {path} missing", flush=True)
                continue
            diff = InpaintDiffusion.load(path)
        t0 = time.perf_counter()
        er = np.random.default_rng(args.seed + 1)
        X_nec = edit(method, X_job, job_mask, target, er, pool=(Xtr, data.y_train), diffusion=diff,
                     guidance=args.guidance, n_steps=args.steps, seed=args.seed, strength=args.strength)
        X_suf = edit(method, X_job, ~job_mask, None, er, pool=(Xtr, data.y_train), diffusion=diff,
                     guidance=1.0, n_steps=args.steps, seed=args.seed + 7, strength=args.strength)
        t_edit = time.perf_counter() - t0
        P_nec = oracle.predict_proba(X_nec)
        p_nec = P_nec[np.arange(len(y_job)), y_job]
        p_suf = oracle.predict_proba(X_suf)[np.arange(len(y_job)), y_job]

        # aggregate the m repeats per (row, kind)
        key = np.array([f"{r}|{k}" for r, k in zip(job_row, job_kind)])
        uniq, inv = np.unique(key, return_inverse=True)
        nec = np.bincount(inv, weights=p_orig - p_nec) / np.bincount(inv)
        suf = np.bincount(inv, weights=p_suf) / np.bincount(inv)
        kind_u = np.array([u.split("|")[1] for u in uniq])
        both = np.clip(nec, 0, None) * suf

        res = {"edit_seconds": t_edit,
               # per (recording, claim type) check scores, for bootstrap intervals over recordings
               "scores": {"row": [int(u.split("|")[0]) for u in uniq], "kind": kind_u.tolist(),
                          "combined": both.tolist()}}
        for score_name, s in (("necessity", nec), ("sufficiency", suf), ("combined", both)):
            res[score_name] = {k: float(np.mean(s[kind_u == k])) for k in kinds}
            res[f"auc_{score_name}"] = {
                k: float(roc_auc_score(np.r_[np.ones((kind_u == "true").sum()), np.zeros((kind_u == k).sum())],
                                       np.r_[s[kind_u == "true"], s[kind_u == k]]))
                for k in kinds[1:]
            }
        true_rows = job_kind == "true"
        # counterfactual validity: does a necessity edit of the TRUE region become the target class?
        res["validity_true"] = float((P_nec[true_rows].argmax(1) == target[true_rows]).mean())
        res["discriminative_acc"] = discriminative_score(X_job[true_rows], X_nec[true_rows],
                                                    job_row[true_rows], seed=args.seed)
        res["group_corr_edited"] = float(nanmean(group_corr(X_nec[true_rows], job_mask[true_rows],
                                                               data.meta.channel_groups)))
        res["group_corr_real"] = float(real_corr)
        results["methods"][spec] = res
        print(f"{spec:24s} AUC(combined) group={res['auc_combined']['group']:.3f} "
              f"window={res['auc_combined']['window']:.3f} random={res['auc_combined']['random']:.3f} | "
              f"nec true={res['necessity']['true']:.3f} suf true={res['sufficiency']['true']:.3f} | "
              f"disc acc={res['discriminative_acc']:.3f} corr {res['group_corr_edited']:.2f} "
              f"(real {real_corr:.2f}) | valid {res['validity_true']:.3f} | {t_edit:.0f}s", flush=True)

    out = ROOT / "results" / "verifier" / f"{args.dataset}_eval_seed{args.seed}{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
