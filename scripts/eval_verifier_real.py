"""Verifier diagnostics on real data, where the true evidence is unknown.

For test series the main oracle classifies correctly, every single-cell claim
(1 window x 1 channel group) is edited once per method (necessity edit towards a
random other class). Reported per method:
  realism      discriminative accuracy of MiniRocket separating edited from
               real series (0.5 = indistinguishable)
  judge_agree  mean Spearman correlation, across a series' claims, between the
               necessity scores given by two different oracles (MultiRocket-Hydra
               and QUANT) on the SAME edits: verdicts that depend on which
               classifier judges them are not evidence about the data
  sharpness    1 - normalised entropy of the positive necessity scores over the
               claims (1 = all necessity in one cell, 0 = spread evenly)
  top_suff     oracle keeps the label when only the top-necessity claim is kept
               and everything else is regenerated (sufficiency of the best claim)
  flip_rate    share of claims whose edit flips the main oracle's label

Usage:
    python scripts/eval_verifier_real.py --dataset pamap2 --n 40 \
        --methods interp splice diffusion:prior+learned hybrid:prior+learned
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.data import load_dataset  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import InpaintDiffusion  # noqa: E402
from veritas.verifier.edits import edit  # noqa: E402
from veritas.verifier.metrics import discriminative_score  # noqa: E402
from veritas.verifier.oracle import Oracle  # noqa: E402

SPLIT = "standard"




def load_oracle(ddir: Path, member: str, X: np.ndarray, y: np.ndarray) -> Oracle:
    path = ddir / f"oracle_{member}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t0 = time.perf_counter()
    oracle = Oracle(members=(member,)).fit(X, y)
    path.write_bytes(pickle.dumps(oracle))
    print(f"oracle {member} fitted in {time.perf_counter() - t0:.0f}s", flush=True)
    return oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--methods", nargs="+",
                    default=["interp", "splice", "diffusion:prior+learned", "hybrid:prior+learned"])
    ap.add_argument("--oracle-max-train", type=int, default=5000)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ddir = ROOT / "results" / "diffusion" / args.dataset / SPLIT
    st = np.load(ddir / "stats.npz")
    data = load_dataset(args.dataset, split=SPLIT, max_test=2000)
    sub = load_dataset(args.dataset, split=SPLIT, max_test=2000, max_train=args.oracle_max_train)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    Xsub = ((sub.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    n_cls = len(data.meta.class_names)
    grid = EvidenceGrid(T=Xtr.shape[-1], channel_groups=data.meta.channel_groups, K=args.K)
    claims = [c for c in grid.claims() if len(c.windows) == 1 and len(c.groups) == 1]
    L = len(claims)

    main_oracle = load_oracle(ddir, "multirocket_hydra", Xsub, sub.y_train)
    judge2 = load_oracle(ddir, "quant", Xsub, sub.y_train)
    p0 = main_oracle.predict_proba(Xte)
    q0 = judge2.predict_proba(Xte)
    correct = np.flatnonzero((p0.argmax(1) == data.y_test) & (q0.argmax(1) == data.y_test))
    print(f"oracle acc main={np.mean(p0.argmax(1) == data.y_test):.3f} "
          f"second={np.mean(q0.argmax(1) == data.y_test):.3f}; both correct on {len(correct)}", flush=True)
    rows = rng.choice(correct, size=min(args.n, len(correct)), replace=False)

    job_row = np.repeat(rows, L)
    job_claim = np.tile(np.arange(L), len(rows))
    masks = np.stack([grid.mask(c) for c in claims])
    job_mask = masks[job_claim]
    y_job = data.y_test[job_row]
    target = (y_job + rng.integers(1, n_cls, size=len(y_job))) % n_cls
    X_job = Xte[job_row]

    results = {"dataset": args.dataset, "n_series": int(len(rows)), "claims_per_series": L,
               "oracle_acc_main": float(np.mean(p0.argmax(1) == data.y_test)),
               "oracle_acc_second": float(np.mean(q0.argmax(1) == data.y_test)),
               "guidance": args.guidance, "strength": args.strength, "methods": {}}
    for spec in args.methods:
        method, _, variant = spec.partition(":")
        diff = None
        if method in ("diffusion", "hybrid"):
            path = ddir / f"{variant}.pt"
            if not path.exists():
                print(f"skip {spec}: {path} missing", flush=True)
                continue
            diff = InpaintDiffusion.load(path)
        er = np.random.default_rng(args.seed + 1)
        t0 = time.perf_counter()
        X_nec = edit(method, X_job, job_mask, target, er, pool=(Xtr, data.y_train), diffusion=diff,
                     guidance=args.guidance, n_steps=args.steps, seed=args.seed, strength=args.strength)
        P = main_oracle.predict_proba(X_nec)
        Q = judge2.predict_proba(X_nec)
        nec_p = (p0[job_row, y_job] - P[np.arange(len(y_job)), y_job]).reshape(len(rows), L)
        nec_q = (q0[job_row, y_job] - Q[np.arange(len(y_job)), y_job]).reshape(len(rows), L)
        flips = (P.argmax(1) != y_job).reshape(len(rows), L)

        agree = [spearmanr(a, b).statistic for a, b in zip(nec_p, nec_q)
                 if np.std(a) > 1e-9 and np.std(b) > 1e-9]
        pos = np.clip(nec_p, 0, None)
        sharp = []
        for s in pos:
            if s.sum() > 1e-9:
                w = s / s.sum()
                ent = -(w[w > 0] * np.log(w[w > 0])).sum() / np.log(L)
                sharp.append(1 - ent)

        # sufficiency of each series' top claim: keep it, regenerate the rest
        top = nec_p.argmax(1)
        keep_mask = ~masks[top]
        X_suf = edit(method, Xte[rows], keep_mask, None, er, pool=(Xtr, data.y_train), diffusion=diff,
                     guidance=1.0, n_steps=args.steps, seed=args.seed + 7, strength=args.strength)
        top_suff = float((main_oracle.predict_proba(X_suf).argmax(1) == data.y_test[rows]).mean())
        t_all = time.perf_counter() - t0

        res = {
            "realism_disc_acc": discriminative_score(X_job, X_nec, job_row, seed=args.seed),
            "judge_agree_spearman": float(np.mean(agree)) if agree else float("nan"),
            "sharpness": float(np.mean(sharp)) if sharp else float("nan"),
            "top_claim_sufficiency": top_suff,
            "flip_rate": float(flips.mean()),
            "mean_necessity": float(nec_p.mean()),
            "seconds": t_all,
        }
        results["methods"][spec] = res
        print(f"{spec:24s} realism={res['realism_disc_acc']:.3f} judge_agree={res['judge_agree_spearman']:.3f} "
              f"sharp={res['sharpness']:.3f} top_suff={top_suff:.3f} flip={res['flip_rate']:.3f} "
              f"| {t_all:.0f}s", flush=True)

    out = ROOT / "results" / "verifier" / f"{args.dataset}_real_eval_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
