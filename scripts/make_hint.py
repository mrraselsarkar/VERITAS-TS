"""Classifier hint for the agents' text view: cross-fitted InceptionTime class probabilities.

The agents classify from a short text view, and on real data their class mostly
follows the 10-neighbour vote in that view, which is weaker than a standard
classifier. This script gives the view a stronger hint: the class probabilities
of InceptionTime (Ismail Fawaz et al., 2020), a classifier that is neither the
oracle of the check nor the independent judge.

No training recording receives a hint from a network that saw it. The training
split uses the stratified folds of the cross-fitted oracles (train_oracles_xf.py):
network k is trained on the other folds (at most 5000 recordings) and gives the
probabilities of the recordings in fold k. A test recording receives the mean of
the five networks, so training and test hints come from equally strong models.

Writes results/hint/<experiment>.npz with rows / prob for the bank rows of each part.

Usage:
    python scripts/make_hint.py --dataset ptbxl_group
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from classifier_baselines import InceptionNet  # noqa: E402
from veritas.agent.config import agent_mode  # noqa: E402
from veritas.data import load_dataset  # noqa: E402
from veritas.data.base import stratified_subsample  # noqa: E402


def fit_proba(Xtr, ytr, n_cls, epochs=60, seed=0):
    """InceptionTime as in classifier_baselines.fit_inception, returning softmax probabilities."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ks = int(min(40, max(8, Xtr.shape[-1] // 4)))
    net = InceptionNet(Xtr.shape[1], n_cls, ks=ks).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, y = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(X))
        for s in range(0, len(X), 64):
            b = torch.as_tensor(perm[s:s + 64])
            if len(b) < 2:  # BatchNorm needs two recordings
                continue
            loss = nn.functional.cross_entropy(net(X[b].cuda()), y[b].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    def proba(Xn):
        with torch.no_grad():
            return np.concatenate([torch.softmax(net(torch.as_tensor(Xn[s:s + 256], dtype=torch.float32,
                                                                     device="cuda")), 1).cpu().numpy()
                                   for s in range(0, len(Xn), 256)])
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="agent experiment, e.g. ptbxl_group")
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    source, _ = agent_mode(args.dataset)
    ddir = ROOT / "results" / "diffusion" / source / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(source, split="standard", max_test=2000)  # the rows the banks index
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    folds = np.load(ddir / "xf" / "folds.npz")
    F, train_fold = int(folds["F"]), folds["train_fold"]
    assert len(train_fold) == len(data.y_train), "fold ids do not match the training split"
    n_cls = len(data.meta.class_names)
    rows = {p: np.load(ROOT / "results" / "bank" / args.dataset / f"{p}.npz")["rows"] for p in ("train", "test")}
    prob_tr = np.full((len(rows["train"]), n_cls), np.nan, dtype=np.float32)
    prob_te = np.zeros((len(rows["test"]), n_cls), dtype=np.float32)
    log = {}
    for k in range(F):
        t0 = time.perf_counter()
        idx = np.flatnonzero(train_fold != k)
        if len(idx) > args.max_train:
            idx = stratified_subsample(idx, data.y_train[idx], args.max_train, k)
        proba = fit_proba(Xtr[idx], data.y_train[idx], n_cls, epochs=args.epochs, seed=k)
        mine = train_fold[rows["train"]] == k
        prob_tr[mine] = proba(Xtr[rows["train"][mine]])
        prob_te += proba(Xte[rows["test"]]) / F
        acc_k = float((prob_tr[mine].argmax(1) == data.y_train[rows["train"][mine]]).mean())
        log[k] = {"n_train": int(len(idx)), "acc_own_fold": acc_k, "seconds": time.perf_counter() - t0}
        print(f"fold {k}: {len(idx)} recordings, accuracy on its fold's bank rows {acc_k:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    assert not np.isnan(prob_tr).any()
    acc_tr = float((prob_tr.argmax(1) == data.y_train[rows["train"]]).mean())
    acc_te = float((prob_te.argmax(1) == data.y_test[rows["test"]]).mean())
    print(f"{args.dataset}: out-of-fold training accuracy {acc_tr:.3f}, test accuracy {acc_te:.3f}", flush=True)
    out = ROOT / "results" / "hint"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{args.dataset}.npz", train_rows=rows["train"], train_prob=prob_tr,
                        test_rows=rows["test"], test_prob=prob_te)
    log.update(acc_train_oof=acc_tr, acc_test=acc_te)
    (out / f"{args.dataset}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
