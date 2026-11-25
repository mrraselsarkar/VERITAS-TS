"""Accuracy of standard time series classifiers vs the agents (review: accuracy comparison).

All models are evaluated on the same test recordings as the agents (the bank's test rows)
and trained on the same training data as the oracle (standardised, at most 5000 recordings).

  MiniRocket      aeon MiniRocketClassifier
  MultiRocket-Hydra, QUANT, oracle (their average), ResNet: the stored oracle members / judge
  InceptionTime   one Inception network (Ismail Fawaz et al., 2020), PyTorch re-implementation
  agents          LLM and non-LLM with the check reward (mean over seeds, from seeds_summary.json)

Usage:
    python scripts/classifier_baselines.py --datasets synth_planted pamap2_group ptbxl_group ucihar_subj_group
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.data import load_dataset  # noqa: E402


class _Inception(nn.Module):
    def __init__(self, cin: int, nf: int = 32, ks: int = 40):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        c = nf if cin > 1 else cin
        ks = [k if k % 2 else k - 1 for k in (ks, ks // 2, ks // 4)]
        self.convs = nn.ModuleList([nn.Conv1d(c, nf, k, padding=k // 2, bias=False) for k in ks])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)

    def forward(self, x):
        h = self.bottleneck(x)
        return torch.relu(self.bn(torch.cat([c(h) for c in self.convs] + [self.pool(x)], 1)))


class InceptionNet(nn.Module):
    def __init__(self, C: int, n_cls: int, depth: int = 6, nf: int = 32, ks: int = 40):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        cin, res_in = C, C
        for d in range(depth):
            self.blocks.append(_Inception(cin, nf, ks))
            cin = 4 * nf
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, cin, 1, bias=False), nn.BatchNorm1d(cin)))
                res_in = cin
        self.head = nn.Linear(cin, n_cls)

    def forward(self, x):
        res, j = x, 0
        for d, b in enumerate(self.blocks):
            x = b(x)
            if d % 3 == 2:
                x = torch.relu(x + self.shortcuts[j](res))
                res, j = x, j + 1
        return self.head(x.mean(-1))


def fit_inception(Xtr, ytr, n_cls, epochs=60, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    ks = int(min(40, max(8, Xtr.shape[-1] // 4)))
    net = InceptionNet(Xtr.shape[1], n_cls, ks=ks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, y = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(X))
        for s in range(0, len(X), 64):
            b = torch.as_tensor(perm[s:s + 64])
            loss = nn.functional.cross_entropy(net(X[b].to(dev)), y[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            return np.concatenate([net(torch.as_tensor(Xn[s:s + 256], dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
                                   for s in range(0, len(Xn), 256)])
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth_planted", "pamap2_group", "ptbxl_group", "ucihar_subj_group"])
    args = ap.parse_args()
    seeds = json.loads((ROOT / "results" / "seeds_summary.json").read_text())
    out_path = ROOT / "results" / "classifier_baselines.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ds in args.datasets:
        source = agent_mode(ds)[0]
        ddir = ROOT / "results" / "diffusion" / source / "standard"
        st = np.load(ddir / "stats.npz")
        data = load_dataset(source, split="standard", max_test=2000, max_train=5000)
        Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
        Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
        rows = np.load(ROOT / "results" / "bank" / ds / "test.npz")["rows"]
        X, y = Xte[rows], data.y_test[rows]
        n_cls = len(data.meta.class_names)
        res = {}
        from aeon.classification.convolution_based import MiniRocketClassifier
        res["MiniRocket"] = float((MiniRocketClassifier(random_state=0).fit(Xtr.astype(np.float64), data.y_train)
                                   .predict(X.astype(np.float64)) == y).mean())
        probs = {}
        for m, name in (("multirocket_hydra", "MultiRocket-Hydra"), ("quant", "QUANT"), ("cnn", "ResNet")):
            f = ddir / f"oracle_{m}.pkl"
            if f.exists():
                model = pickle.loads(f.read_bytes())
                Xin = X if m == "cnn" else X.astype(np.float64)
                probs[m] = model.predict_proba(Xin)
                res[name] = float((probs[m].argmax(1) == y).mean())
        res["Oracle (MR-Hydra + QUANT)"] = float(((probs["multirocket_hydra"] + probs["quant"]).argmax(1) == y).mean())
        res["InceptionTime"] = float((fit_inception(Xtr, data.y_train, n_cls)(X) == y).mean())
        s = seeds.get(ds, {})
        for k, name in (("GRPO check reward", "LLM agent, check reward"), ("non-LLM, check reward", "Non-LLM agent, check reward")):
            if k in s:
                res[name] = s[k]["accuracy"]["mean"]
        out[ds] = res
        print(ds, {k: round(100 * v, 1) for k, v in res.items()}, flush=True)
        out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
