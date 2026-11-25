"""Standard explanation methods on the planted benchmark, scored like the agents.

Each method ranks the 24 cells (window x group) of a test recording and its top
cell is its explanation. Reported per method, over the recordings its model
classifies correctly:
  true-evidence rate   top cell is the planted cell
  verified rate        top cell passes the check (main counterfactual bank)

Methods (gradient-based ones use the differentiable CNN judge):
  grad_x_input        |gradient x input| summed over the cell
  integrated_grad     Integrated Gradients from an all-zero baseline (32 steps)
  occlusion_zero      drop in the CNN's probability when the cell is set to zero
  native_guide_swap   drop when the cell is copied from the nearest training
                      recording of another class (Native Guide style)
  heuristic           most anomalous cell of the cell table (no model)
  checker_only        best cell by the checker's own score (main oracle)

Usage:
    python scripts/attribution_baselines.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.views import heuristic_claim  # noqa: E402
from veritas.data import load_dataset  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402

DS = "synth_planted"


def cell_sum(A: np.ndarray, grid: EvidenceGrid) -> np.ndarray:
    """(n, C, T) attribution -> (n, K*G) summed |attribution| per cell, (window, group) order."""
    names = grid.group_names
    out = np.zeros((len(A), grid.K * len(names)))
    for k in range(grid.K):
        t0, t1 = grid.window_bounds(k)
        for g, nm in enumerate(names):
            out[:, k * len(names) + g] = np.abs(A[:, grid.channel_groups[nm], t0:t1]).sum((1, 2))
    return out


def main():
    ddir = ROOT / "results" / "diffusion" / DS / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(DS, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    y, ytr = data.y_test, data.y_train
    grid = EvidenceGrid(T=Xte.shape[-1], channel_groups=data.meta.channel_groups, K=8)
    names, G = grid.group_names, len(grid.group_names)
    truth = data.groups_test["ev_window"] * G + data.groups_test["ev_group"]
    judge = pickle.loads((ddir / "oracle_cnn.pkl").read_bytes())
    net, dev = judge.model.eval(), judge.device
    bank = Bank(ROOT / "results" / "bank" / DS / "test.npz")
    cell_ids = [bank.claim_id(w, names[g]) for w in range(grid.K) for g in range(G)]

    p_orig = judge.predict_proba(Xte)
    cnn_ok = p_orig.argmax(1) == y
    p_true = p_orig[np.arange(len(y)), y]
    scores = {}

    # gradient x input and Integrated Gradients
    gi = np.zeros_like(Xte)
    ig = np.zeros_like(Xte)
    for s in range(0, len(Xte), 64):
        xb = torch.as_tensor(Xte[s:s + 64], device=dev)
        yb = torch.as_tensor(y[s:s + 64], device=dev)
        xg = xb.clone().requires_grad_(True)
        net(xg).gather(1, yb[:, None]).sum().backward()
        gi[s:s + 64] = (xg.grad * xb).detach().cpu().numpy()
        total = torch.zeros_like(xb)
        for a in np.linspace(1 / 32, 1, 32):
            xa = (a * xb).clone().requires_grad_(True)
            net(xa).gather(1, yb[:, None]).sum().backward()
            total += xa.grad
        ig[s:s + 64] = (xb * total / 32).detach().cpu().numpy()
    scores["grad_x_input"] = cell_sum(gi, grid)
    scores["integrated_grad"] = cell_sum(ig, grid)

    # occlusion with zeros, and Native-Guide-style swap from the nearest unlike neighbour
    flat_tr = Xtr.reshape(len(Xtr), -1)
    nun = np.zeros(len(Xte), dtype=int)
    for i in range(len(Xte)):
        other = np.flatnonzero(ytr != y[i])
        d = ((flat_tr[other] - Xte[i].ravel()) ** 2).sum(1)
        nun[i] = other[d.argmin()]
    occ = np.zeros((len(Xte), grid.K * G))
    swap = np.zeros_like(occ)
    for k in range(grid.K):
        t0, t1 = grid.window_bounds(k)
        for g, nm in enumerate(names):
            ch = grid.channel_groups[nm]
            Xz, Xs = Xte.copy(), Xte.copy()
            Xz[:, ch, t0:t1] = 0.0
            Xs[:, ch, t0:t1] = Xtr[nun][:, ch, t0:t1]
            j = k * G + g
            occ[:, j] = p_true - judge.predict_proba(Xz)[np.arange(len(y)), y]
            swap[:, j] = p_true - judge.predict_proba(Xs)[np.arange(len(y)), y]
    scores["occlusion_zero"] = occ
    scores["native_guide_swap"] = swap

    # heuristic and the checker on its own (main oracle, continuous score for the true label)
    feat = np.load(ROOT / "results" / "agent" / DS / "features.npz")["test_feat"]
    heur = np.zeros_like(occ)
    for i, f in enumerate(feat):
        w, g, _ = heuristic_claim(f)
        heur[i, w * G + g] = 1.0
    scores["heuristic"] = heur
    z = np.load(ROOT / "results" / "bank" / DS / "test.npz")
    p0 = z["p0"][np.arange(len(y)), y]
    chk = (np.clip(p0[:, None] - z["nec_ptrue"].mean(-1), 0, None) * z["suf_ptrue"].mean(-1))[:, cell_ids]
    scores["checker_only"] = chk

    oracle_ok = z["p0"].argmax(1) == y
    res = {}
    for name, S in scores.items():
        top = S.argmax(1)
        ok = oracle_ok if name in ("checker_only", "heuristic") else cnn_ok
        ver = np.array([bank.verified(i, cell_ids[c], int(y[i])) for i, c in enumerate(top)])
        res[name] = {"n_correct": int(ok.sum()),
                     "true_evidence": float((top == truth)[ok].mean()),
                     "verified": float(ver[ok].mean())}
        print(f"{name:18s} true evidence {res[name]['true_evidence']:.3f}  verified {res[name]['verified']:.3f}"
              f"  (n = {ok.sum()})", flush=True)
    out = ROOT / "results" / "agent" / DS / "attribution_baselines.json"
    out.write_text(json.dumps({"cnn_test_accuracy": float(cnn_ok.mean()), "methods": res}, indent=2))
    print(f"CNN judge accuracy {cnn_ok.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
