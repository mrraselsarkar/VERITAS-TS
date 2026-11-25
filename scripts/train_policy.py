"""Non-LLM policy over the same evidence grid (proposal RQ5, ablation "drop the LLM").

A small network reads exactly what the LLM reads (cell features + neighbour
votes) and outputs a label and one cell:
  label head  supervised cross-entropy on the true label
  claim head  scores every cell given the label; trained with REINFORCE on the
              same verified-evidence reward as the LLM (--claim verified), or
              copies the heuristic cell (--claim heuristic, the analogue of
              accuracy-only training, where nothing teaches the evidence)

Writes results/agent/<dataset>/pred_policy_<claim>.json in the LLM's format.

Usage:
    python scripts/train_policy.py --dataset synth_planted --claim verified
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.views import heuristic_claim, ranked_groups  # noqa: E402
from veritas.data.base import load_processed  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402


class Policy(nn.Module):
    """Cell claims: every cell is scored from its own embedding. Group claims
    (n_claims given): the claims are scored jointly from the pooled embedding."""

    def __init__(self, K: int, G: int, F: int, n_cls: int, H: int = 64, n_claims: int | None = None,
                 vdim: int | None = None):
        super().__init__()
        self.K, self.G = K, G
        self.pos = nn.Parameter(torch.randn(K * G, H) * 0.1)
        self.cell = nn.Sequential(nn.Linear(F, H), nn.GELU(), nn.Linear(H, H))
        # vdim: neighbour shares (n_cls), plus the classifier hint (n_cls) when the view has one
        self.label_head = nn.Sequential(nn.Linear(H + (vdim or n_cls), H), nn.GELU(), nn.Linear(H, n_cls))
        self.lab_emb = nn.Embedding(n_cls, H)
        self.claim_head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, n_claims or 1))
        self.pooled = n_claims is not None

    def forward(self, feat, votes, label):
        h = self.cell(feat.flatten(1, 2)) + self.pos              # (B, K*G, H)
        logits_y = self.label_head(torch.cat([h.mean(1), votes], 1))
        if self.pooled:
            logits_c = self.claim_head(h.mean(1) + self.lab_emb(label))
        else:
            logits_c = self.claim_head(h + self.lab_emb(label)[:, None]).squeeze(-1)
        return logits_y, logits_c


def side_inputs(z, part: str, share) -> np.ndarray:
    """Neighbour-vote shares, followed by the classifier hint when the view has one (make_views --hint-from)."""
    v = share(z[f"{part}_votes"])
    return np.concatenate([v, z[f"{part}_hint"]], 1) if f"{part}_hint" in z else v


def robust_norm(feat_tr, feat):
    med = np.median(feat_tr.reshape(-1, feat_tr.shape[-1]), 0)
    mad = np.median(np.abs(feat_tr.reshape(-1, feat_tr.shape[-1]) - med), 0) + 1e-3
    return (feat - med) / mad


def fit(args):
    """Train the policy; returns the network and what is needed to run it on new inputs."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    source, claims = agent_mode(args.dataset)
    meta = load_processed(source).meta
    grid = EvidenceGrid(T=8, channel_groups=meta.channel_groups, K=8)
    names = grid.group_names
    z = np.load(ROOT / "results" / "agent" / args.dataset / "features.npz")
    bank = Bank(ROOT / "results" / "bank" / f"{args.dataset}{getattr(args, 'bank_tag', '')}" / "train.npz")
    ftr, fte = robust_norm(z["train_feat"], z["train_feat"]), robust_norm(z["train_feat"], z["test_feat"])
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    share = lambda v: v / v.sum(1, keepdims=True)  # noqa: E731  neighbour votes as shares
    Ftr, Vtr, Ytr = T(ftr), T(side_inputs(z, "train", share)), torch.as_tensor(z["train_y"])
    Fte, Vte = T(fte), T(side_inputs(z, "test", share))
    if getattr(args, "no_votes", False):  # ablation: the agent sees the cell table only, not the neighbour classes
        Vtr, Vte = torch.zeros_like(Vtr), torch.zeros_like(Vte)
    K, G, F = ftr.shape[1:]
    if claims == "group":  # actions = the bank's claims, in bank order
        keys = sorted(bank.index, key=bank.index.get)               # [(0, "a+b"), ...]
        action_ids = list(range(len(keys)))
        heur = torch.as_tensor([bank.claim_id(0, "+".join(ranked_groups(f, names)[0][0]))
                                for f in z["train_feat"]])
        to_claim = lambda c: keys[c]  # noqa: E731
    else:  # actions = cells in (window, group) order
        action_ids = [bank.claim_id(w, names[g]) for w in range(K) for g in range(G)]
        heur = torch.as_tensor([(lambda w, g, _: w * G + g)(*heuristic_claim(f)) for f in z["train_feat"]])
        to_claim = lambda c: (c // G, names[c % G])  # noqa: E731
    # reward table: verified[i, action] for the true label
    verified = torch.as_tensor(np.stack([bank.verified_cells(i, int(bank.y[i]))[action_ids]
                                         for i in range(len(bank))]), dtype=torch.float32)

    net = Policy(K, G, F, len(meta.class_names), n_claims=len(action_ids) if claims == "group" else None,
                 vdim=Vtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(Ytr)
    for ep in range(args.epochs):
        perm = torch.as_tensor(rng.permutation(n))
        tot_r = 0.0
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            ly, lc = net(Ftr[b], Vtr[b], Ytr[b])
            loss = nn.functional.cross_entropy(ly, Ytr[b])
            if args.claim == "heuristic":
                loss = loss + nn.functional.cross_entropy(lc, heur[b])
            else:  # REINFORCE with a mean baseline on the verified-evidence reward
                dist = torch.distributions.Categorical(logits=lc)
                a = dist.sample()
                r = verified[b, a]
                loss = loss - ((r - r.mean()) * dist.log_prob(a)).mean() - 0.01 * dist.entropy().mean()
                tot_r += r.sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 and args.claim == "verified":
            print(f"epoch {ep + 1}: mean verified reward {tot_r / n:.3f}", flush=True)
    net.eval()
    return net, {"z": z, "Fte": Fte, "Vte": Vte, "to_claim": to_claim, "share": share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--claim", choices=["verified", "heuristic"], default="verified")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the prediction file, e.g. _s1 for another seed")
    ap.add_argument("--no-votes", action="store_true", help="ablation: drop the neighbour classes from the input")
    ap.add_argument("--bank-tag", default="", help="reward bank directory suffix, e.g. _xf for the stricter check")
    ap.add_argument("--also-test", nargs="*", default=[],
                    help="further agent experiments (e.g. <dataset>_shift) whose test views the trained policy answers")
    ap.add_argument("--run-name", default=None, help="name of the prediction file (default: policy_<claim><tag>)")
    args = ap.parse_args()
    net, ctx = fit(args)
    run = args.run_name or f"policy_{args.claim}{args.tag}"
    for name in [args.dataset] + args.also_test:
        if name == args.dataset:
            Fte, Vte = ctx["Fte"], ctx["Vte"]
        else:  # another test set, normalised with the training features
            zo = np.load(ROOT / "results" / "agent" / name / "features.npz")
            Fte = torch.as_tensor(robust_norm(ctx["z"]["train_feat"], zo["test_feat"]), dtype=torch.float32)
            Vte = torch.as_tensor(side_inputs(zo, "test", ctx["share"]), dtype=torch.float32)
            if args.no_votes:
                Vte = torch.zeros_like(Vte)
        predict_and_write(net, Fte, Vte, ctx["to_claim"], name, run)


def predict_and_write(net, Fte, Vte, to_claim, name: str, run: str):
    with torch.no_grad():
        ly, _ = net(Fte, Vte, torch.zeros(len(Fte), dtype=torch.long))
        pred = ly.argmax(1)
        _, lc = net(Fte, Vte, pred)
        cell = lc.argmax(1)
    recs = [json.loads(line) for line in
            (ROOT / "results" / "agent" / name / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    preds = [{"bank_i": r["bank_i"], "label_true": r["label"], "text": "", "label": int(p),
              "window": to_claim(int(c))[0], "group": to_claim(int(c))[1], "prop": None}
             for r, p, c in zip(recs, pred.tolist(), cell.tolist())]
    dest = ROOT / "results" / "agent" / name / f"pred_{run}.json"
    dest.write_text(json.dumps({"run": run, "dataset": name, "preds": preds}, indent=1))
    acc = np.mean([p["label"] == p["label_true"] for p in preds])
    print(f"{run} on {name}: accuracy {acc:.3f} -> {dest}")


if __name__ == "__main__":
    main()
