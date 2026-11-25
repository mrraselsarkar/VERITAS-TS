"""Does the agent's OWN decision depend on the evidence it cites? (review C1)

The check asks whether the cited evidence drives the oracle's decision. Here the agent itself
is re-run on edited recordings. For every correct answer on the test set:
  cited  : the cited claim is replaced as in Test 1 (hybrid edit towards another class)
  random : a random other claim is replaced in the same way (control). For sensor claims the
           control has the same number of sensors and shares no sensor with the cited claim.
The agent's inputs (cell table and neighbour votes) are recomputed from the edited recording,
and we record how often the agent's class changes. If the evidence explains the agent, the
cited edit should change its decision much more often than the control.

Agents: LLM with accuracy / check reward (GRPO, seed 0) and the non-LLM agent with rule-based /
check-reward claims (seed 0, retrained deterministically and checked against the stored
predictions). The oracle's change rate on the same edits is reported for reference.

Usage:
    python scripts/agent_dependence.py --dataset synth_planted
    python scripts/agent_dependence.py --dataset pamap2_group
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_bank import Ensemble  # noqa: E402
from make_views import NOTES  # noqa: E402
from train_policy import fit, robust_norm  # noqa: E402
from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.views import NeighbourIndex, cell_features, parse_answer, render_prompt  # noqa: E402
from veritas.data import load_dataset  # noqa: E402
from veritas.grid import Claim, EvidenceGrid  # noqa: E402
from veritas.verifier.diffusion import InpaintDiffusion  # noqa: E402
from veritas.verifier.edits import edit  # noqa: E402

LLM_RUNS = {"synth_planted": ("grpo_accuracy_div", "grpo_verified_div")}
DEFAULT_LLM_RUNS = ("grpo_accuracy_ver", "grpo_verified_ver")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synth_planted")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    DS = args.dataset
    A = ROOT / "results" / "agent" / DS
    source, claims_mode = agent_mode(DS)
    rng = np.random.default_rng(args.seed)

    def load_preds(run):
        return sorted(json.loads((A / f"pred_{run}.json").read_text())["preds"], key=lambda p: p["bank_i"])

    ddir = ROOT / "results" / "diffusion" / source / "standard"
    st = np.load(ddir / "stats.npz")
    data = load_dataset(source, max_test=2000)
    Xtr = ((data.X_train - st["mu"]) / st["sd"]).astype(np.float32)
    Xte = ((data.X_test - st["mu"]) / st["sd"]).astype(np.float32)
    grid = EvidenceGrid(T=Xtr.shape[-1], channel_groups=data.meta.channel_groups, K=8)  # agent's cell table
    names, classes, n_cls = grid.group_names, data.meta.class_names, len(data.meta.class_names)
    if claims_mode == "group":
        cgrid = EvidenceGrid(T=Xtr.shape[-1], channel_groups=data.meta.channel_groups, K=1, max_groups=2)
        claim_set = [tuple(sorted(c.groups)) for c in cgrid.claims() if len(c.groups) <= 2]

        def mask(p_window, p_group):
            return cgrid.mask(Claim((0,), tuple(p_group.split("+"))))
    else:
        cgrid = grid

        def mask(p_window, p_group):
            return grid.mask(Claim((p_window,), (p_group,)))
    rows = np.load(ROOT / "results" / "bank" / DS / "test.npz")["rows"]
    X0 = Xte[rows]
    index = NeighbourIndex(Xtr, data.y_train, n_cls)
    z = np.load(A / "features.npz")
    feat0 = cell_features(X0, grid)
    votes0 = index.votes(X0, k=10)
    print(f"sanity: features reproduced {np.allclose(feat0, z['test_feat'], atol=1e-4)}, "
          f"votes reproduced {np.array_equal(votes0, z['test_votes'])}", flush=True)

    diff = InpaintDiffusion.load(ddir / "prior+learned.pt")
    oracle = Ensemble(ddir)

    # ---- agents ----
    agents = {}
    for claim in ("heuristic", "verified"):
        net, ctx = fit(argparse.Namespace(dataset=DS, claim=claim, seed=0, epochs=60))
        share = ctx["share"]

        def run_policy(feat, votes, net=net, share=share):
            F = torch.as_tensor(robust_norm(z["train_feat"], feat), dtype=torch.float32)
            V = torch.as_tensor(share(votes.astype(float)), dtype=torch.float32)
            with torch.no_grad():
                return net(F, V, torch.zeros(len(F), dtype=torch.long))[0].argmax(1).numpy()
        stored = np.array([p["label"] for p in load_preds(f"policy_{claim}")])
        print(f"sanity: non-LLM ({claim}) reproduces stored labels: {(run_policy(feat0, votes0) == stored).mean():.3f}",
              flush=True)
        agents[f"non-LLM, {'rule-based claim' if claim == 'heuristic' else 'check reward'}"] = (f"policy_{claim}", run_policy)

    from peft import PeftModel
    from train_agent import BASE, templated
    from transformers import AutoModelForCausalLM, AutoTokenizer
    runs = LLM_RUNS.get(DS, DEFAULT_LLM_RUNS)
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    model = PeftModel.from_pretrained(base, str(A / "runs" / runs[0]), adapter_name=runs[0])
    model.load_adapter(str(A / "runs" / runs[1]), adapter_name=runs[1])
    model.eval()

    def run_llm(feat, votes, adapter):
        model.set_adapter(adapter)
        labels = []
        for s in range(0, len(feat), 32):
            prompts = [templated(tok, render_prompt(f, v, classes, names, NOTES.get(source, ""), claims=claims_mode))
                       for f, v in zip(feat[s:s + 32], votes[s:s + 32])]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
            for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                lab = parse_answer(t, classes, names, grid.K, claims=claims_mode).label
                labels.append(-1 if lab is None else lab)
        return np.array(labels)

    for run in runs:
        stored = np.array([-1 if p["label"] is None else p["label"] for p in load_preds(run)])
        print(f"sanity: LLM ({run}) reproduces stored labels: {(run_llm(feat0, votes0, run) == stored).mean():.3f}",
              flush=True)
        agents[f"LLM, {'accuracy' if 'accuracy' in run else 'check'} reward"] = (
            run, lambda f, v, run=run: run_llm(f, v, run))

    # ---- edits and re-runs ----
    out = {}
    for label, (run, agent) in agents.items():
        P = load_preds(run)
        ok = [p for p in P if p["label"] == p["label_true"] and p["group"]
              and (claims_mode == "group" or (p["window"] is not None and p["group"] in names))]
        i = np.array([p["bank_i"] for p in ok])
        y = np.array([p["label"] for p in ok])
        cited, other = [], []
        for p in ok:
            cited.append(mask(p["window"], p["group"]))
            if claims_mode == "group":
                gs = set(p["group"].split("+"))
                cand = [c for c in claim_set if len(c) == len(gs) and not gs & set(c)]
                cand = cand or [c for c in claim_set if not gs & set(c)]
                c = cand[rng.integers(len(cand))]
                other.append(mask(0, "+".join(c)))
            else:
                w, g = p["window"], names.index(p["group"])
                cells = [(ww, gg) for ww in range(grid.K) for gg in range(len(names)) if (ww, gg) != (w, g)]
                ww, gg = cells[rng.integers(len(cells))]
                other.append(mask(ww, names[gg]))
        res = {"n": len(ok)}
        for kind, masks in (("cited", cited), ("random", other)):
            M = np.stack(masks)
            target = (y + rng.integers(1, n_cls, size=len(y))) % n_cls
            Xe = edit("hybrid", X0[i], M, target, rng, pool=(Xtr, data.y_train), diffusion=diff,
                      guidance=2.0, n_steps=25, strength=0.5, seed=args.seed, batch_size=128)
            fe, ve = cell_features(Xe, grid), index.votes(Xe, k=10)
            agent_lab = agent(fe, ve)
            orc_lab = oracle.predict_proba(Xe).argmax(1)
            res[kind] = {"agent_changed": float((agent_lab != y).mean()),
                         "oracle_changed": float((orc_lab != y).mean())}
        out[label] = res
        print(label, json.dumps(res), flush=True)
    dest = ROOT / "results" / ("agent_dependence.json" if DS == "synth_planted" else f"agent_dependence_{DS}.json")
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
