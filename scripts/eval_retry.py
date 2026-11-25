"""Verify-and-revise at test time (proposal, Method part 5: retry up to k times, then abstain).

Round 1 is the agent's greedy answer. If its claim fails verification (bank lookup
for the predicted label), the agent is told so in a follow-up turn and answers
again (sampled), up to --k attempts. The first verified answer is accepted;
otherwise the system abstains. For every k <= --k we report coverage (share of
series answered), error among answered series, and accuracy of the answers.

Writes results/agent/<dataset>/retry_<run>.json.

Usage:
    python scripts/eval_retry.py --dataset synth_planted --run grpo_verified_div --k 4
    python scripts/eval_retry.py --dataset synth_planted_shift --run grpo_verified_div --adapters-from synth_planted
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_agent import BASE, load_records  # noqa: E402
from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.reward import Bank  # noqa: E402
from veritas.agent.views import parse_answer  # noqa: E402
from veritas.data.base import load_processed  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402

FEEDBACK = ("Your evidence failed the check: changing that part of the signal did not change the "
            "decision as it should, or that part alone was not enough to keep it. Answer again in "
            "the same format. Choose different evidence, or a different class if the evidence for "
            "your class is weak.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapters-from", default=None)
    ap.add_argument("--k", type=int, default=4, help="maximum attempts per series")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    src = args.adapters_from or args.dataset
    model = PeftModel.from_pretrained(model, str(ROOT / "results" / "agent" / src / "runs" / args.run))
    model = model.merge_and_unload().eval()

    source, claims = agent_mode(args.dataset)
    meta = load_processed(source).meta
    grid = EvidenceGrid(T=8, channel_groups=meta.channel_groups, K=8)
    bank = Bank(ROOT / "results" / "bank" / args.dataset / "test.npz")
    recs = load_records(args.dataset, "test")
    chats = [[{"role": "user", "content": r["prompt"]}] for r in recs]
    attempts = [[] for _ in recs]
    open_ = list(range(len(recs)))
    t0 = time.perf_counter()
    for rnd in range(args.k):
        for s in range(0, len(open_), args.batch):
            idx = open_[s:s + args.batch]
            texts = [tok.apply_chat_template(chats[j], add_generation_prompt=True, enable_thinking=False,
                                             tokenize=False) for j in idx]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            sample = rnd > 0
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=48, do_sample=sample,
                                     temperature=args.temperature if sample else None,
                                     top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, text in zip(idx, outs):
                a = parse_answer(text, meta.class_names, grid.group_names, grid.K, claims=claims)
                i = recs[j]["bank_i"]
                ok = bool(bank.verified(i, bank.claim_id(a.window, a.group), a.label))
                attempts[j].append({"text": text, "label": a.label, "window": a.window, "group": a.group,
                                    "verified": ok})
                chats[j] += [{"role": "assistant", "content": text}, {"role": "user", "content": FEEDBACK}]
        open_ = [j for j in open_ if not attempts[j][-1]["verified"]]
        print(f"round {rnd + 1}: {len(open_)} series still unverified ({time.perf_counter() - t0:.0f}s)",
              flush=True)
        if not open_:
            break

    y = np.array([r["label"] for r in recs])
    summary = {}
    for k in range(1, args.k + 1):
        acc_label = np.full(len(recs), -1)
        for j, att in enumerate(attempts):
            hit = next((a for a in att[:k] if a["verified"]), None)
            if hit is not None:
                acc_label[j] = hit["label"]
        answered = acc_label >= 0
        err = float((acc_label[answered] != y[answered]).mean()) if answered.any() else 0.0
        n_err = int((acc_label[answered] != y[answered]).sum())
        summary[k] = {"coverage": float(answered.mean()), "error": err, "n_answered": int(answered.sum()),
                      "n_errors": n_err}
        print(f"k={k}: coverage {answered.mean():.3f}  error among answered {err:.3f} ({n_err}/{answered.sum()})")
    first_acc = float(np.mean([att[0]["label"] == yy for att, yy in zip(attempts, y)]))
    dest = ROOT / "results" / "agent" / args.dataset / f"retry_{args.run}.json"
    dest.write_text(json.dumps({"run": args.run, "dataset": args.dataset, "k": args.k,
                                "first_attempt_accuracy": first_acc, "summary": summary,
                                "attempts": attempts}, indent=1))
    print(f"first-attempt accuracy {first_acc:.3f} -> {dest}")


if __name__ == "__main__":
    main()
