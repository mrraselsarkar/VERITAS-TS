"""The LLM agent's own confidence in its class, for the abstention baselines.

For every test recording the answer line "label: <class>" is scored for each class
name after the prompt (teacher forcing); a softmax over the classes gives the agent's
class probabilities. Written to results/agent/<dataset>/conf_<run>.json as
{"bank_i": [...], "probs": [[...], ...]}.

Usage:
    python scripts/agent_confidence.py --dataset synth_planted_shift --run grpo_verified_div \
        --adapters-from synth_planted
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

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.data.base import load_processed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_agent import BASE, load_records, templated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapters-from", default=None)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    src = args.adapters_from or args.dataset
    model = PeftModel.from_pretrained(model, str(ROOT / "results" / "agent" / src / "runs" / args.run))
    model = model.merge_and_unload().eval()

    source, _ = agent_mode(args.dataset)
    names = load_processed(source).meta.class_names
    conts = [tok(f"label: {c}\n", add_special_tokens=False)["input_ids"] for c in names]
    recs = load_records(args.dataset, "test")
    out_i, out_p, t0 = [], [], time.perf_counter()
    for r in recs:
        prompt = tok(templated(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        seqs = [prompt + c for c in conts]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):  # right padding keeps the positions of the real tokens
            ids[k, :len(s)], att[k, :len(s)] = torch.as_tensor(s), 1
        with torch.no_grad():  # logits at the answer positions only: the full logits of 12 classes overflow the GPU
            hid = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            score = []
            for k, c in enumerate(conts):
                pos = torch.arange(len(prompt) - 1, len(prompt) - 1 + len(c))
                logp = torch.log_softmax(model.lm_head(hid[k, pos]).float(), -1)
                score.append(logp[torch.arange(len(c)), torch.as_tensor(c)].sum().item())
        p = np.exp(np.array(score) - max(score))
        out_i.append(r["bank_i"])
        out_p.append((p / p.sum()).round(6).tolist())
    dest = ROOT / "results" / "agent" / args.dataset / f"conf_{args.run}.json"
    dest.write_text(json.dumps({"run": args.run, "classes": names, "bank_i": out_i, "probs": out_p}))
    print(f"{args.run} on {args.dataset}: {len(out_i)} recordings ({time.perf_counter() - t0:.0f}s) -> {dest}")


if __name__ == "__main__":
    main()
