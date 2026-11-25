"""Generate the agent's answers on test series and store them for analysis.

  --run base   the untuned Qwen3 model, prompted only (zero-shot)
  --run sft    SFT warm start
  --run grpo_accuracy | grpo_verified   after GRPO

Writes results/agent/<dataset>/pred_<run>.json: one record per test series with
the raw answer and parsed label / window / group / property.

Usage:
    python scripts/eval_agent.py --dataset synth_planted --run grpo_verified
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.views import parse_answer  # noqa: E402
from veritas.data.base import load_processed  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_agent import BASE, load_records, templated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--adapters-from", default=None,
                    help="agent experiment whose trained runs to load (e.g. evaluate synth_planted runs on synth_planted_shift)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    if args.run != "base":
        from peft import PeftModel
        src = args.adapters_from or args.dataset
        model = PeftModel.from_pretrained(model, str(ROOT / "results" / "agent" / src / "runs" / args.run))
        model = model.merge_and_unload().eval()

    source, claims = agent_mode(args.dataset)
    meta = load_processed(source).meta
    grid = EvidenceGrid(T=8, channel_groups=meta.channel_groups, K=8)
    recs = load_records(args.dataset, "test")
    out, t0 = [], time.perf_counter()
    for s in range(0, len(recs), args.batch):
        chunk = recs[s:s + args.batch]
        enc = tok([templated(tok, r["prompt"]) for r in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, text in zip(chunk, texts):
            a = parse_answer(text, meta.class_names, grid.group_names, grid.K, claims=claims)
            out.append({"bank_i": r["bank_i"], "label_true": r["label"], "text": text,
                        "label": a.label, "window": a.window, "group": a.group, "prop": a.prop})
    acc = sum(o["label"] == o["label_true"] for o in out) / len(out)
    dest = ROOT / "results" / "agent" / args.dataset / f"pred_{args.run}.json"
    dest.write_text(json.dumps({"run": args.run, "dataset": args.dataset, "seconds": time.perf_counter() - t0,
                                "preds": out}, indent=1))
    print(f"{args.run}: accuracy {acc:.3f} on {len(out)} series ({time.perf_counter() - t0:.0f}s) -> {dest}")


if __name__ == "__main__":
    main()
