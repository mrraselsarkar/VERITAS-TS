"""Train the LLM agent: SFT warm start, then GRPO (proposal, Method parts 2 and 4).

  sft   LoRA fine-tuning on (view -> true label + heuristic, unverified cell).
        It teaches the answer format and the label, not which evidence is real.
  grpo  continues the SFT adapter with GRPO. --reward accuracy pays for the label
        only (TimeMaster-style baseline); --reward verified also pays +1 when the
        cited cell passes the bank's necessity and sufficiency tests.

Usage:
    python scripts/train_agent.py sft  --dataset synth_planted
    python scripts/train_agent.py grpo --dataset synth_planted --reward verified
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veritas.agent.config import agent_mode  # noqa: E402
from veritas.agent.reward import Bank, reward  # noqa: E402
from veritas.agent.views import (format_answer, format_group_answer, parse_answer,  # noqa: E402
                                 ranked_cells, ranked_groups)
from veritas.data.base import load_processed  # noqa: E402
from veritas.grid import EvidenceGrid  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"


def templated(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


def load_records(dataset: str, part: str) -> list[dict]:
    path = ROOT / "results" / "agent" / dataset / f"{part}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_sft(args, tok):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    recs = load_records(args.dataset, "train")
    rows = [{"prompt": templated(tok, r["prompt"]), "completion": r["sft_target"]} for r in recs]
    if args.targets != "heuristic":
        source, claims = agent_mode(args.dataset)
        meta = load_processed(source).meta
        names, classes = sorted(meta.channel_groups), meta.class_names
        feat = np.load(ROOT / "results" / "agent" / args.dataset / "features.npz")["train_feat"]
        # --bank-tag: warm-start targets filtered by another bank (masking edits or the second judge)
        bank = Bank(ROOT / "results" / "bank" / f"{args.dataset}{args.bank_tag}" / "train.npz")
        rows = []
        for r in recs:
            i, lab = r["bank_i"], classes[r["label"]]
            ok = bank.verified_cells(i, r["label"])
            if claims == "group":  # candidates: ([group, ...], property)
                ranked = ranked_groups(feat[i], names)
                key = lambda c: bank.claim_id(0, "+".join(sorted(c[0])))  # noqa: E731
                fmt = lambda c: format_group_answer(lab, c[0], c[1])  # noqa: E731
            else:  # candidates: (window, group index, property)
                ranked = ranked_cells(feat[i])
                key = lambda c: bank.claim_id(c[0], names[c[1]])  # noqa: E731
                fmt = lambda c: format_answer(lab, c[0], names[c[1]], c[2])  # noqa: E731
            if args.targets == "diverse":  # the three top-ranked candidates, one example each
                cands = ranked[:3]
            else:  # "verified": rejection sampling, keep the first ranked candidate that passes the verifier
                best = next((c for c in ranked if key(c) is not None and ok[key(c)]), None)
                if best is None and claims == "group" and ok.any():  # any verified claim, e.g. a pair the heuristic never ranks
                    g = next(k[1] for k, j in bank.index.items() if j == int(np.flatnonzero(ok)[0]))
                    best = (tuple(g.split("+")), ranked[0][1])
                cands = [best or ranked[0]]
            p = templated(tok, r["prompt"])
            rows += [{"prompt": p, "completion": fmt(c)} for c in cands]
    ds = Dataset.from_list(rows).shuffle(seed=args.seed)
    cfg = SFTConfig(output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
                    per_device_train_batch_size=8, gradient_accumulation_steps=2, bf16=True,
                    logging_steps=20, save_strategy="no", report_to="none", max_length=1536,
                    completion_only_loss=True, gradient_checkpointing=True, seed=args.seed)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=BASE, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(args.out))


def run_grpo(args, tok):
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    source, claims = agent_mode(args.dataset)
    meta = load_processed(source).meta
    grid = EvidenceGrid(T=8, channel_groups=meta.channel_groups, K=8)  # only names and K are used
    part = "test" if args.smoke else "train"  # --smoke: a few steps on test data, only to check the setup
    # --bank-tag selects the bank the reward reads, e.g. "_zero" for an ablation with masking edits
    bank = Bank(ROOT / "results" / "bank" / f"{args.dataset}{args.bank_tag}" / f"{part}.npz")
    recs = load_records(args.dataset, part)
    ds = Dataset.from_list([{"prompt": templated(tok, r["prompt"]), "label": r["label"], "bank_i": r["bank_i"]}
                            for r in recs]).shuffle(seed=args.seed)
    stats = {"n": 0, "acc": 0.0, "ver": 0.0}

    def reward_fn(prompts, completions, label, bank_i, **_):
        out = []
        for text, y, i in zip(completions, label, bank_i):
            ans = parse_answer(text, meta.class_names, grid.group_names, grid.K, claims=claims)
            out.append(reward(ans, y, bank, i, args.reward))
            stats["n"] += 1
            stats["acc"] += ans.label == y
            stats["ver"] += ans.label == y and bank.verified(i, bank.claim_id(ans.window, ans.group), y)
        if stats["n"] >= 512:
            print(f"[rollouts] acc {stats['acc'] / stats['n']:.3f}  verified {stats['ver'] / stats['n']:.3f}",
                  flush=True)
            stats.update(n=0, acc=0.0, ver=0.0)
        return out

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(args.init), is_trainable=True)
    cfg = GRPOConfig(output_dir=str(args.out), max_steps=args.steps, learning_rate=args.lr,
                     per_device_train_batch_size=16, num_generations=8, gradient_accumulation_steps=2,
                     max_completion_length=48, temperature=args.temperature, beta=0.0, bf16=True,
                     logging_steps=10, save_strategy="no", report_to="none", seed=args.seed,
                     gradient_checkpointing=True)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds,
                          processing_class=tok)
    trainer.train()
    trainer.save_model(str(args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "grpo"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reward", choices=["accuracy", "verified"], default="verified")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200, help="GRPO optimiser steps")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0, help="GRPO sampling temperature")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument("--targets", choices=["heuristic", "diverse", "verified"], default="heuristic",
                    help="SFT evidence targets: top anomalous cell, top-3 cells, or verifier-filtered cell")
    ap.add_argument("--init", default="sft", help="GRPO: SFT run to start from")
    ap.add_argument("--bank-tag", default="", help="reward (GRPO) or target-filter (SFT) bank directory suffix")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    runs = ROOT / "results" / "agent" / args.dataset / "runs"
    args.init = runs / args.init
    if args.stage == "sft":
        name = ("sft" if args.targets == "heuristic" else f"sft_{args.targets}") + args.tag
        args.out, args.lr = runs / name, args.lr or 2e-4
        run_sft(args, tok)
    else:
        args.out, args.lr = runs / f"grpo_{args.reward}{args.tag}{'_smoke' if args.smoke else ''}", args.lr or 2e-5
        run_grpo(args, tok)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
