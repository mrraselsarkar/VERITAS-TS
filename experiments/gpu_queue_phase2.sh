#!/usr/bin/env bash
# Phase 2 GPU queue: run sequentially (the laptop GPU is shared and throttled).
# Each training retries with checkpoint-resume after Windows GPU watchdog resets.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

train() {  # dataset graph log extra-args...
  local ds=$1 g=$2 log=$3; shift 3
  for attempt in 1 2 3 4 5 6; do
    [ -f "results/diffusion/$ds/standard/$g.pt" ] && return 0
    echo "=== attempt $attempt" >> "$log"
    $PY -u scripts/train_diffusion.py --dataset "$ds" --graph "$g" "$@" >> "$log" 2>&1
  done
}

$PY scripts/prepare_data.py synth_context
train synth_context prior+learned results/logs/diffusion_synth_context.log \
  --channels 48 --layers 10 --batch-size 32 --iters 3000
$PY -u scripts/eval_verifier_synth.py --dataset synth_context --n 100 --m 2 \
  --methods zero noise interp splice diffusion:prior+learned hybrid:prior+learned \
  > results/logs/eval_verifier_synth_context.log 2>&1
$PY -u scripts/eval_verifier_synth.py --dataset synth_planted --n 100 --m 2 \
  --methods splice hybrid:prior+learned --seed 0 \
  > results/logs/eval_verifier_synth_planted_hybrid.log 2>&1
train pamap2 prior+learned results/logs/diffusion_pamap2.log \
  --channels 48 --layers 10 --batch-size 32 --iters 3000
train ptbxl prior+learned results/logs/diffusion_ptbxl.log \
  --channels 48 --layers 12 --dilation-cycle 8 --batch-size 16 --iters 3000
echo "queue done"
