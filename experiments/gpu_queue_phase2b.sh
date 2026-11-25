#!/usr/bin/env bash
# Phase 2b: starts after gpu_queue_phase2.sh finishes. Real-data verifier
# diagnostics, plus a re-run of all synth_planted methods with the paired realism metric.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
until grep -q "queue done" results/logs/gpu_queue_phase2.log 2>/dev/null; do sleep 60; done

$PY -u scripts/eval_verifier_synth.py --dataset synth_planted --n 100 --m 2 \
  --methods zero noise interp splice diffusion:prior+learned diffusion:none hybrid:prior+learned \
  > results/logs/eval_verifier_synth_planted_all.log 2>&1
for ds in pamap2 ptbxl; do
  [ -f "results/diffusion/$ds/standard/prior+learned.pt" ] || continue
  $PY -u scripts/eval_verifier_real.py --dataset "$ds" --n 30 \
    --methods zero interp splice diffusion:prior+learned hybrid:prior+learned \
    > "results/logs/eval_verifier_real_$ds.log" 2>&1
done
echo "queue 2b done"
