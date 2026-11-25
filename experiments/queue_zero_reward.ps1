# Ablation (review M3): GRPO with a check reward built from zero-masking edits instead of the
# hybrid realistic edits. Same warm start, steps and seed as grpo_verified_div. The trained agent is
# then evaluated with the standard (hybrid) bank and the planted-evidence ground truth.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
$A = "results\agent\synth_planted"
if (-not (Test-Path "$A\runs\grpo_verified_div_zero\adapter_model.safetensors")) {
  & $py -u scripts\train_agent.py grpo --dataset synth_planted --reward verified --init sft_diverse --tag _div_zero --bank-tag _zero *>> "$L\grpo_zero_reward.log"
}
if (-not (Test-Path "$A\pred_grpo_verified_div_zero.json")) {
  & $py -u scripts\eval_agent.py --dataset synth_planted --run grpo_verified_div_zero *>> "$L\eval_zero_reward.log"
}
"done" | Out-File "$L\zero_reward.done"
