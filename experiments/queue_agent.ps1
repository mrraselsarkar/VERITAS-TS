# Phase 3-5 queue for one dataset: warm starts, GRPO runs, non-LLM policies, analysis.
# Waits for the heuristic SFT adapter and the training bank. Steps whose output exists are skipped.
#   sft           warm start on the single most anomalous cell (peaked: little exploration)
#   sft_diverse   warm start on the three most anomalous cells (the GRPO starting point)
#   sft_verified  warm start on a verifier-filtered cell (proposal's rejection-sampling warm start)
#   grpo_{verified,accuracy}_div   GRPO from sft_diverse with each reward
param([string]$Dataset = "synth_planted", [int]$Steps = 200, [double]$Alpha = 0.05)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$A = "results\agent\$Dataset"
$log = "results\logs\agent_$Dataset"

while (-not ((Test-Path "$A\runs\sft\adapter_model.safetensors") -and (Test-Path "results\bank\$Dataset\train.npz"))) {
  Start-Sleep 60
}
function Step($out, [string[]]$cmd, $name) {
  if (Test-Path $out) { return }
  & $py -u @cmd *>> "${log}_$name.log"
}
function Eval($run) { Step "$A\pred_$run.json" @("scripts\eval_agent.py", "--dataset", $Dataset, "--run", $run) "eval_$run" }

Eval "base"
Eval "sft"
Step "$A\runs\sft_diverse\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $Dataset, "--targets", "diverse", "--epochs", "1") "sft_diverse"
Eval "sft_diverse"
Step "$A\runs\sft_verified\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $Dataset, "--targets", "verified", "--epochs", "2") "sft_verified"
Eval "sft_verified"
foreach ($r in @("verified", "accuracy")) {
  Step "$A\runs\grpo_${r}_div\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $Dataset, "--reward", $r, "--steps", "$Steps", "--init", "sft_diverse", "--tag", "_div") "grpo_${r}_div"
  Eval "grpo_${r}_div"
}
foreach ($c in @("verified", "heuristic")) {
  Step "$A\pred_policy_$c.json" @("scripts\train_policy.py", "--dataset", $Dataset, "--claim", $c) "policy_$c"
}
& $py -u scripts\analyze_agent.py --dataset $Dataset --alpha $Alpha *> "${log}_analyze.log"
"done" | Out-File "${log}_queue.done"
