# Ablation: GRPO with a reward for agreeing with occlusion (top occlusion cell of the oracle)
# instead of passing the check. Same warm start, steps and seeds as the check-reward runs.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
$A = "results\agent\synth_planted"
foreach ($s in 0, 1, 2) {
  $sfx = if ($s -eq 0) { "" } else { "_s$s" }
  if (-not (Test-Path "$A\runs\grpo_verified_div_occl$sfx\adapter_model.safetensors")) {
    & $py -u scripts\train_agent.py grpo --dataset synth_planted --reward verified --init sft_diverse --tag "_div_occl$sfx" --bank-tag _occl --seed $s *>> "$L\grpo_occl_reward$sfx.log"
  }
  if (-not (Test-Path "$A\pred_grpo_verified_div_occl$sfx.json")) {
    & $py -u scripts\eval_agent.py --dataset synth_planted --run "grpo_verified_div_occl$sfx" *>> "$L\eval_occl_reward$sfx.log"
  }
}
"done" | Out-File "$L\occl_reward.done"
