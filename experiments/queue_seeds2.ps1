# Review round 3 (M3): seeds 1 and 2 for PTB-XL and UCI-HAR (both rewards and the non-LLM agent)
# and for the masking-reward ablation on the synthetic data. Same settings as the seed-0 runs.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
foreach ($s in 1, 2) {
  # non-LLM agent (fast)
  foreach ($ds in "ptbxl_group", "ucihar_subj_group") {
    Run "results\agent\$ds\pred_policy_verified_s$s.json" @("scripts\train_policy.py", "--dataset", $ds, "--claim", "verified", "--seed", "$s", "--tag", "_s$s") "seed_policy_${ds}_verified_s$s"
  }
  # LLM agent, both rewards
  foreach ($ds in "ptbxl_group", "ucihar_subj_group") {
    foreach ($r in "accuracy", "verified") {
      $A = "results\agent\$ds"
      Run "$A\runs\grpo_${r}_ver_s$s\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $ds, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver_s$s", "--seed", "$s") "seed_${ds}_${r}_s$s"
      Run "$A\pred_grpo_${r}_ver_s$s.json" @("scripts\eval_agent.py", "--dataset", $ds, "--run", "grpo_${r}_ver_s$s") "eval_seed_${ds}_${r}_s$s"
    }
  }
  # masking-reward ablation
  $A = "results\agent\synth_planted"
  Run "$A\runs\grpo_verified_div_zero_s$s\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "synth_planted", "--reward", "verified", "--init", "sft_diverse", "--tag", "_div_zero_s$s", "--bank-tag", "_zero", "--seed", "$s") "seed_zero_reward_s$s"
  Run "$A\pred_grpo_verified_div_zero_s$s.json" @("scripts\eval_agent.py", "--dataset", "synth_planted", "--run", "grpo_verified_div_zero_s$s") "eval_seed_zero_reward_s$s"
}
"done" | Out-File "$L\seeds2.done"
