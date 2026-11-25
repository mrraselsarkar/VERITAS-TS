# Review round 5, GPU part, most important first:
#  (1) masking-check reward on real data (UCI-HAR, PTB-XL): warm start and GRPO both use zero-masking verdicts
#  (2) oracle and judge swapped: the reward reads the CNN judge's verdicts (synthetic, UCI-HAR)
#  (3) LLM agent without neighbour classes (synthetic)
#  (4) seeds 1 and 2 of the masking-check reward
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review5"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
function WaitFor($path) { while (-not (Test-Path $path)) { Start-Sleep -Seconds 60 } }

# CNN-judged training banks (same edits as the main banks: same rows and seeds)
Run "results\bank\synth_planted_cnn\train.npz" @("scripts\build_bank.py", "--dataset", "synth_planted", "--part", "train", "--n", "1680", "--oracles", "cnn", "--tag", "_cnn") "cnn_train_synth"
Run "results\bank\ucihar_subj_group_cnn\train.npz" @("scripts\build_bank.py", "--dataset", "ucihar_subj", "--part", "train", "--n", "1000", "--K", "1", "--max-groups", "2", "--tag", "_group_cnn", "--oracles", "cnn") "cnn_train_ucihar"

# (1) masking-check reward, seed 0
foreach ($ds in "ucihar_subj_group", "ptbxl_group") {
  $src = $ds -replace "_group$", ""
  WaitFor "results\bank\${src}_group_zero\train.npz"
  $A = "results\agent\$ds"
  Run "$A\runs\sft_verified_zero\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $ds, "--targets", "verified", "--bank-tag", "_zero", "--tag", "_zero") "sft_zero_$ds"
  Run "$A\runs\grpo_verified_ver_zero\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $ds, "--reward", "verified", "--steps", "100", "--init", "sft_verified_zero", "--tag", "_ver_zero", "--bank-tag", "_zero") "grpo_zero_$ds"
  Run "$A\pred_grpo_verified_ver_zero.json" @("scripts\eval_agent.py", "--dataset", $ds, "--run", "grpo_verified_ver_zero") "eval_zero_$ds"
}

# (2) swapped judges, seed 0
$A = "results\agent\synth_planted"
Run "$A\runs\grpo_verified_div_cnn\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "synth_planted", "--reward", "verified", "--init", "sft_diverse", "--tag", "_div_cnn", "--bank-tag", "_cnn") "grpo_cnn_synth"
Run "$A\pred_grpo_verified_div_cnn.json" @("scripts\eval_agent.py", "--dataset", "synth_planted", "--run", "grpo_verified_div_cnn") "eval_cnn_synth"
$A = "results\agent\ucihar_subj_group"
Run "$A\runs\sft_verified_cnn\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", "ucihar_subj_group", "--targets", "verified", "--bank-tag", "_cnn", "--tag", "_cnn") "sft_cnn_ucihar"
Run "$A\runs\grpo_verified_ver_cnn\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "ucihar_subj_group", "--reward", "verified", "--steps", "100", "--init", "sft_verified_cnn", "--tag", "_ver_cnn", "--bank-tag", "_cnn") "grpo_cnn_ucihar"
Run "$A\pred_grpo_verified_ver_cnn.json" @("scripts\eval_agent.py", "--dataset", "ucihar_subj_group", "--run", "grpo_verified_ver_cnn") "eval_cnn_ucihar"

# (3) LLM without neighbour classes (synthetic), warm start and both rewards
Run "results\agent\synth_planted_nonb\test.jsonl" @("scripts\make_nonb_views.py", "--dataset", "synth_planted") "nonb_views"
$A = "results\agent\synth_planted_nonb"
Run "$A\runs\sft_diverse\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", "synth_planted_nonb", "--targets", "diverse") "sft_nonb"
foreach ($r in "accuracy", "verified") {
  Run "$A\runs\grpo_${r}_div\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "synth_planted_nonb", "--reward", $r, "--init", "sft_diverse", "--tag", "_div") "grpo_nonb_$r"
  Run "$A\pred_grpo_${r}_div.json" @("scripts\eval_agent.py", "--dataset", "synth_planted_nonb", "--run", "grpo_${r}_div") "eval_nonb_$r"
}
"main done" | Out-File "$L\gpu_main.done"

# (4) masking-check reward, seeds 1 and 2
foreach ($s in 1, 2) {
  foreach ($ds in "ucihar_subj_group", "ptbxl_group") {
    $A = "results\agent\$ds"
    Run "$A\runs\grpo_verified_ver_zero_s$s\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $ds, "--reward", "verified", "--steps", "100", "--init", "sft_verified_zero", "--tag", "_ver_zero_s$s", "--bank-tag", "_zero", "--seed", "$s") "grpo_zero_${ds}_s$s"
    Run "$A\pred_grpo_verified_ver_zero_s$s.json" @("scripts\eval_agent.py", "--dataset", $ds, "--run", "grpo_verified_ver_zero_s$s") "eval_zero_${ds}_s$s"
  }
}
"done" | Out-File "$L\gpu.done"
