# Review round, GPU queue 2: agents on the new real datasets (sensor claims), the real
# subject shift on UCI-HAR, then extra seeds for the main comparisons. Runs after queue 1.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
New-Item -ItemType Directory -Force $L | Out-Null
while (-not (Test-Path "$L\queue1.done")) { Start-Sleep 60 }
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
function Agent($name, $run) { Run "results\agent\$name\pred_$run.json" @("scripts\eval_agent.py", "--dataset", $name, "--run", $run) "eval_${name}_$run" }

# ---- 1. new real datasets with sensor claims ----
foreach ($ds in @("ptbxl", "ucihar_subj")) {
  $N = "${ds}_group"; $A = "results\agent\$N"
  Run "$A\train.jsonl" @("scripts\make_views.py", "--dataset", $ds, "--claims", "group") "views_$N"
  Run "$A\runs\sft_verified\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $N, "--targets", "verified", "--epochs", "2") "sft_verified_$N"
  Agent $N "base"
  Agent $N "sft_verified"
  foreach ($r in @("verified", "accuracy")) {
    Run "$A\runs\grpo_${r}_ver\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver") "grpo_${r}_ver_$N"
    Agent $N "grpo_${r}_ver"
  }
  foreach ($c in @("verified", "heuristic")) {
    Run "$A\pred_policy_$c.json" @("scripts\train_policy.py", "--dataset", $N, "--claim", $c) "policy_${c}_$N"
  }
  Run $null @("scripts\analyze_agent.py", "--dataset", $N, "--alpha", "0.1") "analyze_$N"
}

# ---- 2. real subject shift on UCI-HAR ----
$S = "ucihar_subj_group_shift"
Run "results\agent\$S\test.jsonl" @("scripts\make_views.py", "--dataset", "ucihar_subj", "--claims", "group", "--split", "shift", "--parts", "test") "views_$S"
foreach ($r in @("sft_verified", "grpo_accuracy_ver", "grpo_verified_ver")) {
  Run "results\agent\$S\pred_$r.json" @("scripts\eval_agent.py", "--dataset", $S, "--run", $r, "--adapters-from", "ucihar_subj_group") "eval_${S}_$r"
}
Run $null @("scripts\shift_eval.py", "--dataset", "ucihar_subj_group", "--runs", "sft_verified", "grpo_accuracy_ver", "grpo_verified_ver", "--alpha", "0.1") "shift_eval_ucihar"
foreach ($pair in @(@("ucihar_subj_group", ""), @($S, "ucihar_subj_group"))) {
  $ds, $from = $pair; $extra = @(); if ($from) { $extra = @("--adapters-from", $from) }
  Run "results\agent\$ds\retry_grpo_verified_ver.json" (@("scripts\eval_retry.py", "--dataset", $ds, "--run", "grpo_verified_ver", "--k", "4") + $extra) "retry_$ds"
}

# ---- 3. extra seeds for the main comparisons ----
foreach ($s in @(1, 2)) {
  foreach ($r in @("verified", "accuracy")) {
    Run "results\agent\synth_planted\runs\grpo_${r}_div_s$s\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "synth_planted", "--reward", $r, "--steps", "200", "--init", "sft_diverse", "--tag", "_div_s$s", "--seed", "$s") "seed_synth_${r}_s$s"
    Agent "synth_planted" "grpo_${r}_div_s$s"
    Run "results\agent\pamap2_group\runs\grpo_${r}_ver_s$s\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", "pamap2_group", "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver_s$s", "--seed", "$s") "seed_pamap2_${r}_s$s"
    Agent "pamap2_group" "grpo_${r}_ver_s$s"
  }
  foreach ($ds in @("synth_planted", "pamap2_group")) {
    foreach ($c in @("verified", "heuristic")) {
      Run "results\agent\$ds\pred_policy_${c}_s$s.json" @("scripts\train_policy.py", "--dataset", $ds, "--claim", $c, "--seed", "$s", "--tag", "_s$s") "seed_policy_${ds}_${c}_s$s"
    }
  }
}
"done" | Out-File "$L\queue2.done"
