# Review round 6, GPU part. Every step waits until enough GPU memory is free, because other jobs
# may be using the GPU; finished steps are skipped on a restart. Runs alongside queue_review6_cpu2.ps1:
# a bank locked by the CPU queue is skipped here.
#  (2) subject-disjoint test sets: UCI-HAR seeds 1-2, PAMAP2 (trained on part of the people) seeds 1-2;
#      agents on all PTB-XL test recordings of myocardial infarction (bank and views: queue_review6_mi.ps1);
#      the agent's own confidence
#  (3) LLM with the stricter check, seed 0 (building any bank the CPU queue has not started)
#  (4) LLM with the stricter check, seeds 1-2
#  (5) seeds 1-2 of the oracle/judge swap; LLM without neighbour classes (synthetic seeds 1-2, UCI-HAR)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review6"
New-Item -ItemType Directory -Force $L | Out-Null
function WaitGPU($mb) {
  while ($true) {
    $free = [int]((nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1)
    if ($free -ge $mb) { return }
    Start-Sleep -Seconds 120
  }
}
function Run($done, $mb, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  if ($mb -gt 0) { WaitGPU $mb }
  "[$(Get-Date -Format s)] start (gpu)" | Out-File -Append "$L\$log.log"
  & $py -u @cmd *>> "$L\$log.log"
}
function RunLocked($done, $mb, [string[]]$cmd, $log) {
  if ((Test-Path $done) -or (Test-Path "$done.lock")) { return }
  WaitGPU $mb
  if ((Test-Path $done) -or (Test-Path "$done.lock")) { return }
  "gpu" | Out-File "$done.lock"
  try { Run $null 0 $cmd $log } finally { Remove-Item "$done.lock" -ErrorAction SilentlyContinue }
}
function WaitFor($path) { while (-not (Test-Path $path)) { Start-Sleep -Seconds 60 } }
# after a restart, let an agent job of the previous queue finish first
function WaitIdle {
  while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "scripts\\(train_agent|eval_agent|agent_confidence)\.py" }) { Start-Sleep -Seconds 60 }
}
WaitIdle
$A = "results\agent"; $B = "results\bank"

# stricter-check banks: name -> data, K, max groups, tag, train n, test n
$sets = @{"synth_planted" = @("synth_planted", "8", "1", "_xf", "1680", "720");
          "ucihar_subj_group" = @("ucihar_subj", "1", "2", "_group_xf", "1000", "400");
          "natops_group" = @("natops", "1", "2", "_group_xf", "180", "180");
          "cricket_group" = @("cricket", "1", "1", "_group_xf", "108", "72");
          "pamap2_group" = @("pamap2", "1", "2", "_group_xf", "1000", "400");
          "ptbxl_group" = @("ptbxl", "1", "2", "_group_xf", "1000", "400")}
function XfBank($N, $part) {
  $ds, $K, $mg, $tag, $ntr, $nte = $sets[$N]
  WaitFor "$L\xf_oracles_$ds.done"
  $common = @("--dataset", $ds, "--K", $K, "--max-groups", $mg, "--tag", $tag, "--crossfit", "--suf-donors", "other", "--chunk", "20")
  if ($part -eq "train") { $extra = @("--n", $ntr, "--extra-judges", "full") } else { $extra = @("--n", $nte, "--extra-judges", "full", "cnn") }
  RunLocked "$B\$N`_xf\$part.npz" 3000 (@("scripts\build_bank.py", "--part", $part) + $extra + $common) "xf_bank_${N}_$part"
  WaitFor "$B\$N`_xf\$part.npz"
}

# (2)
foreach ($s in "_s1", "_s2") {
  foreach ($r in "accuracy", "verified") {
    Run "$A\ucihar_subj_group_shift\pred_grpo_${r}_ver$s.json" 8000 @("scripts\eval_agent.py", "--dataset", "ucihar_subj_group_shift", "--run", "grpo_${r}_ver$s", "--adapters-from", "ucihar_subj_group") "eval_shift_ucihar_${r}$s"
  }
}
WaitFor "results\logs\review6\mi_views.done"
foreach ($r in "accuracy", "verified") {
  foreach ($s in "", "_s1", "_s2") {
    Run "$A\ptbxl_group_mi\pred_grpo_${r}_ver$s.json" 8000 @("scripts\eval_agent.py", "--dataset", "ptbxl_group_mi", "--run", "grpo_${r}_ver$s", "--adapters-from", "ptbxl_group") "eval_mi_${r}$s"
  }
}
foreach ($x in @(@("synth_planted", "grpo_verified_div"), @("ucihar_subj_group", "grpo_verified_ver"), @("pamap2_subj_group", "grpo_verified_ver"))) {
  $N, $run = $x
  foreach ($d in $N, "${N}_shift") {
    Run "$A\$d\conf_$run.json" 8000 @("scripts\agent_confidence.py", "--dataset", $d, "--run", $run, "--adapters-from", $N) "conf_$d"
  }
}
$N = "pamap2_subj_group"
foreach ($seed in 1, 2) {
  foreach ($r in "accuracy", "verified") {
    Run "$A\$N\runs\grpo_${r}_ver_s$seed\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver_s$seed", "--seed", "$seed") "grpo_${N}_${r}_s$seed"
    Run "$A\$N\pred_grpo_${r}_ver_s$seed.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_${r}_ver_s$seed") "eval_${N}_${r}_s$seed"
    Run "$A\${N}_shift\pred_grpo_${r}_ver_s$seed.json" 8000 @("scripts\eval_agent.py", "--dataset", "${N}_shift", "--run", "grpo_${r}_ver_s$seed", "--adapters-from", $N) "eval_${N}_shift_${r}_s$seed"
  }
}
"subj" | Out-File "$L\subj.done"

# (3)
function XfRun($N, $seed) {
  $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
  if ($N -eq "synth_planted") { $init = "sft_diverse"; $r = "div"; $steps = "200" } else { $init = "sft_verified"; $r = "ver"; $steps = "100" }
  XfBank $N "train"
  Run "$A\$N\runs\grpo_verified_${r}_xf$sfx\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", "verified", "--steps", $steps, "--init", $init, "--tag", "_${r}_xf$sfx", "--bank-tag", "_xf", "--seed", "$seed") "grpo_xf_$N$sfx"
  XfBank $N "test"
  Run "$A\$N\pred_grpo_verified_${r}_xf$sfx.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_verified_${r}_xf$sfx") "eval_xf_$N$sfx"
}
# PAMAP2 last: its training bank is built on the CPU and is the last one ready
foreach ($N in "synth_planted", "ucihar_subj_group", "natops_group", "cricket_group", "ptbxl_group", "pamap2_group") { XfRun $N 0 }
"seed0" | Out-File "$L\xf_seed0.done"

# (5) before (4): these results had a single seed
foreach ($seed in 1, 2) {
  $N = "synth_planted"
  Run "$A\$N\runs\grpo_verified_div_cnn_s$seed\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", "verified", "--init", "sft_diverse", "--tag", "_div_cnn_s$seed", "--bank-tag", "_cnn", "--seed", "$seed") "grpo_cnn_synth_s$seed"
  Run "$A\$N\pred_grpo_verified_div_cnn_s$seed.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_verified_div_cnn_s$seed") "eval_cnn_synth_s$seed"
  $N = "ucihar_subj_group"
  Run "$A\$N\runs\grpo_verified_ver_cnn_s$seed\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", "verified", "--steps", "100", "--init", "sft_verified_cnn", "--tag", "_ver_cnn_s$seed", "--bank-tag", "_cnn", "--seed", "$seed") "grpo_cnn_ucihar_s$seed"
  Run "$A\$N\pred_grpo_verified_ver_cnn_s$seed.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_verified_ver_cnn_s$seed") "eval_cnn_ucihar_s$seed"
}
$N = "synth_planted_nonb"
foreach ($seed in 1, 2) {
  foreach ($r in "accuracy", "verified") {
    Run "$A\$N\runs\grpo_${r}_div_s$seed\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--init", "sft_diverse", "--tag", "_div_s$seed", "--seed", "$seed") "grpo_nonb_${r}_s$seed"
    Run "$A\$N\pred_grpo_${r}_div_s$seed.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_${r}_div_s$seed") "eval_nonb_${r}_s$seed"
  }
}
$N = "ucihar_subj_group_nonb"
Run "$A\$N\test.jsonl" 0 @("scripts\make_nonb_views.py", "--dataset", "ucihar_subj_group") "nonb_views_ucihar"
Run "$A\$N\runs\sft_verified\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "sft", "--dataset", $N, "--targets", "verified") "sft_nonb_ucihar"
foreach ($seed in 0, 1, 2) {
  $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
  foreach ($r in "accuracy", "verified") {
    Run "$A\$N\runs\grpo_${r}_ver$sfx\adapter_model.safetensors" 11000 @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver$sfx", "--seed", "$seed") "grpo_nonb_ucihar_${r}$sfx"
    Run "$A\$N\pred_grpo_${r}_ver$sfx.json" 8000 @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_${r}_ver$sfx") "eval_nonb_ucihar_${r}$sfx"
  }
}

# (4)
foreach ($seed in 1, 2) { foreach ($N in "synth_planted", "ucihar_subj_group", "natops_group", "cricket_group", "pamap2_group", "ptbxl_group") { XfRun $N $seed } }
"xf" | Out-File "$L\xf.done"
"done" | Out-File "$L\gpu.done"
