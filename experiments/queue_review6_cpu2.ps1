# Review round 6, second CPU queue (the GPU is hidden; runs alongside queue_review6_gpu.ps1, which takes
# over any bank not yet locked here once the GPU is free):
#  (1) non-LLM agents on the subject-disjoint test sets, three seeds
#  (2) checker evaluation with per-recording scores (bootstrap intervals for Table III), three seeds
#  (3) independent judge on the subject-disjoint test sets
#  (4) stricter-check banks (cross-fitted oracle, other-class content in Test 2) and non-LLM agents on them
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$env:CUDA_VISIBLE_DEVICES = "-1"  # an empty value would delete the variable in PowerShell
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review6"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  "[$(Get-Date -Format s)] start (cpu)" | Out-File -Append "$L\$log.log"
  & $py -u @cmd *>> "$L\$log.log"
}
function RunLocked($done, [string[]]$cmd, $log) {
  if ((Test-Path $done) -or (Test-Path "$done.lock")) { return }
  "cpu" | Out-File "$done.lock"
  try { Run $null $cmd $log } finally { Remove-Item "$done.lock" -ErrorAction SilentlyContinue }
}
function WaitFor($path) { while (-not (Test-Path $path)) { Start-Sleep -Seconds 60 } }
$A = "results\agent"; $B = "results\bank"; $V = "results\verifier"

# (1)
foreach ($N in "ucihar_subj_group", "pamap2_subj_group") {
  foreach ($seed in 0, 1, 2) {
    $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
    foreach ($c in "verified", "heuristic") {
      Run "$A\${N}_shift\pred_policy_$c$sfx.json" @("scripts\train_policy.py", "--dataset", $N, "--claim", $c, "--seed", "$seed", "--tag", $sfx, "--also-test", "${N}_shift") "policy_shift_$N$c$sfx"
    }
  }
}
# (2)
foreach ($s in 0, 1, 2) {
  Run "$V\synth_planted_eval_seed${s}_ci.json" @("scripts\eval_verifier_synth.py", "--dataset", "synth_planted", "--n", "100", "--m", "2", "--seed", "$s", "--out-tag", "_ci", "--methods", "zero", "noise", "interp", "splice", "diffusion:none", "diffusion:prior+learned", "hybrid:none", "hybrid:prior+learned") "verifier_ci_s$s"
}
# (3)
foreach ($ds in "ucihar_subj", "pamap2_subj") {
  RunLocked "$B\${ds}_group_shift\test_judge2.npz" @("scripts\build_bank.py", "--dataset", $ds, "--split", "shift", "--part", "test", "--n", "400", "--K", "1", "--max-groups", "2", "--tag", "_group_shift", "--oracles", "cnn", "--out-name", "test_judge2.npz") "judge2_shift_$ds"
}
# (4) name, data, K, max groups, tag, train n, test n
$sets = @(@("natops_group", "natops", "1", "2", "_group_xf", "180", "180"),
          @("cricket_group", "cricket", "1", "1", "_group_xf", "108", "72"),
          @("ucihar_subj_group", "ucihar_subj", "1", "2", "_group_xf", "1000", "400"),
          @("synth_planted", "synth_planted", "8", "1", "_xf", "1680", "720"),
          @("pamap2_group", "pamap2", "1", "2", "_group_xf", "1000", "400"),
          @("ptbxl_group", "ptbxl", "1", "2", "_group_xf", "1000", "400"))
foreach ($x in $sets) {
  $N, $ds, $K, $mg, $tag, $ntr, $nte = $x
  WaitFor "$L\xf_oracles_$ds.done"
  $common = @("--dataset", $ds, "--K", $K, "--max-groups", $mg, "--tag", $tag, "--crossfit", "--suf-donors", "other", "--chunk", "20")
  # test banks here; the remaining training banks are built by queue_review6_cpu3.ps1 (PAMAP2, PTB-XL)
  # and on the GPU (synthetic)
  RunLocked "$B\$N`_xf\test.npz" (@("scripts\build_bank.py", "--part", "test", "--n", $nte, "--extra-judges", "full", "cnn") + $common) "xf_bank_${N}_test"
}
foreach ($x in $sets) {
  $N = $x[0]
  WaitFor "$B\$N`_xf\train.npz"
  foreach ($seed in 0, 1, 2) {
    $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
    Run "$A\$N\pred_policy_verified_xf$sfx.json" @("scripts\train_policy.py", "--dataset", $N, "--claim", "verified", "--seed", "$seed", "--tag", "_xf$sfx", "--bank-tag", "_xf") "policy_xf_$N$sfx"
  }
}
"done" | Out-File "$L\cpu2.done"
