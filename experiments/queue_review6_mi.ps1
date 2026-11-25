# Review round 6: all PTB-XL test recordings of myocardial infarction (CPU). The bank checks every claim of
# every MI recording; the views let the agents answer them (LLM evaluation: queue_review6_gpu.ps1).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$env:CUDA_VISIBLE_DEVICES = ""
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review6"
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  "[$(Get-Date -Format s)] start (cpu)" | Out-File -Append "$L\$log.log"
  & $py -u @cmd *>> "$L\$log.log"
}
$N = "ptbxl_group_mi"
Run "results\bank\$N\test.npz" @("scripts\build_bank.py", "--dataset", "ptbxl", "--part", "test", "--n", "100000", "--only-class", "1", "--K", "1", "--max-groups", "2", "--tag", "_group_mi", "--extra-judges", "cnn") "mi_bank"
Run "results\agent\$N\test.jsonl" @("scripts\make_views.py", "--dataset", "ptbxl", "--claims", "group", "--parts", "test", "--name", $N) "mi_views"
"done" | Out-File "$L\mi_views.done"
foreach ($c in "verified", "heuristic") {
  Run "results\agent\$N\pred_policy_$c.json" @("scripts\train_policy.py", "--dataset", "ptbxl_group", "--claim", $c, "--seed", "0", "--also-test", $N) "mi_policy_$c"
  foreach ($s in 1, 2) {
    Run "results\agent\$N\pred_policy_${c}_s$s.json" @("scripts\train_policy.py", "--dataset", "ptbxl_group", "--claim", $c, "--seed", "$s", "--tag", "_s$s", "--also-test", $N) "mi_policy_${c}_s$s"
  }
}
"done" | Out-File "$L\mi.done"
