# Review round 6, CPU part: cross-fitted oracles (5 folds) for the stricter check.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review6"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
foreach ($ds in "natops", "cricket", "ucihar_subj", "synth_planted", "pamap2", "ptbxl") {
  Run "results\diffusion\$ds\standard\xf\oracle_fold4.pkl" @("scripts\train_oracles_xf.py", "--dataset", $ds) "xf_oracles_$ds"
  "done" | Out-File "$L\xf_oracles_$ds.done"
}
"done" | Out-File "$L\cpu.done"
