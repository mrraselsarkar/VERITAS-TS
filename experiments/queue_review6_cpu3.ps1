# Review round 6, third CPU worker: stricter-check training banks of PAMAP2 and PTB-XL, in parallel with
# queue_review6_cpu2.ps1, which builds the test banks (lock files keep the queues from building the same bank).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$env:CUDA_VISIBLE_DEVICES = "-1"  # an empty value would delete the variable in PowerShell
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review6"
function RunLocked($done, [string[]]$cmd, $log) {
  if ((Test-Path $done) -or (Test-Path "$done.lock")) { return }
  "cpu3" | Out-File "$done.lock"
  try {
    "[$(Get-Date -Format s)] start (cpu3)" | Out-File -Append "$L\$log.log"
    & $py -u @cmd *>> "$L\$log.log"
  } finally { Remove-Item "$done.lock" -ErrorAction SilentlyContinue }
}
function WaitFor($path) { while (-not (Test-Path $path)) { Start-Sleep -Seconds 60 } }
foreach ($x in @(@("pamap2_group", "pamap2"), @("ptbxl_group", "ptbxl"))) {
  $N, $ds = $x
  WaitFor "$L\xf_oracles_$ds.done"
  $common = @("--dataset", $ds, "--K", "1", "--max-groups", "2", "--tag", "_group_xf", "--crossfit", "--suf-donors", "other", "--chunk", "20")
  RunLocked "results\bank\${N}_xf\train.npz" (@("scripts\build_bank.py", "--part", "train", "--n", "1000", "--extra-judges", "full") + $common) "xf_bank_${N}_train"
}
"done" | Out-File "$L\cpu3.done"
