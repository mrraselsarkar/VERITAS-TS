# Phase 3 GPU queue: counterfactual banks, run one after another (they share the GPU).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results\logs | Out-Null
$jobs = @(
  @("synth_planted", "test", 720),
  @("synth_planted", "train", 1680),
  @("pamap2", "test", 400),
  @("pamap2", "train", 1000)
)
foreach ($j in $jobs) {
  $ds, $part, $n = $j
  if (Test-Path "results\bank\$ds\$part.npz") { continue }
  & $py -u scripts\build_bank.py --dataset $ds --part $part --n $n *>> "results\logs\bank_${ds}_${part}.log"
}
"queue done" | Out-File results\logs\queue_banks.done
