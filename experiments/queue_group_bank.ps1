# Sensor-level (whole-series, 1-2 channel groups) counterfactual bank for PAMAP2.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
foreach ($j in @(@("test", 400), @("train", 1000))) {
  $part, $n = $j
  if (Test-Path "results\bank\pamap2_group\$part.npz") { continue }
  & $py -u scripts\build_bank.py --dataset pamap2 --part $part --n $n --K 1 --max-groups 2 --tag _group --chunk 20 *>> "results\logs\bank_pamap2_group_$part.log"
}
