# Review round, GPU queue 3 (after queue 2): independent-judge banks for the new datasets.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
while (-not (Test-Path "$L\queue2.done")) { Start-Sleep 60 }
foreach ($j in @(@("ptbxl", "_group"), @("ucihar_subj", "_group"))) {
  $ds, $tag = $j
  if (-not (Test-Path "results\bank\$ds$tag\test_judge2.npz")) {
    & $py -u scripts\build_bank.py --dataset $ds --part test --n 400 --K 1 --max-groups 2 --tag $tag --oracles cnn --out-name test_judge2.npz *>> "$L\bank_${ds}_judge2.log"
  }
  & $py -u scripts\judge2_eval.py --dataset "$ds$tag" --runs grpo_accuracy_ver sft_verified policy_heuristic policy_verified grpo_verified_ver *>> "$L\judge2_$ds.log"
}
"done" | Out-File "$L\queue3.done"
