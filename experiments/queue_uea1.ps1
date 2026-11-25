# UEA datasets (NATOPS, Cricket), stage 1: diffusion models, oracles, and training banks for
# both claim types (cells and sensors) so that the claim-type rule can be applied.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\uea"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
foreach ($DS in "natops", "cricket") {
  $D = "results\diffusion\$DS\standard"
  Run "$D\prior+learned.pt" @("scripts\train_diffusion.py", "--dataset", $DS, "--channels", "48", "--layers", "10", "--batch-size", "32", "--iters", "3000", "--sensor-masks") "diff_$DS"
  Run "$D\oracle_cnn.pkl" @("scripts\train_oracles.py", "--dataset", $DS) "oracles_$DS"
  $n_tr = if ($DS -eq "natops") { 180 } else { 108 }
  $n_te = if ($DS -eq "natops") { 180 } else { 72 }
  # claim-type rule: share of training recordings with a passing claim, cells first
  Run "results\bank\$DS\train.npz" @("scripts\build_bank.py", "--dataset", $DS, "--part", "train", "--n", "$n_tr", "--K", "8", "--max-groups", "1", "--oracles", "multirocket_hydra", "quant") "bank_${DS}_cell_train"
  $mg = if ($DS -eq "natops") { "2" } else { "1" }
  Run "results\bank\${DS}_group\train.npz" @("scripts\build_bank.py", "--dataset", $DS, "--part", "train", "--n", "$n_tr", "--K", "1", "--max-groups", $mg, "--tag", "_group", "--oracles", "multirocket_hydra", "quant") "bank_${DS}_group_train"
}
"done" | Out-File "$L\uea1.done"
