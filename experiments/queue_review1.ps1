# Review round, GPU queue 1: new models and banks.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
$D = "results\diffusion"
# 1. UCI-HAR with a real subject shift: diffusion model, then the judges
Run "$D\ucihar_subj\standard\prior+learned.pt" @("scripts\train_diffusion.py", "--dataset", "ucihar_subj", "--channels", "48", "--layers", "10", "--batch-size", "32", "--iters", "3000") "diff_ucihar_subj"
Run "$D\ucihar_subj\standard\oracle_cnn.pkl" @("scripts\train_oracles.py", "--dataset", "ucihar_subj") "oracles_ucihar_subj"
# 2. independent CNN judges
foreach ($ds in @("synth_planted", "pamap2", "ptbxl")) {
  Run "$D\$ds\standard\oracle_cnn.pkl" @("scripts\train_oracles.py", "--dataset", $ds, "--members", "cnn") "cnn_$ds"
}
# 3. PAMAP2 diffusion model that has also seen whole-sensor masks
Run "$D\pamap2\standard\prior+learned+sensor.pt" @("scripts\train_diffusion.py", "--dataset", "pamap2", "--channels", "48", "--layers", "10", "--batch-size", "32", "--iters", "3000", "--sensor-masks", "--name", "prior+learned+sensor") "diff_pamap2_sensor"
# 4. banks
$B = "results\bank"
Run "$B\synth_planted\test_judge2.npz" @("scripts\build_bank.py", "--dataset", "synth_planted", "--part", "test", "--n", "720", "--oracles", "cnn", "--out-name", "test_judge2.npz") "bank_synth_judge2"
Run "$B\pamap2_group\test_judge2.npz" @("scripts\build_bank.py", "--dataset", "pamap2", "--part", "test", "--n", "400", "--K", "1", "--max-groups", "2", "--tag", "_group", "--oracles", "cnn", "--out-name", "test_judge2.npz") "bank_pamap2_judge2"
Run "$B\pamap2_group\test_sensormask.npz" @("scripts\build_bank.py", "--dataset", "pamap2", "--part", "test", "--n", "400", "--K", "1", "--max-groups", "2", "--tag", "_group", "--diffusion-name", "prior+learned+sensor", "--out-name", "test_sensormask.npz") "bank_pamap2_sensormask"
foreach ($j in @(@("ptbxl", "test", 400, "standard", "_group"), @("ptbxl", "train", 1000, "standard", "_group"),
                 @("ucihar_subj", "test", 400, "standard", "_group"), @("ucihar_subj", "train", 1000, "standard", "_group"),
                 @("ucihar_subj", "test", 400, "shift", "_group_shift"))) {
  $ds, $part, $n, $split, $tag = $j
  Run "$B\$ds$tag\$part.npz" @("scripts\build_bank.py", "--dataset", $ds, "--part", $part, "--n", "$n", "--split", $split, "--K", "1", "--max-groups", "2", "--tag", $tag) "bank_$ds${tag}_$part"
}
"done" | Out-File "$L\queue1.done"
