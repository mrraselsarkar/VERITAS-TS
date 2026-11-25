# Review round 5, CPU part: zero-masking banks on the real data (judged by the oracle and by the
# independent CNN judge) and the non-LLM agent without neighbour classes.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review5"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
# 1. zero-masking test banks (oracle and CNN judge) for every real dataset
$real = @(@("ucihar_subj", 400, 2), @("ptbxl", 400, 2), @("pamap2", 400, 2), @("natops", 180, 2), @("cricket", 72, 1))
foreach ($d in $real) {
  $ds, $n, $mg = $d[0], "$($d[1])", "$($d[2])"
  $B = "results\bank\${ds}_group_zero"
  Run "$B\test.npz" @("scripts\build_bank.py", "--dataset", $ds, "--part", "test", "--n", $n, "--K", "1", "--max-groups", $mg, "--tag", "_group_zero", "--method", "zero") "zero_test_$ds"
  Run "$B\test_judge2.npz" @("scripts\build_bank.py", "--dataset", $ds, "--part", "test", "--n", $n, "--K", "1", "--max-groups", $mg, "--tag", "_group_zero", "--method", "zero", "--oracles", "cnn", "--out-name", "test_judge2.npz") "zero_judge2_$ds"
}
# 2. zero-masking training banks for the two datasets of the masking-reward experiment
foreach ($ds in "ucihar_subj", "ptbxl") {
  $B = "results\bank\${ds}_group_zero"
  Run "$B\train.npz" @("scripts\build_bank.py", "--dataset", $ds, "--part", "train", "--n", "1000", "--K", "1", "--max-groups", "2", "--tag", "_group_zero", "--method", "zero") "zero_train_$ds"
}
"banks done" | Out-File "$L\cpu_banks.done"
# 3. non-LLM agent without neighbour classes, both claim modes, three seeds
foreach ($ds in "synth_planted", "pamap2_group", "ptbxl_group", "ucihar_subj_group", "natops_group", "cricket_group") {
  foreach ($c in "verified", "heuristic") {
    foreach ($s in 0, 1, 2) {
      $tag = if ($s -eq 0) { "_novote" } else { "_novote_s$s" }
      Run "results\agent\$ds\pred_policy_$c$tag.json" @("scripts\train_policy.py", "--dataset", $ds, "--claim", $c, "--seed", "$s", "--tag", $tag, "--no-votes") "novote_${ds}_${c}_s$s"
    }
  }
}
"done" | Out-File "$L\cpu.done"
