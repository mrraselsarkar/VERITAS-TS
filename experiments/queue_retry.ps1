# Verify-and-revise (retry) evaluation for the main agent configurations.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$jobs = @(
  @("synth_planted", "grpo_verified_div", ""),
  @("synth_planted", "grpo_accuracy_div", ""),
  @("synth_planted", "sft_diverse", ""),
  @("synth_planted_shift", "grpo_verified_div", "synth_planted"),
  @("synth_planted_shift", "grpo_accuracy_div", "synth_planted"),
  @("pamap2_group", "grpo_verified_ver", ""),
  @("pamap2_group", "grpo_accuracy_ver", "")
)
foreach ($j in $jobs) {
  $ds, $run, $from = $j
  if (Test-Path "results\agent\$ds\retry_$run.json") { continue }
  $extra = @(); if ($from) { $extra = @("--adapters-from", $from) }
  & $py -u scripts\eval_retry.py --dataset $ds --run $run --k 4 @extra *>> "results\logs\retry_${ds}_$run.log"
}
"done" | Out-File results\logs\retry.done
