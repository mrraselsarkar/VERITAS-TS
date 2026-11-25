# RQ4 under shift: bank and views for the shifted synthetic test set, evaluation of the
# trained synthetic agents on it, and threshold transfer from in-distribution calibration.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$S = "synth_planted_shift"
if (-not (Test-Path "results\bank\$S\test.npz")) {
  & $py -u scripts\build_bank.py --dataset synth_planted --split shift --part test --n 800 --tag _shift *>> "results\logs\bank_${S}_test.log"
}
if (-not (Test-Path "results\agent\$S\test.jsonl")) {
  & $py -u scripts\make_views.py --dataset synth_planted --split shift --parts test *> "results\logs\agent_${S}_views.log"
}
$runs = @("sft_diverse", "grpo_accuracy_div", "grpo_verified_div")
foreach ($r in $runs) {
  if (-not (Test-Path "results\agent\$S\pred_$r.json")) {
    & $py -u scripts\eval_agent.py --dataset $S --run $r --adapters-from synth_planted *>> "results\logs\agent_${S}_eval.log"
  }
}
& $py -u scripts\shift_eval.py --dataset synth_planted --runs @runs *> "results\logs\agent_${S}_summary.log"
