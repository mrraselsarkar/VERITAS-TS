# Sensitivity of the checker (hybrid edit) on the planted benchmark: noise strength s,
# guidance weight w and number of edits m. One setting changes at a time; defaults s=0.5, w=2, m=2.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
$runs = @(
  @("0.25", "2", "2"), @("0.75", "2", "2"), @("1.0", "2", "2"),
  @("0.5", "1", "2"), @("0.5", "4", "2"),
  @("0.5", "2", "1"), @("0.5", "2", "4")
)
foreach ($r in $runs) {
  $s, $w, $m = $r
  $tag = "_sens_s${s}_w${w}_m${m}"
  if (Test-Path "results\verifier\synth_planted_eval_seed0$tag.json") { continue }
  & $py -u scripts\eval_verifier_synth.py --dataset synth_planted --n 100 --m $m --methods hybrid:prior+learned --strength $s --guidance $w --out-tag $tag *>> "$L\sensitivity.log"
}
"done" | Out-File "$L\sensitivity.done"
