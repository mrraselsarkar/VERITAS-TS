# Agents with the classifier hint (cross-fitted InceptionTime probabilities in the text view).
# Pilot first (PTB-XL and NATOPS, seed 0, both rewards), then the other datasets, then seeds 1-2.
# Steps whose output exists are skipped, so the queue can be relaunched after a restart.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\hint"
New-Item -ItemType Directory -Force $L | Out-Null
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
# experiment -> source dataset, claim type
$src = @{ "synth_planted" = @("synth_planted", "cell"); "pamap2_group" = @("pamap2", "group");
          "ptbxl_group" = @("ptbxl", "group"); "ucihar_subj_group" = @("ucihar_subj", "group");
          "natops_group" = @("natops", "group"); "cricket_group" = @("cricket", "group") }

function Prepare($N) {
  $S, $C = $src[$N]; $H = "${N}_hint"
  Run "results\hint\$N.npz" @("scripts\make_hint.py", "--dataset", $N) "hint_$N"
  if (-not (Test-Path "results\bank\$H\test.npz")) {
    New-Item -ItemType Directory -Force "results\bank\$H" | Out-Null
    Copy-Item "results\bank\$N\*.npz" "results\bank\$H\"
  }
  Run "results\agent\$H\features.npz" @("scripts\make_views.py", "--dataset", $S, "--claims", $C, "--name", $H, "--hint-from", $N) "views_$H"
  foreach ($seed in 0, 1, 2) {
    $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
    foreach ($c in "verified", "heuristic") {
      # PowerShell 5.1 drops an empty argument, so seed 0 passes no --tag at all
      $tagArg = if ($sfx) { @("--tag", $sfx) } else { @() }
      Run "results\agent\$H\pred_policy_$c$sfx.json" (@("scripts\train_policy.py", "--dataset", $H, "--claim", $c, "--seed", "$seed") + $tagArg) "policy_${H}_$c$sfx"
    }
  }
}

function Llm($N, $seed, [string[]]$rewards) {
  $H = "${N}_hint"; $A = "results\agent\$H"
  if ($N -eq "synth_planted") { $init = "sft_diverse"; $tg = "diverse"; $ep = "1"; $tag = "_div"; $steps = "200" }
  else { $init = "sft_verified"; $tg = "verified"; $ep = "2"; $tag = "_ver"; $steps = "100" }
  Run "$A\runs\$init\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $H, "--targets", $tg, "--epochs", $ep) "sft_$H"
  Run "$A\pred_$init.json" @("scripts\eval_agent.py", "--dataset", $H, "--run", $init) "eval_${H}_$init"
  $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
  foreach ($r in $rewards) {
    $run = "grpo_${r}${tag}$sfx"
    Run "$A\runs\$run\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $H, "--reward", $r, "--steps", $steps, "--init", $init, "--tag", "$tag$sfx", "--seed", "$seed") "${run}_$H"
    Run "$A\pred_$run.json" @("scripts\eval_agent.py", "--dataset", $H, "--run", $run) "eval_${H}_$run"
  }
}

$pilot = @("ptbxl_group", "natops_group")
$rest = @("synth_planted", "pamap2_group", "ucihar_subj_group", "cricket_group")
foreach ($N in $pilot) { Prepare $N }
foreach ($N in $pilot) { Llm $N 0 @("verified") }
foreach ($N in $pilot) { Llm $N 0 @("accuracy") }
& $py -u scripts\hint_summary.py --datasets @pilot *>> "$L\summary_pilot.log"
"done" | Out-File "$L\pilot.done"
foreach ($N in $rest) { Prepare $N }
foreach ($N in $rest) { Llm $N 0 @("verified", "accuracy") }
foreach ($seed in 1, 2) {
  foreach ($N in $pilot + $rest) { Llm $N $seed @("verified", "accuracy") }
}
& $py -u scripts\hint_summary.py *>> "$L\summary_all.log"
"done" | Out-File "$L\all.done"
