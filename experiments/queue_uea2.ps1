# UEA datasets (NATOPS, Cricket), stage 2: sensor-claim banks (test, judge 2), agent views,
# non-LLM agents (3 seeds), then -- once the GPU is free of the occlusion ablation -- SFT warm start,
# GRPO with both rewards (3 seeds), evaluation, analysis and the independent judge.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\uea"
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
$sets = @(@("natops", "180", "2"), @("cricket", "72", "1"))
foreach ($s in $sets) {
  $DS, $n_te, $mg = $s
  $N = "${DS}_group"; $B = "results\bank\$N"
  Run "$B\test.npz" @("scripts\build_bank.py", "--dataset", $DS, "--part", "test", "--n", $n_te, "--K", "1", "--max-groups", $mg, "--tag", "_group", "--oracles", "multirocket_hydra", "quant") "bank_${DS}_group_test"
  Run "$B\test_judge2.npz" @("scripts\build_bank.py", "--dataset", $DS, "--part", "test", "--n", $n_te, "--K", "1", "--max-groups", $mg, "--tag", "_group", "--out-name", "test_judge2.npz", "--oracles", "cnn") "bank_${DS}_group_judge2"
  Run "results\agent\$N\train.jsonl" @("scripts\make_views.py", "--dataset", $DS, "--claims", "group") "views_$N"
  foreach ($seed in 0, 1, 2) {
    $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
    foreach ($c in "verified", "heuristic") {
      Run "results\agent\$N\pred_policy_$c$sfx.json" @("scripts\train_policy.py", "--dataset", $N, "--claim", $c, "--seed", "$seed", "--tag", $sfx) "policy_${N}_$c$sfx"
    }
  }
}
# the LLM parts need the whole GPU: wait for the occlusion ablation
while (-not (Test-Path "results\logs\review\occl_reward.done")) { Start-Sleep 60 }
foreach ($s in $sets) {
  $DS = $s[0]; $N = "${DS}_group"; $A = "results\agent\$N"
  Run "$A\runs\sft_verified\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $N, "--targets", "verified", "--epochs", "2") "sft_verified_$N"
  Run "$A\pred_sft_verified.json" @("scripts\eval_agent.py", "--dataset", $N, "--run", "sft_verified") "eval_${N}_sft_verified"
  foreach ($seed in 0, 1, 2) {
    $sfx = if ($seed -eq 0) { "" } else { "_s$seed" }
    foreach ($r in "accuracy", "verified") {
      Run "$A\runs\grpo_${r}_ver$sfx\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver$sfx", "--seed", "$seed") "grpo_${r}_ver${sfx}_$N"
      Run "$A\pred_grpo_${r}_ver$sfx.json" @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_${r}_ver$sfx") "eval_${N}_grpo_${r}_ver$sfx"
    }
  }
  Run $null @("scripts\analyze_agent.py", "--dataset", $N, "--alpha", "0.1") "analyze_$N"
  Run $null @("scripts\judge2_eval.py", "--dataset", $N, "--runs", "grpo_accuracy_ver", "grpo_verified_ver", "policy_heuristic", "policy_verified", "sft_verified") "judge2_$N"
}
"done" | Out-File "$L\uea2.done"
