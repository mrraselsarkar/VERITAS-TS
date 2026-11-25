# Review round, GPU queue 4 (after queue 3): PAMAP2 with a real subject shift (pamap2_subj).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$L = "results\logs\review"
while (-not (Test-Path "$L\queue3.done")) { Start-Sleep 60 }
function Run($done, [string[]]$cmd, $log) {
  if ($done -and (Test-Path $done)) { return }
  & $py -u @cmd *>> "$L\$log.log"
}
$DS = "pamap2_subj"; $N = "${DS}_group"; $S = "${N}_shift"; $D = "results\diffusion\$DS\standard"; $B = "results\bank"
Run "$D\prior+learned.pt" @("scripts\train_diffusion.py", "--dataset", $DS, "--channels", "48", "--layers", "10", "--batch-size", "32", "--iters", "3000", "--sensor-masks") "diff_$DS"
Run "$D\oracle_cnn.pkl" @("scripts\train_oracles.py", "--dataset", $DS) "oracles_$DS"
foreach ($j in @(@("test", 400, "standard", "_group", "test.npz", "multirocket_hydra quant"), @("train", 1000, "standard", "_group", "train.npz", "multirocket_hydra quant"),
                 @("test", 400, "shift", "_group_shift", "test.npz", "multirocket_hydra quant"), @("test", 400, "standard", "_group", "test_judge2.npz", "cnn"))) {
  $part, $count, $split, $tag, $out, $orc = $j
  Run "$B\$DS$tag\$out" (@("scripts\build_bank.py", "--dataset", $DS, "--part", $part, "--n", "$count", "--split", $split, "--K", "1", "--max-groups", "2", "--tag", $tag, "--out-name", $out, "--oracles") + $orc.Split(" ")) "bank_$DS${tag}_$out"
}
$A = "results\agent\$N"
Run "$A\train.jsonl" @("scripts\make_views.py", "--dataset", $DS, "--claims", "group") "views_$N"
Run "$A\runs\sft_verified\adapter_model.safetensors" @("scripts\train_agent.py", "sft", "--dataset", $N, "--targets", "verified", "--epochs", "2") "sft_verified_$N"
foreach ($r in @("sft_verified")) { Run "$A\pred_$r.json" @("scripts\eval_agent.py", "--dataset", $N, "--run", $r) "eval_${N}_$r" }
foreach ($r in @("verified", "accuracy")) {
  Run "$A\runs\grpo_${r}_ver\adapter_model.safetensors" @("scripts\train_agent.py", "grpo", "--dataset", $N, "--reward", $r, "--steps", "100", "--init", "sft_verified", "--tag", "_ver") "grpo_${r}_ver_$N"
  Run "$A\pred_grpo_${r}_ver.json" @("scripts\eval_agent.py", "--dataset", $N, "--run", "grpo_${r}_ver") "eval_${N}_grpo_${r}_ver"
}
foreach ($c in @("verified", "heuristic")) { Run "$A\pred_policy_$c.json" @("scripts\train_policy.py", "--dataset", $N, "--claim", $c) "policy_${c}_$N" }
Run $null @("scripts\analyze_agent.py", "--dataset", $N, "--alpha", "0.1") "analyze_$N"
Run "results\agent\$S\test.jsonl" @("scripts\make_views.py", "--dataset", $DS, "--claims", "group", "--split", "shift", "--parts", "test") "views_$S"
foreach ($r in @("sft_verified", "grpo_accuracy_ver", "grpo_verified_ver")) {
  Run "results\agent\$S\pred_$r.json" @("scripts\eval_agent.py", "--dataset", $S, "--run", $r, "--adapters-from", $N) "eval_${S}_$r"
}
Run $null @("scripts\shift_eval.py", "--dataset", $N, "--runs", "sft_verified", "grpo_accuracy_ver", "grpo_verified_ver", "--alpha", "0.1") "shift_eval_$N"
Run "$A\retry_grpo_verified_ver.json" @("scripts\eval_retry.py", "--dataset", $N, "--run", "grpo_verified_ver", "--k", "4") "retry_$N"
Run "results\agent\$S\retry_grpo_verified_ver.json" @("scripts\eval_retry.py", "--dataset", $S, "--run", "grpo_verified_ver", "--k", "4", "--adapters-from", $N) "retry_$S"
Run $null @("scripts\judge2_eval.py", "--dataset", $N, "--runs", "grpo_accuracy_ver", "sft_verified", "policy_heuristic", "policy_verified", "grpo_verified_ver") "judge2_$DS"
"done" | Out-File "$L\queue4.done"
