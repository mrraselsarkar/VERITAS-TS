# Follow-up for one agent experiment: both GRPO rewards starting from the verifier-filtered
# warm start (sft_verified), for data where the diverse warm start never proposes the
# claims that verify. Waits for the main queue of that experiment to finish.
param([string]$Dataset = "pamap2_group", [int]$Steps = 100, [double]$Alpha = 0.05)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
$A = "results\agent\$Dataset"
$log = "results\logs\agent_$Dataset"
while (-not (Test-Path "${log}_queue.done")) { Start-Sleep 60 }
foreach ($r in @("verified", "accuracy")) {
  if (-not (Test-Path "$A\runs\grpo_${r}_ver\adapter_model.safetensors")) {
    & $py -u scripts\train_agent.py grpo --dataset $Dataset --reward $r --steps $Steps --init sft_verified --tag _ver *>> "${log}_grpo_${r}_ver.log"
  }
  if (-not (Test-Path "$A\pred_grpo_${r}_ver.json")) {
    & $py -u scripts\eval_agent.py --dataset $Dataset --run "grpo_${r}_ver" *>> "${log}_eval_grpo_${r}_ver.log"
  }
}
& $py -u scripts\analyze_agent.py --dataset $Dataset --alpha $Alpha *> "${log}_analyze.log"
"done" | Out-File "${log}_extra.done"
