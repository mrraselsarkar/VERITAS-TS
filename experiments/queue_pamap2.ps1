# PAMAP2 agent pipeline with sensor-level claims (one or two channel groups over the
# whole series): waits for the sensor-level bank, renders views, runs the heuristic
# SFT warm start and then the shared agent queue under the name pamap2_group.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
while (-not ((Test-Path results\bank\pamap2_group\train.npz) -and (Test-Path results\bank\pamap2_group\test.npz))) {
  Start-Sleep 60
}
if (-not (Test-Path results\agent\pamap2_group\train.jsonl)) {
  & $py -u scripts\make_views.py --dataset pamap2 --claims group *> results\logs\agent_pamap2_group_views.log
}
if (-not (Test-Path results\agent\pamap2_group\runs\sft\adapter_model.safetensors)) {
  & $py -u scripts\train_agent.py sft --dataset pamap2_group --epochs 2 *> results\logs\agent_pamap2_group_sft.log
}
& powershell -NoProfile -ExecutionPolicy Bypass -File experiments\queue_agent.ps1 -Dataset pamap2_group -Steps 100 -Alpha 0.05
