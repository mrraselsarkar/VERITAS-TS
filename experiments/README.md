# Experiment Queues

These are the exact experiment queues that produced the results of the paper, kept for provenance. They
record the order, arguments and seeds of every run.

**How they work**

- The queues are Windows PowerShell 5.1 scripts (two early ones are Bash). Each one changes to the repository
  root and calls the Python entry points in [`../scripts`](../scripts) with `.venv\Scripts\python.exe`.
- A step is skipped when its output file already exists, so a queue can be stopped and restarted at any time.
- Some queues wait for files written by an earlier queue, or for free GPU memory, before they start a step.
- Logs are written to `results/logs/`.

```powershell
# example: the full agent pipeline for one experiment (after its bank and views exist)
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\queue_agent.ps1 -Dataset synth_planted -Steps 200 -Alpha 0.05
```

On Linux or macOS, the same Python commands can be run directly; every queue is a short list of them.

## Experiment names

| Name | Data | Claims |
|---|---|---|
| `synth_planted` | synthetic benchmark with planted evidence | one cell (time window x channel group) |
| `pamap2_group` | PAMAP2 | one or two whole sensors |
| `ptbxl_group` | PTB-XL | one or two lead groups |
| `ucihar_subj_group` | UCI-HAR, with a test set of new people (`_shift`) | one or two whole sensors |
| `natops_group`, `cricket_group` | UEA NATOPS and Cricket | one or two sensors (Cricket: one wrist) |
| `<name>_hint` | the same, with the classifier hint in the text view | as above |

Run names: `sft_diverse` and `sft_verified` are the warm starts; `grpo_verified_*` uses the **check reward**
and `grpo_accuracy_*` the accuracy reward; `_s1` and `_s2` mark seeds 1 and 2; `policy_verified` and
`policy_heuristic` are the non-LLM agent with the check reward and with rule-based claims.

## Queues by paper section

| Queue | Purpose | Paper |
|---|---|---|
| `gpu_queue_phase2.sh`, `gpu_queue_phase2b.sh` | diffusion models; comparison of editing methods on synthetic and real data | Q1 |
| `queue_sensitivity.ps1` | sensitivity of the check to noise level, guidance weight and number of edits | Q1 |
| `queue_banks.ps1`, `queue_group_bank.ps1` | counterfactual banks (synthetic cells, PAMAP2 sensors) | Q2 |
| `queue_agent.ps1`, `queue_pamap2.ps1`, `queue_extra.ps1` | warm starts, GRPO with both rewards, non-LLM agents, analysis | Q2 |
| `queue_review1.ps1` ... `queue_review4.ps1`, `queue_seeds2.ps1` | PTB-XL and UCI-HAR agents, independent judges, new people, seeds 1-2 | Q2, Q3 |
| `queue_uea1.ps1`, `queue_uea2.ps1` | NATOPS and Cricket: models, banks, agents (three seeds) | Q2 |
| `queue_zero_reward.ps1`, `queue_occl_reward.ps1` | reward ablations: masking-check reward, occlusion reward | Q2 |
| `queue_review5_cpu.ps1`, `queue_review5_gpu.ps1` | masking reward on real data, swapped oracle and judge, agents without neighbor classes | Q2, Q3 |
| `queue_review6_*.ps1` | stricter check with cross-fitted oracles, subject-disjoint test sets, myocardial-infarction anatomy, agent confidence | Q3, Q5 |
| `queue_hint.ps1` | agents with the classifier hint (three seeds, both rewards) | Q4 |
| `queue_shift.ps1`, `queue_retry.ps1` | abstention under shift, verify-and-revise | Q5 |

The analysis scripts that turn these runs into the numbers of the paper are listed in the main
[README](../README.md#reproducing-the-paper).
