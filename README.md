# Skill-RM Original v6 Rebuttal Workflow

This repository contains the archived Skill-RM v6 implementation used for the paper experiments and a narrowly scoped workflow for the EMNLP rebuttal resource-completeness analysis.

The new experiment is a matched comparison of the full fair resource pool and two complementary 50% resource subsets on RewardBench2, RM-Bench, and JudgeBench. It does not rerun the paper baseline, operational setting, prior ablations, additional backbones, or RL experiments.

## Codex entry point

The complete unattended execution instructions are in [docs/codex_rebuttal_experiment_plan.md](docs/codex_rebuttal_experiment_plan.md). A coauthor only needs to provide working Qwen3.5-27B vLLM URLs to a new Codex session and point it at that document.

The later matched compute follow-up, which records exact vLLM usage and returns all Direct Judge/Skill-RM outputs and traces for local analysis, is documented in [docs/codex_compute_followup_plan.md](docs/codex_compute_followup_plan.md). Its ready-to-send coauthor prompt is in [docs/coauthor_compute_prompt.md](docs/coauthor_compute_prompt.md).

Historical standard-input and resource-enhanced traces can be analyzed locally with `scripts/analyze_historical_joint_traces.py`. It produces separated full-trace statistics and a 30-case, outcome-blinded two-author annotation package without sending historical traces to a model endpoint.

## Local verification

```bash
python -m pip install -e '.[test]'
pytest -q
DRY_RUN=1 SKILLRM_DATA_ROOT=/tmp/skillrm-placeholder SKILLRM_BASE_URLS=http://127.0.0.1:8000/v1 bash scripts/run_original_v6_rebuttal.sh
```

The dry run prepares configs and prints exactly nine redacted commands. It sends no model requests and does not require the placeholder dataset files to exist.

## Data and outputs

Benchmark data, experiment outputs, endpoints, logs, audit cases, and delivery archives are excluded from Git. The workflow downloads the four pinned OpenRS JSONL files when `SKILLRM_DATA_ROOT` is not supplied. Final delivery archives are intended for private transfer only.

## Provenance

`ORIGINAL_V6_SHA256SUMS` records the copied v6 core files. Rebuttal scripts validate those hashes before running. The orchestration layer does not modify the original inference modules.

Licensed under Apache-2.0.
