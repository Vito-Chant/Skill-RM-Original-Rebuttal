# Codex Execution Plan: Matched Compute Follow-Up

## Objective

Run exactly six full Qwen3.5-27B jobs and return complete sanitized outputs to the paper author. The coauthor has already started vLLM 0.17.0-compatible servers and supplied their URLs in the message linking this document. Continue through endpoint validation, inference, summary, packaging, and delivery verification; do not stop after giving a plan.

The six jobs are:

- Direct Judge and standard-input Skill-RM
- RewardBench2, RM-Bench, and JudgeBench

Do not run resource subsets, the resource-available setting, older ablations, another backbone, BoN, IF-RewardBench, IF-RL, a trace audit, or any training job.

## 1. Clone and install

```bash
git clone https://github.com/Vito-Chant/Skill-RM-Original-Rebuttal.git
cd Skill-RM-Original-Rebuttal
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q
```

Use Python 3.10 or newer. Do not commit outputs, data, endpoints, credentials, or annotation material.

## 2. Configure and validate vLLM endpoints

Normalize the URLs from the coauthor's message without persisting or echoing them:

```bash
export SKILLRM_BASE_URLS="$(python scripts/normalize_vllm_urls.py '<URLs from the coauthor message>')"
export SKILLRM_MODEL="Qwen3.5-27B"
export SKILLRM_WORKERS="${SKILLRM_WORKERS:-80}"
export NO_PROXY='*'
export no_proxy='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

Check every endpoint. Both ordinary chat and forced tool calls must return exact `prompt_tokens`, `completion_tokens`, and `total_tokens` in the non-streaming OpenAI-compatible `usage` object:

```bash
python scripts/check_endpoints.py --chat --tools --require-usage --redact-urls
```

Do not run the full experiment if usage is missing or inconsistent. Remove only failed endpoint indices and repeat the check. At least one endpoint must pass all checks.

## 3. Prepare the pinned data

Reuse an existing validated data tree when available:

```bash
export SKILLRM_DATA_ROOT=/path/to/OpenRS/data
python scripts/prepare_rebuttal_data.py --output "$SKILLRM_DATA_ROOT" --validate-only
```

Otherwise prepare the pinned revision in the ignored local directory:

```bash
python scripts/prepare_rebuttal_data.py --output data/openrs
export SKILLRM_DATA_ROOT="$PWD/data/openrs"
```

Do not substitute another dataset revision.

## 4. Dry-run and execute

Use a new persistent terminal or `tmux` session. First verify exactly six jobs and no model traffic:

```bash
DRY_RUN=1 bash scripts/run_compute_comparison.sh
```

Then launch the full workflow:

```bash
bash scripts/run_compute_comparison.sh
```

The default output root is `outputs/compute_comparison/qwen35_27b`. It must be empty before launch so older non-instrumented rows cannot mix with this run. If the workflow is interrupted, preserve the failed directory for diagnosis, choose a new output root, and restart all six jobs; do not manually splice JSONL files.

The runner automatically:

- verifies the tracked source hashes;
- repeats the exact-usage endpoint preflight;
- validates the pinned datasets;
- runs Direct Judge then Skill-RM for each benchmark with the same endpoints and workers;
- records full predictions, raw outputs, Skill-RM traces, per-request usage, metrics, configs, logs, and job durations;
- summarizes token/call/latency/wall-clock costs;
- recursively sanitizes network and path metadata;
- builds and verifies both private delivery archives.

## 5. Deliver complete outputs

Do not perform model-based or human trace evaluation in the company workspace. The only files to send privately to the paper author are:

- `skillrm_compute_summary_qwen35_27b.tar.gz`
- `skillrm_compute_full_analysis_qwen35_27b.tar.gz`

They are written under:

`outputs/compute_comparison/qwen35_27b/delivery`

The full-analysis archive must retain all six predictions, all Skill-RM traces, every model step's usage/tool observations/raw output, metrics, resolved configs, attempts, and logs. It must not contain source datasets, endpoints, IP addresses, absolute paths, credentials, virtual environments, Git history, or vLLM server logs.

Report the absolute path, byte size, and SHA256 of both archives. If blocked, report the exact failed step and a redacted error after endpoint, data, and fresh-output checks have been attempted.
