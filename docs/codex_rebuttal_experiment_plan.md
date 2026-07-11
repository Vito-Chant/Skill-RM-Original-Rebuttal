# Codex Execution Plan: Original v6 Minimal Rebuttal

## Objective

Execute the workflow to completion. Do not stop after producing a plan. The human coauthor has already started Qwen3.5-27B vLLM servers and supplied their URLs in the message that linked this document. The coauthor should not need to participate again unless a real external blocker remains after the recovery steps below.

The only inference jobs are:

- `full_fair`, `subset_a`, and `subset_b`
- RewardBench2, RM-Bench, and JudgeBench
- exactly 9 jobs total

Do not add baseline, operational, older ablations, extra seeds, another backbone, BoN, IF-RewardBench, IF-RL, or RL jobs.

## 1. Clone and set up

Clone the repository and work from its root:

```bash
git clone https://github.com/Vito-Chant/Skill-RM-Original-Rebuttal.git
cd Skill-RM-Original-Rebuttal
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q
```

Use Python 3.10 or newer. Do not place data or outputs in a Git-tracked directory other than the ignored paths already defined by this repository.

## 2. Configure endpoints without persisting them

Normalize the URLs supplied by the coauthor and export them as `SKILLRM_BASE_URLS`. Do not put them in `.env`, YAML, Markdown, shell scripts, Codex messages, or Git. Do not echo the normalized value.

```bash
export SKILLRM_BASE_URLS="$(python scripts/normalize_vllm_urls.py '<URLs from the coauthor message>')"
export SKILLRM_MODEL="Qwen3.5-27B"
export SKILLRM_WORKERS="${SKILLRM_WORKERS:-80}"
```

Unset proxy variables for calls to the company-local vLLM servers, while preserving any proxy settings needed for the initial clone/data download in a separate shell if necessary:

```bash
export NO_PROXY='*'
export no_proxy='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

Check every endpoint using redacted labels. The checks make one tiny chat request and one required tool-call request per endpoint; they are infrastructure checks, not experiment samples.

```bash
python scripts/check_endpoints.py --chat --tools --redact-urls
```

If an endpoint fails, remove only that failed endpoint by index from the environment and rerun the check. Never print or save the endpoint itself. Continue when at least one endpoint passes both checks.

## 3. Prepare data

If a complete OpenRS data tree already exists, export its parent data directory:

```bash
export SKILLRM_DATA_ROOT=/path/to/OpenRS/data
python scripts/prepare_rebuttal_data.py --output "$SKILLRM_DATA_ROOT" --validate-only
```

Otherwise download the pinned files into the ignored local data directory. If internet access requires a proxy, do this before unsetting proxy variables or in a separate shell:

```bash
python scripts/prepare_rebuttal_data.py --output data/openrs
export SKILLRM_DATA_ROOT="$PWD/data/openrs"
```

The preparation command must validate the pinned commit, SHA256, JSON syntax, and row counts. Do not substitute another dataset revision.

## 4. Dry run and launch

First confirm that the runner emits exactly nine commands and no model traffic:

```bash
DRY_RUN=1 bash scripts/run_original_v6_rebuttal.sh
```

Then run the real workflow in a persistent terminal such as `tmux`:

```bash
bash scripts/run_original_v6_rebuttal.sh
```

The runner:

- verifies original-v6 source hashes;
- creates the fixed complementary resource subsets and matched configs;
- writes under `outputs/original_v6_rebuttal/qwen35_27b` only;
- resumes completed work when possible;
- streams redacted logs and removes endpoint/path metadata from completed run files;
- validates all nine outputs;
- writes aggregate and process statistics;
- prepares the blinded 30-case audit input.

If the process is interrupted, run the same command again. If validation reports an incomplete RewardBench2 raw line count after a resume, move only that named run directory to an ignored quarantine directory and rerun that one job from a clean output directory; duplicate RewardBench2 IDs make a partially resumed raw file unsafe. Do not edit predictions by hand.

Do not treat an intermediate request retry as a failed sample when the final prediction is valid. A run is complete only when the expected metric count, `missing=0`, primary metric, raw prediction count, and trace count all validate.

## 5. Perform the blinded model-based trace audit

After all nine jobs pass, read:

- `outputs/original_v6_rebuttal/qwen35_27b/audit/audit_instructions.md`
- `outputs/original_v6_rebuttal/qwen35_27b/audit/audit_input.jsonl`

Do **not** open, search, print, summarize, hash manually, or otherwise inspect `audit_key.jsonl` before all 30 result rows are complete. The input contains anonymous audit IDs, reconstructed visible messages, the recorded trace, and the final verdict, but no gold answer, correctness flag, sample ID, or dataset row index.

Audit every case independently and write exactly 30 JSONL rows to:

`outputs/original_v6_rebuttal/qwen35_27b/audit/audit_results.jsonl`

Each row must follow the schema in `audit_instructions.md`. Do not infer correctness. Assess only whether the visible trace supports its own recorded verdict. Work through the cases in bounded batches if needed, but continue until all 30 are complete.

Validate and lock the blinded results before unblinding:

```bash
python scripts/summarize_trace_audit.py \
  --audit-dir outputs/original_v6_rebuttal/qwen35_27b/audit
```

This command checks the result schema and hashes first, then joins the hidden outcome key and writes the audit summary and bounded qualitative cases. The report must describe this as an outcome-stratified, non-method-blinded model audit, not human evaluation or a population estimate.

## 6. Package and verify

After the audit summary succeeds:

```bash
python scripts/package_original_v6_rebuttal.py \
  --run-root outputs/original_v6_rebuttal/qwen35_27b
python scripts/verify_delivery.py \
  --delivery-dir outputs/original_v6_rebuttal/qwen35_27b/delivery
```

Packaging must fail if it detects an endpoint, URL, IP address, absolute path, credential, unexpected benchmark source file, or unapproved archive member. Do not weaken the scanner to make packaging pass; locate and sanitize the offending generated artifact instead.

The only files the coauthor sends privately to the paper author are:

- `skillrm_original_v6_rebuttal_summary_qwen35_27b.tar.gz`
- `skillrm_original_v6_rebuttal_analysis_qwen35_27b.tar.gz`

The summary archive contains aggregate material only. The analysis archive contains the complete sanitized nine-run outputs, logs, audit inputs/results/key, and analysis material while preserving JSONL row order and counts. It excludes source benchmark files, `.venv`, caches, Git history, server logs, and any unsanitized staging tree.

## 7. Final report to the coauthor

Do not commit or push experiment outputs. Report:

- status of all 9 jobs;
- whether all validation and audit checks passed;
- absolute path, byte size, and SHA256 for each of the two archives;
- a clear instruction that those two archives, and only those two, should be sent privately to the paper author.

If blocked, report the exact failed step and redacted error. A blocker is legitimate only after retrying transient requests, removing failed endpoint indices, validating the pinned data, and attempting resume as specified above.
