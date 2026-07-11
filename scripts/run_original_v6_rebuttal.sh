#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: scripts/run_original_v6_rebuttal.sh [run_root]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${1:-outputs/original_v6_rebuttal/qwen35_27b}"
workers="${SKILLRM_WORKERS:-80}"
dry_run="${DRY_RUN:-0}"

: "${SKILLRM_DATA_ROOT:?Set SKILLRM_DATA_ROOT to the pinned OpenRS data directory.}"
: "${SKILLRM_BASE_URLS:?Set SKILLRM_BASE_URLS to comma-separated OpenAI-compatible /v1 URLs.}"

export SKILLRM_MODEL="${SKILLRM_MODEL:-Qwen3.5-27B}"
export SKILLRM_MODEL_KEY="qwen35_27b"
export NO_PROXY="*"
export no_proxy="*"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

RUN_ROOT="$run_root" REPO_ROOT="$repo_root" python - <<'PY'
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"]).resolve()
target = Path(os.environ["RUN_ROOT"])
if not target.is_absolute():
    target = (repo / target).resolve()
else:
    target = target.resolve()

protected = [
    repo / "runs" / "paper_main",
    repo / "runs" / "ablations",
    repo / "runs" / "release_20260517",
    repo / "runs" / "ablations_20260517",
]
for root in protected:
    root = root.resolve()
    if target == root or root in target.parents:
        raise SystemExit(f"Refusing to write rebuttal outputs under protected evidence root: {target}")
print(f"validated_run_root={target}")
PY

python scripts/verify_original_sources.py --repo-root "$repo_root"
python scripts/prepare_original_v6_resource_subsets.py \
  --repo-root "$repo_root" \
  --run-root "$run_root"

if [[ "$dry_run" != "1" ]]; then
  python scripts/prepare_rebuttal_data.py --output "$SKILLRM_DATA_ROOT" --validate-only
fi

mkdir -p "$run_root/logs"

echo "run_root=$run_root"
echo "model=$SKILLRM_MODEL"
echo "workers=$workers"
echo "dry_run=$dry_run"
echo "base_url_count=$(python - <<'PY'
import os
print(len([item for item in os.environ["SKILLRM_BASE_URLS"].split(",") if item.strip()]))
PY
)"

run_one() {
  local method="$1"
  local bench="$2"
  local config="configs/rebuttal_original_v6/${bench}_${method}.yaml"
  local output="${run_root}/${method}/${bench}"
  local log="${run_root}/logs/${bench}_${method}.log"
  local module

  if [[ "$bench" == "rewardbench2" ]]; then
    module="skillrm.qwen_baseline"
  else
    module="skillrm.openrs_benchmark"
  fi

  if [[ "$dry_run" == "1" ]]; then
    printf 'DRY_RUN method=%s benchmark=%s module=%s workers=%s resume=true\n' \
      "$method" "$bench" "$module" "$workers"
    return
  fi

  local started_at finished_at start_epoch end_epoch duration status exit_code
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"
  echo "[$started_at] start ${method}/${bench}" | tee -a "$log"
  set +e
  python -m "$module" \
    --config "$config" \
    --output "$output" \
    --workers "$workers" \
    --resume 2>&1 | python scripts/redact_stream.py | tee -a "$log"
  exit_code="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"
  if [[ "$exit_code" == "0" ]]; then
    status="completed"
  else
    status="failed"
  fi
  python scripts/record_run_attempt.py \
    --path "$run_root/run_attempts.jsonl" \
    --method "$method" \
    --benchmark "$bench" \
    --module "$module" \
    --config "$config" \
    --output "$output" \
    --started-at "$started_at" \
    --finished-at "$finished_at" \
    --duration-sec "$duration" \
    --status "$status" \
    --exit-code "$exit_code"
  if [[ -d "$output" ]]; then
    python scripts/sanitize_run_artifacts.py "$output" | tee -a "$log"
  fi
  echo "[$finished_at] ${status} ${method}/${bench}" | tee -a "$log"
  if [[ "$exit_code" != "0" ]]; then
    return "$exit_code"
  fi
}

for bench in rewardbench2 judgebench rmbench; do
  run_one full_fair "$bench"
  run_one subset_a "$bench"
  run_one subset_b "$bench"
done

if [[ "$dry_run" == "1" ]]; then
  echo "DRY_RUN completed; no model requests were sent."
  exit 0
fi

python scripts/summarize_original_v6_rebuttal.py --run-root "$run_root"
python scripts/prepare_trace_audit.py --run-root "$run_root"
echo "inference_and_analysis_complete=true"
echo "next_step=complete the blinded model audit described in ${run_root}/audit/audit_instructions.md"
