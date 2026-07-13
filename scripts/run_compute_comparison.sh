#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: scripts/run_compute_comparison.sh [run_root]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${1:-outputs/compute_comparison/qwen35_27b}"
workers="${SKILLRM_WORKERS:-80}"
dry_run="${DRY_RUN:-0}"

: "${SKILLRM_DATA_ROOT:?Set SKILLRM_DATA_ROOT to the pinned OpenRS data directory.}"
: "${SKILLRM_BASE_URLS:?Set SKILLRM_BASE_URLS to comma-separated OpenAI-compatible /v1 URLs.}"

export SKILLRM_MODEL="${SKILLRM_MODEL:-Qwen3.5-27B}"
export NO_PROXY="*"
export no_proxy="*"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ "$dry_run" != "1" && -d "$run_root" ]] && find "$run_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing non-empty compute run root: $run_root" >&2
  echo "Use a fresh directory so usage-instrumented results cannot mix with older outputs." >&2
  exit 2
fi

python scripts/verify_original_sources.py --repo-root "$repo_root"

if [[ "$dry_run" != "1" ]]; then
  python scripts/check_endpoints.py --chat --tools --require-usage --redact-urls
  python scripts/prepare_rebuttal_data.py --output "$SKILLRM_DATA_ROOT" --validate-only
fi

mkdir -p "$run_root/logs"

echo "run_root=$run_root"
echo "model=$SKILLRM_MODEL"
echo "workers=$workers"
echo "dry_run=$dry_run"

run_one() {
  local method="$1"
  local bench="$2"
  local source_method module config output log

  if [[ "$method" == "direct" ]]; then
    source_method="baseline"
  else
    source_method="skill_fair"
  fi
  config="configs/${bench}/${source_method}.yaml"
  output="${run_root}/${method}/${bench}"
  log="${run_root}/logs/${bench}_${method}.log"
  if [[ "$bench" == "rewardbench2" ]]; then
    module="skillrm.qwen_baseline"
  else
    module="skillrm.openrs_benchmark"
  fi

  if [[ "$dry_run" == "1" ]]; then
    printf 'DRY_RUN method=%s benchmark=%s module=%s workers=%s fresh=true\n' \
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
    --workers "$workers" 2>&1 | python scripts/redact_stream.py | tee -a "$log"
  exit_code="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"
  status="failed"
  [[ "$exit_code" == "0" ]] && status="completed"
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
  [[ "$exit_code" == "0" ]] || return "$exit_code"
}

for bench in rewardbench2 rmbench judgebench; do
  run_one direct "$bench"
  run_one skill_rm "$bench"
done

if [[ "$dry_run" == "1" ]]; then
  echo "DRY_RUN completed; exactly six jobs were described and no model requests were sent."
  exit 0
fi

python scripts/summarize_compute_comparison.py --run-root "$run_root"
python scripts/package_compute_comparison.py --run-root "$run_root"
python scripts/verify_compute_delivery.py --delivery-dir "$run_root/delivery"
echo "compute_comparison_complete=true"
