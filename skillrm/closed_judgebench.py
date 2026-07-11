from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import openrs_benchmark as openrs
from .closed_chat_client import call_closed_with_retries
from .qwen_baseline import load_config


def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), args)
    original_call_with_retries = openrs.call_with_retries
    try:
        openrs.call_with_retries = call_closed_with_retries
        if args.recompute_metrics_only:
            openrs.recompute_metrics(config)
        else:
            openrs.run_openrs_benchmark(config)
    finally:
        openrs.call_with_retries = original_call_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JudgeBench with a closed chat API model.")
    parser.add_argument("--config", default="configs/closed_models/judgebench/gemini3_flash_skill_operational.yaml")
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--recompute-metrics-only", action="store_true")
    return parser.parse_args()


def merge_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in ("output_dir", "model", "limit", "workers", "timeout", "max_tokens", "max_agent_steps"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    if args.resume:
        merged["resume"] = True
    if args.no_resume:
        merged["resume"] = False
    merged.setdefault("benchmark", "judgebench")
    merged.setdefault("evaluation_mode", "self_select_skill_pairwise")
    merged.setdefault("skill_path", "skills/reward_judge_operational")
    merged.setdefault("skill_loading_mode", "progressive")
    merged.setdefault("skill_allowed_setting", "skill_operational")
    merged.setdefault("baseline_fallback_query_types", False)
    merged.setdefault("model", "gemini-3-flash-preview")
    merged.setdefault("base_urls", ["closed://chat"])
    merged.setdefault("seed", 0)
    merged.setdefault("workers", 8)
    merged.setdefault("resume", True)
    merged.setdefault("temperature", 0.0)
    merged.setdefault("top_p", 1.0)
    merged.setdefault("max_tokens", 4096)
    merged.setdefault("timeout", 120)
    merged.setdefault("retries", 2)
    merged.setdefault("progress_every", 10)
    merged.setdefault("record_trace", True)
    merged.setdefault("tool_choice", "auto")
    merged.setdefault("max_agent_steps", 6)
    merged.setdefault("output_dir", "runs/closed_models_20260518/gemini-3-flash-preview/judgebench/skill_operational")
    output_dir = Path(str(merged["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    return merged


if __name__ == "__main__":
    main()
