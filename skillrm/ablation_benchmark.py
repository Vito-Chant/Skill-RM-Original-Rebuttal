from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

from .openrs_benchmark import (
    OpenRSPairTask,
    build_pairwise_row,
    compute_openrs_metrics,
    format_judgebench_prompt,
    load_completed_rows,
    load_openrs_tasks,
    pairwise_final_answer_tool,
    pairwise_final_valid_for_benchmark,
    parse_pairwise_final,
    parse_pairwise_final_answer_tool_call,
    write_openrs_summary,
)
from .qwen_baseline import (
    DEFAULT_ENDPOINTS,
    OFFICIAL_RANKING_SYSTEM_PROMPT,
    OFFICIAL_RANKING_USER_TEMPLATE,
    call_with_retries,
    first_final_answer_tool_call,
    is_ties_record,
    judge_official_ratings,
    load_completed,
    load_config,
    load_official_records,
    load_skill_package,
    normalize_base_urls,
    official_format_ranking_record,
    official_metrics_from_rows,
    official_ranking_score,
    operational_sample_resources,
    parse_final_answer_tool_call,
    parse_official_winner,
    print_progress,
    response_output_fields,
    run_python_sandbox_tool,
    write_json,
    write_summary,
)


FINAL_LISTWISE_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the final listwise judgment. The verdict must be exactly one of A, B, C, or D.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "rationale": {"type": "string"},
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}


PYTHON_SANDBOX_TOOL = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "Run small Python checks over only the visible prompt and candidate responses. "
            "Use for deterministic visible-text evidence such as word counts, exact format, regex, "
            "JSON/Markdown structure, simple arithmetic, or required terms. No external data is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code. Available variables: prompt (str), candidates (dict label->text), "
                        "sample (dict with prompt and candidates). Print compact JSON/text evidence."
                    ),
                },
                "reason": {"type": "string"},
            },
            "required": ["code", "reason"],
            "additionalProperties": False,
        },
    },
}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output:
        config["output_dir"] = args.output
    if args.base_urls:
        config["base_urls"] = args.base_urls
    if args.model:
        config["model"] = args.model
    if args.workers is not None:
        config["workers"] = args.workers
    if args.limit is not None:
        config["limit"] = args.limit
    if args.resume:
        config["resume"] = True
    config.setdefault("resume", True)
    config.setdefault("temperature", 0.0)
    config.setdefault("top_p", 1.0)
    config.setdefault("max_tokens", 4096)
    config.setdefault("timeout", 300)
    config.setdefault("retries", 2)
    config.setdefault("progress_every", 25)
    benchmark = str(config.get("benchmark") or "").lower()
    if benchmark in {"rewardbench2", "rewardbench_v2", "rb2"}:
        run_rewardbench2_ablation(config)
    elif benchmark.startswith("judgebench") or benchmark in {"rmbench", "rm-bench", "rm_bench"}:
        run_pairwise_ablation(config)
    else:
        raise ValueError(f"Unsupported ablation benchmark: {benchmark}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Skill-RM ablation experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--base-urls")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_rewardbench2_ablation(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_official_records(
        config["data_source"],
        limit=config.get("limit"),
        include_ties=bool(config.get("include_ties", True)),
    )
    base_urls = normalize_base_urls(config.get("base_urls") or DEFAULT_ENDPOINTS)
    workers = int(config.get("workers") or max(1, len(base_urls)))
    skill_package = load_skill_package(config) if config.get("ablation_mode") == "flat_prompt" else None
    resolved = config | {"base_urls": base_urls}
    if skill_package:
        resolved["skill_package_sha256"] = skill_package["sha256"]
    write_json(output_dir / "config_resolved.json", resolved)
    write_json(output_dir / "dataset_summary.json", {"n": len(records)})

    completed = load_completed(output_dir / "predictions.jsonl") if config.get("resume") else {}
    pending = [record for record in records if str(record["id"]) not in completed]
    rows: dict[str, dict[str, Any]] = dict(completed)
    started_at = time.time()
    trace_handle = (output_dir / "traces.jsonl").open("a", encoding="utf-8") if bool(config.get("record_trace", True)) else None
    try:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        judge_rewardbench2_record,
                        record,
                        base_urls[index % len(base_urls)],
                        config,
                        skill_package,
                    ): record
                    for index, record in enumerate(pending)
                }
                for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    row = future.result()
                    trace = row.pop("_trace", None)
                    rows[row["sample_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if trace_handle is not None and trace is not None:
                        trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        trace_handle.flush()
                    if done_count % int(config.get("progress_every", 25)) == 0:
                        print_progress(done_count, len(pending), started_at)
    finally:
        if trace_handle is not None:
            trace_handle.close()

    ordered_rows = [rows[str(record["id"])] for record in records if str(record["id"]) in rows]
    metrics = official_metrics_from_rows(records, ordered_rows)
    metrics["ablation_mode"] = config.get("ablation_mode")
    write_json(output_dir / "metrics.json", metrics)
    write_summary(output_dir / "summary.md", config, [], ordered_rows, metrics, time.time() - started_at)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def judge_rewardbench2_record(
    record: dict[str, Any],
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any] | None,
) -> dict[str, Any]:
    if is_ties_record(record):
        row = judge_official_ratings(record, base_url, config, is_ties=True)
        row["mode"] = f"ablation_{config.get('ablation_mode')}_ratings"
        return row
    if config.get("ablation_mode") == "flat_prompt":
        if skill_package is None:
            raise ValueError("flat_prompt requires skill_path.")
        return judge_rewardbench2_flat_prompt(record, base_url, config, skill_package)
    if config.get("ablation_mode") == "tool_only":
        return judge_rewardbench2_tool_only(record, base_url, config)
    raise ValueError(f"unknown ablation_mode: {config.get('ablation_mode')}")


def judge_rewardbench2_flat_prompt(
    record: dict[str, Any],
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any],
) -> dict[str, Any]:
    formatted = official_format_ranking_record(record, seed=int(config.get("seed", 0)))
    prompt = flat_listwise_prompt(record, formatted, skill_package, config, benchmark="rewardbench2")
    started_at = time.time()
    response = call_with_retries(
        base_url,
        [{"role": "user", "content": prompt}],
        config,
        tools=[FINAL_LISTWISE_TOOL],
        tool_choice={"type": "function", "function": {"name": "final_answer"}},
    )
    raw_output = response.get("content", "")
    final_tool = first_final_answer_tool_call(response.get("tool_calls") or [])
    parsed = parse_final_answer_tool_call(final_tool) if final_tool is not None else parse_listwise_final(raw_output)
    winner = parsed["winner"]
    score = official_ranking_score(winner, formatted["chosen_label"])
    valid = winner in {"A", "B", "C", "D"}
    return {
        "sample_id": str(record["id"]),
        "subset_for_metrics_only": record.get("subset"),
        "mode": "ablation_flat_prompt_official_ranking",
        "chosen_label": formatted["chosen_label"],
        "predicted_label": winner,
        "official_score": score,
        "correct": score == 1.0,
        "valid": valid,
        "shuffle_position": formatted["shuffle_position"],
        "endpoint": base_url,
        "skill_path": skill_package["source"],
        "skill_package_sha256": skill_package["sha256"],
        "flat_resources_injected": flat_resource_paths(skill_package, "rewardbench2"),
        "verdict_source": parsed["source"],
        "parse_error": None if valid else "official verdict not found",
        "latency_sec": time.time() - started_at,
        **response_output_fields(response, config),
    }


def judge_rewardbench2_tool_only(record: dict[str, Any], base_url: str, config: dict[str, Any]) -> dict[str, Any]:
    formatted = official_format_ranking_record(record, seed=int(config.get("seed", 0)))
    messages = [
        {"role": "system", "content": tool_only_listwise_system_prompt()},
        {"role": "user", "content": formatted["user_prompt"]},
    ]
    response, parsed, trace = run_tool_only_loop(
        messages,
        base_url,
        config,
        record={"prompt": record.get("prompt", "")},
        formatted={"responses": formatted["responses"]},
        final_tool=FINAL_LISTWISE_TOOL,
        parse_content=lambda content, finish_reason=None: parse_listwise_final(content),
        valid_labels={"A", "B", "C", "D"},
    )
    winner = parsed["winner"]
    score = official_ranking_score(winner, formatted["chosen_label"])
    valid = winner in {"A", "B", "C", "D"}
    row = {
        "sample_id": str(record["id"]),
        "subset_for_metrics_only": record.get("subset"),
        "mode": "ablation_tool_only_official_ranking",
        "chosen_label": formatted["chosen_label"],
        "predicted_label": winner,
        "official_score": score,
        "correct": score == 1.0,
        "valid": valid,
        "shuffle_position": formatted["shuffle_position"],
        "endpoint": base_url,
        "tool_call_count": trace["tool_call_count"],
        "python_sandbox_call_count": trace["python_sandbox_call_count"],
        "tool_error_count": trace["tool_error_count"],
        "agent_step_count": len(trace["steps"]),
        "parse_error": None if valid else "official verdict not found",
        **response_output_fields(response, config),
        "_trace": trace,
    }
    return row


def run_pairwise_ablation(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_openrs_tasks(config)
    base_urls = normalize_base_urls(config.get("base_urls") or DEFAULT_ENDPOINTS)
    workers = int(config.get("workers") or max(1, len(base_urls)))
    skill_package = load_skill_package(config) if config.get("ablation_mode") == "flat_prompt" else None
    resolved = config | {"base_urls": base_urls}
    if skill_package:
        resolved["skill_package_sha256"] = skill_package["sha256"]
    write_json(output_dir / "config_resolved.json", resolved)
    write_json(output_dir / "dataset_summary.json", {"n": len(tasks)})

    completed = load_completed_rows(output_dir / "predictions.jsonl") if config.get("resume") else {}
    pending = [task for task in tasks if task.task_id not in completed]
    rows: dict[str, dict[str, Any]] = dict(completed)
    started_at = time.time()
    trace_handle = (output_dir / "traces.jsonl").open("a", encoding="utf-8") if bool(config.get("record_trace", True)) else None
    try:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        judge_pairwise_task,
                        task,
                        base_urls[index % len(base_urls)],
                        config,
                        skill_package,
                    ): task
                    for index, task in enumerate(pending)
                }
                for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    row = future.result()
                    trace = row.pop("_trace", None)
                    rows[row["sample_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if trace_handle is not None and trace is not None:
                        trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        trace_handle.flush()
                    if done_count % int(config.get("progress_every", 25)) == 0:
                        print_progress(done_count, len(pending), started_at)
    finally:
        if trace_handle is not None:
            trace_handle.close()

    ordered_rows = [rows[task.task_id] for task in tasks if task.task_id in rows]
    metrics = compute_openrs_metrics(tasks, ordered_rows, config)
    metrics["ablation_mode"] = config.get("ablation_mode")
    write_json(output_dir / "metrics.json", metrics)
    write_openrs_summary(output_dir / "summary.md", config, metrics, len(tasks), len(ordered_rows), time.time() - started_at)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def judge_pairwise_task(
    task: OpenRSPairTask,
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.get("ablation_mode") == "flat_prompt":
        if skill_package is None:
            raise ValueError("flat_prompt requires skill_path.")
        return judge_pairwise_flat_prompt(task, base_url, config, skill_package)
    if config.get("ablation_mode") == "tool_only":
        return judge_pairwise_tool_only(task, base_url, config)
    raise ValueError(f"unknown ablation_mode: {config.get('ablation_mode')}")


def judge_pairwise_flat_prompt(
    task: OpenRSPairTask,
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any],
) -> dict[str, Any]:
    prompt = flat_pairwise_prompt(task, skill_package, config)
    started_at = time.time()
    response = call_with_retries(
        base_url,
        [{"role": "user", "content": prompt}],
        config,
        tools=[pairwise_final_answer_tool(task.benchmark)],
        tool_choice={"type": "function", "function": {"name": "final_answer"}},
    )
    final_tool = first_final_answer_tool_call(response.get("tool_calls") or [])
    parsed = (
        parse_pairwise_final_answer_tool_call(final_tool)
        if final_tool is not None
        else parse_pairwise_final(response.get("content", ""), finish_reason=response.get("finish_reason"))
    )
    valid = pairwise_final_valid_for_benchmark(task.benchmark, parsed)
    return build_pairwise_row(
        task,
        base_url,
        config,
        mode="ablation_flat_prompt_pairwise",
        winner=parsed["winner"],
        verdict=parsed["verdict"],
        verdict_source=parsed["source"],
        valid=valid,
        parse_error=None if valid else "could not parse winner",
        latency_sec=time.time() - started_at,
        raw_output=response.get("content", ""),
        response=response,
    ) | {
        "skill_path": skill_package["source"],
        "skill_package_sha256": skill_package["sha256"],
        "flat_resources_injected": flat_resource_paths(skill_package, str(config.get("benchmark") or "")),
    }


def judge_pairwise_tool_only(task: OpenRSPairTask, base_url: str, config: dict[str, Any]) -> dict[str, Any]:
    if task.benchmark.startswith("judgebench"):
        user_prompt = format_judgebench_prompt(task)
        system_prompt = tool_only_judgebench_system_prompt()
    else:
        user_prompt = f"[User Prompt]\n{task.prompt}\n\n[Response A]\n{task.responses['A']}\n\n[Response B]\n{task.responses['B']}"
        system_prompt = tool_only_pairwise_system_prompt()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response, parsed, trace = run_tool_only_loop(
        messages,
        base_url,
        config,
        record={"prompt": task.prompt},
        formatted={"responses": task.responses},
        final_tool=pairwise_final_answer_tool(task.benchmark),
        parse_content=lambda content, finish_reason=None: parse_pairwise_final(content, finish_reason=finish_reason),
        valid_labels={"A", "B", "Tie"},
    )
    valid = pairwise_final_valid_for_benchmark(task.benchmark, parsed)
    row = build_pairwise_row(
        task,
        base_url,
        config,
        mode="ablation_tool_only_pairwise",
        winner=parsed["winner"],
        verdict=parsed["verdict"],
        verdict_source=parsed["source"],
        valid=valid,
        parse_error=None if valid else "could not parse winner",
        latency_sec=sum(float(step.get("latency_sec") or 0) for step in trace["steps"]),
        raw_output=response.get("content", ""),
        response=response,
    )
    row.update(
        {
            "tool_call_count": trace["tool_call_count"],
            "python_sandbox_call_count": trace["python_sandbox_call_count"],
            "tool_error_count": trace["tool_error_count"],
            "agent_step_count": len(trace["steps"]),
            "_trace": trace,
        }
    )
    return row


def run_tool_only_loop(
    messages: list[dict[str, Any]],
    base_url: str,
    config: dict[str, Any],
    *,
    record: dict[str, Any],
    formatted: dict[str, Any],
    final_tool: dict[str, Any],
    parse_content,
    valid_labels: set[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    tools = [PYTHON_SANDBOX_TOOL, final_tool]
    max_steps = int(config.get("max_agent_steps", 4))
    tool_choice = config.get("tool_choice", "auto")
    trace = {"mode": "ablation_tool_only", "steps": [], "tool_call_count": 0, "python_sandbox_call_count": 0, "tool_error_count": 0}
    final_response: dict[str, Any] = {"content": "", "latency_sec": 0.0}
    parsed = {"verdict": "error", "winner": "error", "source": "missing"}
    for step in range(1, max_steps + 1):
        response = call_with_retries(base_url, messages, config, tools=tools, tool_choice=tool_choice)
        final_response = response
        tool_calls = response.get("tool_calls") or []
        step_trace = {
            "step": step,
            "assistant_content": response.get("content", ""),
            "finish_reason": response.get("finish_reason"),
            "latency_sec": response.get("latency_sec"),
            "tool_calls": compact_tool_calls(tool_calls),
            "tool_results": [],
        }
        final_tool_call = first_final_answer_tool_call(tool_calls)
        if final_tool_call is not None:
            parsed = parse_final_answer_tool_call(final_tool_call) if final_tool["function"]["name"] == "final_answer" else {"verdict": "error", "winner": "error", "source": "tool.unparsed"}
            step_trace["final"] = parsed
            trace["steps"].append(step_trace)
            break
        if tool_calls:
            trace["tool_call_count"] += len(tool_calls)
            messages.append({"role": "assistant", "content": response.get("content", "") or "", "tool_calls": tool_calls})
            for tool_call in tool_calls:
                name = tool_name(tool_call)
                args = tool_args(tool_call)
                if name == "python_sandbox":
                    trace["python_sandbox_call_count"] += 1
                    tool_result = run_python_sandbox_tool(args, record, formatted, config)
                else:
                    tool_result = {"ok": False, "tool": name, "error": "only python_sandbox and final_answer are available in tool_only"}
                if not tool_result.get("ok"):
                    trace["tool_error_count"] += 1
                step_trace["tool_results"].append({"tool": name, "ok": tool_result.get("ok"), "error": tool_result.get("error")})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id") or f"call_{step}_{len(step_trace['tool_results'])}"),
                        "name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            trace["steps"].append(step_trace)
            continue
        parsed = parse_content(response.get("content", ""), response.get("finish_reason"))
        step_trace["final"] = parsed
        trace["steps"].append(step_trace)
        if parsed.get("winner") in valid_labels:
            break
        break
    if parsed.get("winner") not in valid_labels and bool(config.get("enable_forced_finalization", True)):
        retry_messages = list(messages)
        retry_messages.append({"role": "user", "content": "No more tool calls. Reply with exactly the required final label and nothing else."})
        forced_config = dict(config)
        forced_config["max_tokens"] = int(config.get("forced_finalization_max_tokens", 128))
        response = call_with_retries(base_url, retry_messages, forced_config, tools=None, tool_choice=None)
        final_response = response
        parsed = parse_content(response.get("content", ""), response.get("finish_reason"))
        trace["steps"].append(
            {
                "step": len(trace["steps"]) + 1,
                "forced_finalization": True,
                "assistant_content": response.get("content", ""),
                "finish_reason": response.get("finish_reason"),
                "latency_sec": response.get("latency_sec"),
                "final": parsed,
            }
        )
    return final_response, parsed, trace


def flat_listwise_prompt(record: dict[str, Any], formatted: dict[str, Any], skill_package: dict[str, Any], config: dict[str, Any], *, benchmark: str) -> str:
    return "\n\n".join(
        [
            "You are an impartial reward judge. This is a flat-prompt ablation: there is no skill tool and no progressive disclosure.",
            "Use the visible task, candidates, and the flattened resource text below. Do not assume any hidden labels.",
            flat_resource_block(skill_package, benchmark, record, {"responses": formatted["responses"]}, config),
            "## Judge Task",
            OFFICIAL_RANKING_SYSTEM_PROMPT,
            OFFICIAL_RANKING_USER_TEMPLATE.format(
                question=str(record.get("prompt", "")),
                answer_a=formatted["responses"]["A"],
                answer_b=formatted["responses"]["B"],
                answer_c=formatted["responses"]["C"],
                answer_d=formatted["responses"]["D"],
            ),
            'End with exactly one final verdict token: "[[A]]", "[[B]]", "[[C]]", or "[[D]]".',
        ]
    )


def flat_pairwise_prompt(task: OpenRSPairTask, skill_package: dict[str, Any], config: dict[str, Any]) -> str:
    record = {
        "id": task.task_id,
        "prompt": task.prompt,
        "benchmark": task.benchmark,
        "query_type": task.query_type,
        "domain": task.domain,
        "pair": task.pair,
        "order": task.order,
        **dict(task.sample_resources or {}),
    }
    formatted = {"responses": task.responses}
    if task.benchmark.startswith("judgebench"):
        judge_task = format_judgebench_prompt(task)
        final = 'End with exactly one label and no explanation: Output (a) or Output (b).'
    else:
        judge_task = f"[User Prompt]\n{task.prompt}\n\n[Response A]\n{task.responses['A']}\n\n[Response B]\n{task.responses['B']}"
        final = "End with exactly one label: Final: A, Final: B, or Final: Tie."
    return "\n\n".join(
        [
            "You are an impartial reward judge. This is a flat-prompt ablation: there is no skill tool and no progressive disclosure.",
            "Use the visible task, candidates, and the flattened resource text below. Do not assume any hidden labels.",
            flat_resource_block(skill_package, str(config.get("benchmark") or ""), record, formatted, config),
            "## Judge Task",
            judge_task,
            final,
        ]
    )


def flat_resource_block(
    skill_package: dict[str, Any],
    benchmark: str,
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> str:
    paths = flat_resource_paths(skill_package, benchmark)
    max_chars = int(config.get("flat_prompt_max_resource_chars", 18000))
    chunks = []
    budget = max_chars
    for path in paths:
        text = str(skill_package.get("files", {}).get(path, "")).strip()
        if not text or budget <= 0:
            continue
        clipped = text[:budget]
        budget -= len(clipped)
        chunks.append(f"### {path}\n{clipped}")
    if str(config.get("skill_allowed_setting") or "") == "skill_operational":
        runtime_index, runtime_files = operational_sample_resources(record, formatted, config)
        if runtime_files:
            chunks.append("## Sample-Visible Operational Resources")
            for path, text in runtime_files.items():
                chunks.append(f"### {path}\n{text[: int(config.get('flat_prompt_max_sample_resource_chars', 6000))]}")
        if runtime_index:
            chunks.append("### sample/resource_index.json\n" + json.dumps(runtime_index, ensure_ascii=False, indent=2))
    return "## Flattened Resources\n" + "\n\n".join(chunks)


def flat_resource_paths(skill_package: dict[str, Any], benchmark: str) -> list[str]:
    normalized = benchmark.lower().replace("-", "").replace("_", "")
    bench_paths: list[str] = []
    if "rewardbench2" in normalized or "rewardbenchv2" in normalized or normalized == "rb2":
        bench_paths = ["references/rewardbench2_principles.md", "rubrics/rewardbench2.md"]
    elif "judgebench" in normalized:
        bench_paths = ["references/judgebench_principles.md", "rubrics/judgebench.md"]
    elif "rmbench" in normalized:
        bench_paths = ["references/rmbench_principles.md", "rubrics/rmbench.md"]
    preferred = [
        "SKILL.md",
        "resources.yaml",
        "references/generic_principles.md",
        "references/bias_control.md",
        "references/output_format.md",
        "references/operational_aggregation.md",
        "rubrics/generic_pairwise.md",
        *bench_paths,
    ]
    files = skill_package.get("files", {})
    return [path for path in preferred if path in files]


def parse_listwise_final(content: str) -> dict[str, str]:
    label = parse_listwise_label(content)
    return {
        "winner": label,
        "verdict": label,
        "source": "listwise_label" if label in {"A", "B", "C", "D"} else "unparsed",
    }


def parse_listwise_label(content: str) -> str:
    official = parse_official_winner(content)
    if official in {"A", "B", "C", "D"}:
        return official
    tail = content.strip()[-300:]
    patterns = [
        r"(?im)^\s*Final\s*:\s*([ABCD])\s*\.?\s*$",
        r"(?im)^\s*(?:Answer|Verdict|Choice|Best)\s*:\s*([ABCD])\s*\.?\s*$",
        r"(?im)^\s*\[?([ABCD])\]?\s*\.?\s*$",
        r"(?i)\b(?:assistant|response|candidate)\s*([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return str(matches[-1]).upper()
    json_match = re.search(r"\{.*\}", tail, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            for key in ("verdict", "winner", "best_label", "choice", "selected"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip().upper() in {"A", "B", "C", "D"}:
                    return value.strip().upper()
    return "error"


def tool_only_listwise_system_prompt() -> str:
    return (
        "You are an impartial reward judge. You may use python_sandbox only for deterministic checks over the visible "
        "prompt and candidates. No skill, metadata, reference, ground truth, checklist, or external resource is available. "
        "When ready, call final_answer with A, B, C, or D."
    )


def tool_only_pairwise_system_prompt() -> str:
    return (
        "You are an impartial pairwise reward judge. You may use python_sandbox only for deterministic checks over the visible "
        "prompt and candidates. No skill, metadata, reference, ground truth, checklist, or external resource is available. "
        "When ready, give the required pairwise final label."
    )


def tool_only_judgebench_system_prompt() -> str:
    return (
        "You are an impartial forced-choice response judge. You may use python_sandbox only for deterministic checks over the "
        "visible instruction and Output (a)/Output (b). No skill, metadata, reference, ground truth, checklist, or external "
        "resource is available. A maps to Output (a), and B maps to Output (b). When ready, call final_answer with A or B."
    )


def compact_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": item.get("id"), "name": tool_name(item), "arguments": tool_args(item)} for item in tool_calls]


def tool_name(tool_call: dict[str, Any]) -> str:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return str(fn.get("name") or tool_call.get("name") or "")


def tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    raw = fn.get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
