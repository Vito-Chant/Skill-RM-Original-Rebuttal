#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BENCHMARKS = ("rewardbench2", "rmbench", "judgebench")
SETTINGS = ("standard", "resource_enhanced")
METHOD_DIR = {"standard": "skill_fair", "resource_enhanced": "skill_operational"}
QUOTAS = {"standard": 7, "resource_enhanced": 3}
DIMENSIONS = ("action_relevance", "evidence_validity", "evidence_faithfulness", "verdict_support", "auditability")
SEED = 0


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(items: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)
    result = {}
    for line_index, item in enumerate(items):
        sample_id = str(item.get("sample_id") or "")
        occurrence = counts[sample_id]
        counts[sample_id] += 1
        result[(sample_id, occurrence)] = {**item, "_line_index": line_index, "_occurrence": occurrence}
    return result


def tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or ((call.get("function") or {}).get("name")) or "")


def tool_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for step in trace.get("steps") or []:
        calls = step.get("tool_calls") or []
        results = step.get("tool_results") or []
        for index, call in enumerate(calls):
            result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            name = tool_name(call)
            if name in {"view_resource", "python_sandbox", "run_resource", "wiki_search"}:
                events.append({"name": name, "call": call, "result": result, "step": step.get("step")})
    return events


def evidence_types(trace: dict[str, Any]) -> set[str]:
    found = set()
    for event in tool_events(trace):
        if event["result"].get("ok") is not True:
            continue
        found.add("resource" if event["name"] == "view_resource" else "python" if event["name"] == "python_sandbox" else "runtime")
    return found


def outcome(baseline: dict[str, Any], method: dict[str, Any]) -> str:
    before, after = baseline.get("correct") is True, method.get("correct") is True
    if after and not before:
        return "corrected"
    if before and after:
        return "both_correct"
    if before and not after:
        return "regressed"
    return "both_wrong"


def family(path: str) -> str:
    value = path.lower()
    if "reference" in value or "ground_truth" in value or "expected" in value:
        return "sample_reference" if "sample" in value or "ground_truth" in value or "expected" in value else "general_principles"
    if "metadata" in value:
        return "sample_metadata"
    if "checklist" in value or "constraint" in value:
        return "sample_checklist"
    if "verifier" in value or "checker" in value:
        return "verifier"
    if "rubric" in value:
        return "pairwise_rubric"
    if "bias" in value:
        return "bias_control"
    if "aggregation" in value:
        return "aggregation_guidance"
    return "other"


def failure_counts(prediction: dict[str, Any], trace: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    steps = trace.get("steps") or []
    for step_index, step in enumerate(steps):
        if step.get("request_error"):
            counts["request_error"] += 1
            if step_index + 1 < len(steps):
                counts["request_retry"] += 1
        if step.get("finish_reason") == "length":
            counts["truncated_step"] += 1
        if step.get("forced_finalization"):
            counts["forced_finalization"] += 1
        if step.get("forced_finalization_retry_text"):
            counts["forced_finalization_retry"] += 1
        if step.get("emergency_direct_judge_finalization"):
            counts["emergency_direct_finalization_step"] += 1
    events = tool_events(trace)
    for event_index, event in enumerate(events):
        result = event["result"]
        if result and result.get("ok") is not True:
            counts["tool_failure"] += 1
            if any(later["name"] == event["name"] and later["result"].get("ok") is True for later in events[event_index + 1 :]):
                counts["successful_tool_retry"] += 1
            text = json.dumps(result, ensure_ascii=False).lower()
            if any(token in text for token in ("unsupported", "cannot execute", "not installed", "language")):
                counts["unsupported_execution"] += 1
    if prediction.get("emergency_direct_finalization_used"):
        counts["emergency_finalization"] += 1
    if prediction.get("valid") is False:
        counts["invalid_final"] += 1
    return counts


def summarize(run_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"settings": {}}
    for setting in SETTINGS:
        method = METHOD_DIR[setting]
        report["settings"][setting] = {}
        for benchmark in BENCHMARKS:
            base = keyed(rows(run_root / "baseline" / benchmark / "predictions.jsonl"))
            preds = keyed(rows(run_root / method / benchmark / "predictions.jsonl"))
            traces = keyed(rows(run_root / method / benchmark / "traces.jsonl"))
            common = sorted(set(base) & set(preds) & set(traces))
            families: Counter[str] = Counter()
            failures: Counter[str] = Counter()
            groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
            provenance = Counter()
            for key in common:
                pred, trace = preds[key], traces[key]
                group = outcome(base[key], pred)
                groups[group].append(pred)
                failures.update(failure_counts(pred, trace))
                viewed = set(str(item) for item in pred.get("resources_viewed") or [])
                for path in viewed:
                    families[family(path)] += 1
                successful_views = {
                    str(event["result"].get("path") or ((event["call"].get("arguments") or {}).get("path")) or "")
                    for event in tool_events(trace)
                    if event["name"] == "view_resource" and event["result"].get("ok") is True
                }
                provenance["phantom_resource_claims"] += len({path for path in viewed if path and path not in successful_views})
                provenance["records_with_tool_failure"] += int(any(event["result"] and event["result"].get("ok") is not True for event in tool_events(trace)))

            def process(items: list[dict[str, Any]]) -> dict[str, Any]:
                if not items:
                    return {"n": 0}
                return {
                    "n": len(items),
                    "skill_invocation_rate": mean(bool(row.get("skill_triggered")) for row in items),
                    "resource_views_per_sample": mean(float(row.get("resource_view_count") or 0) for row in items),
                    "tool_calls_per_sample": mean(float(row.get("tool_call_count") or 0) for row in items),
                    "python_calls_per_sample": mean(float(row.get("python_sandbox_call_count") or 0) for row in items),
                    "agent_steps_per_sample": mean(float(row.get("agent_step_count") or 0) for row in items),
                }

            report["settings"][setting][benchmark] = {
                **process([preds[key] for key in common]),
                "outcomes": {group: process(groups[group]) for group in ("corrected", "both_correct", "regressed", "both_wrong")},
                "resource_families": dict(families.most_common()),
                "failures": dict(failures),
                "provenance": dict(provenance),
            }
    return report


def stable(case: dict[str, Any]) -> str:
    return hashlib.sha256(repr(case["key"]).encode()).hexdigest()


def candidates(run_root: Path, setting: str, benchmark: str) -> list[dict[str, Any]]:
    method = METHOD_DIR[setting]
    base = keyed(rows(run_root / "baseline" / benchmark / "predictions.jsonl"))
    preds = keyed(rows(run_root / method / benchmark / "predictions.jsonl"))
    traces = keyed(rows(run_root / method / benchmark / "traces.jsonl"))
    result = []
    for key in set(base) & set(preds) & set(traces):
        kinds = evidence_types(traces[key])
        if not kinds or preds[key].get("valid") is not True:
            continue
        result.append({"key": key, "setting": setting, "benchmark": benchmark, "outcome": outcome(base[key], preds[key]), "evidence_types": sorted(kinds), "baseline": base[key], "prediction": preds[key], "trace": traces[key]})
    return sorted(result, key=stable)


def select(pool: list[dict[str, Any]], n: int, salt: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{SEED}:{salt}")
    grouped = {name: [case for case in pool if case["outcome"] == name] for name in ("corrected", "both_correct", "regressed", "both_wrong")}
    for values in grouped.values():
        rng.shuffle(values)
    targets = ["corrected", "both_correct", "regressed"]
    selected = []
    while len(selected) < n and any(grouped.values()):
        made_progress = False
        for group in targets + ["both_wrong"]:
            if len(selected) >= n:
                break
            if grouped[group]:
                selected.append(grouped[group].pop())
                made_progress = True
        if not made_progress:
            break
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} eligible cases for {salt}; need {n}")
    return selected


def reconstructors(v6_root: Path, data_root: Path):
    sys.path.insert(0, str(v6_root))
    from skillrm.openrs_benchmark import format_pairwise_user_prompt, load_openrs_tasks, pairwise_task_runtime_record
    from skillrm.qwen_baseline import format_self_select_skill_user_prompt, load_official_records, official_format_ranking_record, operational_sample_resources

    cache: dict[str, Any] = {}

    def visible(benchmark: str, sample_id: str, occurrence: int, setting: str) -> tuple[str, dict[str, str]]:
        if benchmark == "rewardbench2":
            records = cache.setdefault("rb2", load_official_records(str(data_root / "rewardbench_v2/rewardbench_v2.jsonl"), include_ties=True))
            matches = [record for record in records if str(record.get("id")) == sample_id]
            record = matches[min(occurrence, len(matches) - 1)]
            formatted = official_format_ranking_record(record, seed=0)
            config = {"skill_allowed_setting": "skill_operational" if setting == "resource_enhanced" else "skill_fair"}
            _, runtime_files = operational_sample_resources(record, formatted, config)
            return format_self_select_skill_user_prompt(record, formatted), runtime_files
        key = f"tasks:{benchmark}"
        if key not in cache:
            sources = [str(data_root / "rmbench/rmbench.jsonl")] if benchmark == "rmbench" else [str(path) for path in sorted((data_root / "judgebench").glob("*.jsonl"))]
            config = {"benchmark": benchmark, "data_source": sources[0], "data_sources": sources, "seed": 0}
            cache[key] = {task.task_id: task for task in load_openrs_tasks(config)}
        task = cache[key][sample_id]
        config = {"skill_allowed_setting": "skill_operational" if setting == "resource_enhanced" else "skill_fair"}
        _, runtime_files = operational_sample_resources(pairwise_task_runtime_record(task), {}, config)
        return format_pairwise_user_prompt(task, with_skill_hint=True), runtime_files

    return visible


def write_csv(path: Path, audit_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", *DIMENSIONS, "error_type", "brief_reason"])
        writer.writeheader()
        for audit_id in audit_ids:
            writer.writerow({"audit_id": audit_id})


def make_audit(run_root: Path, v6_root: Path, data_root: Path, output: Path) -> None:
    visible = reconstructors(v6_root, data_root)
    selected = []
    for benchmark in BENCHMARKS:
        for setting in SETTINGS:
            selected.extend(select(candidates(run_root, setting, benchmark), QUOTAS[setting], f"{setting}:{benchmark}"))
    random.Random(SEED).shuffle(selected)
    inputs, keys = [], []
    for index, case in enumerate(selected, 1):
        audit_id = f"TRACE-{index:03d}"
        sample_id, occurrence = case["key"]
        trace = {key: value for key, value in case["trace"].items() if key not in {"sample_id", "trace_id", "original_sample_id"}}
        visible_input, runtime_files = visible(case["benchmark"], sample_id, occurrence, case["setting"])
        viewed_paths = {
            str(event["result"].get("path") or ((event["call"].get("arguments") or {}).get("path")) or "")
            for event in tool_events(case["trace"])
            if event["name"] == "view_resource" and event["result"].get("ok") is True
        }
        resource_observations = {path: runtime_files[path] for path in sorted(viewed_paths) if path in runtime_files}
        inputs.append({"audit_id": audit_id, "setting": case["setting"], "benchmark": case["benchmark"], "visible_input": visible_input, "trace": trace, "resource_observations": resource_observations, "recorded_final": {"predicted_label": case["prediction"].get("predicted_label"), "skill_final_verdict": case["prediction"].get("skill_final_verdict")}})
        keys.append({"audit_id": audit_id, "setting": case["setting"], "benchmark": case["benchmark"], "sample_id": sample_id, "occurrence": occurrence, "outcome": case["outcome"], "gold_label": case["prediction"].get("gold_label"), "correct": case["prediction"].get("correct"), "baseline_label": case["baseline"].get("predicted_label"), "evidence_types": case["evidence_types"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("audit_input.jsonl", inputs), ("sealed_outcome_key.jsonl", keys)):
        (output / name).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    write_csv(output / "annotator_a.csv", [row["audit_id"] for row in inputs])
    write_csv(output / "annotator_b.csv", [row["audit_id"] for row in inputs])
    (output / "annotation_instructions.md").write_text(
        "# Two-Author Manual Trace Evaluation\n\nAnnotate independently before opening `sealed_outcome_key.jsonl`. "
        "Do not inspect the preliminary model-assisted labels while annotating. The resource observations shown for the resource-enhanced setting are part of the evidence available to that method.\n\n"
        "Use `Yes`, `Partial`, `No`, or `N/A` for:\n\n"
        "- action_relevance: the selected resource/tool addresses a material uncertainty in the comparison;\n"
        "- evidence_validity: the observation is correct and warranted by the visible execution/resource;\n"
        "- evidence_faithfulness: the final reasoning represents and uses the observation without distortion;\n"
        "- verdict_support: the visible task, evidence, and reasoning support the recorded verdict;\n"
        "- auditability: another reader can reconstruct the decision from the trace.\n\n"
        "Choose one error type: none, irrelevant_action, invalid_evidence, tool_failure, unsupported_claim, reasoning_gap, final_mapping, insufficient_trace, or other. "
        "Give one concrete sentence citing the decisive trace evidence.\n",
        encoding="utf-8",
    )
    manifest = {"seed": SEED, "cases": len(inputs), "quotas_per_benchmark": QUOTAS, "input_sha256": hashlib.sha256((output / "audit_input.jsonl").read_bytes()).hexdigest(), "key_sha256": hashlib.sha256((output / "sealed_outcome_key.jsonl").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Standard and Resource-Enhanced Trace Analysis", "", "| Setting | Benchmark | N | Skill invoked | Resource views | Tool calls | Python calls | Agent steps |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for setting in SETTINGS:
        for benchmark in BENCHMARKS:
            item = report["settings"][setting][benchmark]
            lines.append(f"| {setting} | {benchmark} | {item['n']} | {item['skill_invocation_rate']:.3f} | {item['resource_views_per_sample']:.3f} | {item['tool_calls_per_sample']:.3f} | {item['python_calls_per_sample']:.3f} | {item['agent_steps_per_sample']:.3f} |")
    lines += ["", "Outcome-conditioned statistics, resource-family counts, failures, and provenance checks are retained in `joint_trace_statistics.json`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    v6_root, data_root, output = map(lambda value: Path(value).resolve(), (args.v6_root, args.data_root, args.output))
    run_root = v6_root / "runs/paper_main/qwen35_27b"
    report = summarize(run_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "joint_trace_statistics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "joint_trace_statistics.md").write_text(markdown(report), encoding="utf-8")
    make_audit(run_root, v6_root, data_root, output / "human_audit")
    print(f"output={output}")
    print("audit_cases=30 standard=21 resource_enhanced=9")


if __name__ == "__main__":
    main()
