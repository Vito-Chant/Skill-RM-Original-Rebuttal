from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import AgentDecision, BenchmarkResult, RB2Example


def evaluate_decisions(
    examples: list[RB2Example],
    decisions: list[AgentDecision],
) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    by_id = {decision.sample_id: decision for decision in decisions}
    results: list[BenchmarkResult] = []
    for example in examples:
        decision = by_id.get(example.sample_id)
        predicted = decision.best_label if decision else None
        valid = bool(decision and decision.valid)
        correct = bool(valid and predicted == example.chosen_label)
        results.append(
            BenchmarkResult(
                sample_id=example.sample_id,
                subset=example.subset,
                chosen_label=example.chosen_label,
                predicted_label=predicted,
                correct=correct,
                valid=valid,
                error=decision.error if decision else "missing decision",
            )
        )
    return results, build_metrics(results)


def build_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
    total = len(results)
    valid_count = sum(1 for result in results if result.valid)
    correct_count = sum(1 for result in results if result.correct)
    by_subset: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        by_subset[str(result.subset or "unknown")].append(result)

    subset_metrics = {}
    for subset, subset_results in sorted(by_subset.items()):
        n = len(subset_results)
        subset_metrics[subset] = {
            "n": n,
            "valid": sum(1 for item in subset_results if item.valid),
            "correct": sum(1 for item in subset_results if item.correct),
            "accuracy": _safe_div(sum(1 for item in subset_results if item.correct), n),
            "invalid_rate": _safe_div(sum(1 for item in subset_results if not item.valid), n),
        }

    macro_accuracy = _safe_div(
        sum(item["accuracy"] for item in subset_metrics.values()),
        len(subset_metrics),
    )

    return {
        "n": total,
        "valid": valid_count,
        "correct": correct_count,
        "micro_accuracy": _safe_div(correct_count, total),
        "macro_accuracy_by_subset": macro_accuracy,
        "invalid_rate": _safe_div(total - valid_count, total),
        "by_subset": subset_metrics,
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
