from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .agent import SkillJudgeAgent
from .evaluator import evaluate_decisions
from .llm import MockLLM, OpenAICompatibleLLM
from .rb2 import load_rb2_examples
from .resources import ResourceBank


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli(config, args)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_rb2_examples(
        config["data_source"],
        limit=config.get("limit"),
        seed=int(config.get("seed", 0)),
        expose_subset=bool(config.get("expose_subset", False)),
        include_ties=bool(config.get("include_ties", False)),
    )

    llm = build_llm(config)
    resources = ResourceBank(
        root=config.get("skill_root", "skills"),
        setting=config.get("resource_setting", "blind"),
    )
    agent = SkillJudgeAgent(
        llm=llm,
        resources=resources,
        max_steps=int(config.get("max_steps", 4)),
    )
    decisions = [agent.judge(example) for example in examples]
    results, metrics = evaluate_decisions(examples, decisions)

    write_outputs(output_dir, config, examples, decisions, results, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Skill-RM RBv2 MVP experiments.")
    parser.add_argument("--config", help="YAML config path.")
    parser.add_argument("--data", dest="data_source", help="RBv2 data source path or hf:// repo.")
    parser.add_argument("--output", dest="output_dir", help="Output run directory.")
    parser.add_argument("--backend", choices=["mock", "openai"], help="LLM backend.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint base URL.")
    parser.add_argument("--model", help="Model name for OpenAI-compatible endpoint.")
    parser.add_argument("--limit", type=int, help="Number of samples to evaluate.")
    parser.add_argument("--seed", type=int, help="Response shuffle seed.")
    parser.add_argument("--skill-root", help="Skill root directory.")
    parser.add_argument("--max-steps", type=int, help="Max agent loop steps.")
    parser.add_argument("--include-ties", action="store_true", help="Include Ties subset.")
    parser.add_argument("--expose-subset", action="store_true", help="Expose subset to agent.")
    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in (
        "data_source",
        "output_dir",
        "backend",
        "base_url",
        "model",
        "limit",
        "seed",
        "skill_root",
        "max_steps",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    if args.include_ties:
        merged["include_ties"] = True
    if args.expose_subset:
        merged["expose_subset"] = True

    merged.setdefault("backend", "mock")
    merged.setdefault("output_dir", "runs/smoke")
    merged.setdefault("data_source", "data/smoke/rb2_mock.jsonl")
    merged.setdefault("seed", 0)
    merged.setdefault("include_ties", False)
    merged.setdefault("expose_subset", False)
    merged.setdefault("resource_setting", "blind")
    return merged


def build_llm(config: dict[str, Any]):
    backend = config.get("backend", "mock")
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAICompatibleLLM.from_env(
            base_url=config.get("base_url"),
            model=config.get("model"),
        )
    raise ValueError(f"Unsupported backend: {backend}")


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    examples,
    decisions,
    results,
    metrics: dict[str, Any],
) -> None:
    with (output_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for example, decision, result in zip(examples, decisions, results, strict=True):
            row = {
                "sample_id": example.sample_id,
                "predicted_label": decision.best_label,
                "chosen_label": example.chosen_label,
                "correct": result.correct,
                "valid": result.valid,
                "subset_for_metrics_only": example.subset,
                "final": decision.final,
                "error": decision.error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(
                json.dumps(
                    {"sample_id": decision.sample_id, "trace": decision.trace},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    summary = [
        "# Skill-RM RBv2 Run Summary",
        "",
        f"- created_at: {datetime.now(timezone.utc).isoformat()}",
        f"- data_source: {config.get('data_source')}",
        f"- backend: {config.get('backend')}",
        f"- n: {metrics['n']}",
        f"- micro_accuracy: {metrics['micro_accuracy']}",
        f"- macro_accuracy_by_subset: {metrics['macro_accuracy_by_subset']}",
        f"- invalid_rate: {metrics['invalid_rate']}",
        "",
        "Subset is kept out of the agent payload unless `expose_subset=true`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
