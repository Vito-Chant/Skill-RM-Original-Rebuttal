from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


SOURCE_SKILL = Path("skills/reward_judge_fair")
CONFIG_SOURCES = {
    "rewardbench2": Path("configs/rewardbench2/skill_fair.yaml"),
    "rmbench": Path("configs/rmbench/skill_fair.yaml"),
    "judgebench": Path("configs/judgebench/skill_fair.yaml"),
}

ELIGIBLE_RESOURCE_IDS = [
    "rubric.generic_pairwise",
    "principle.generic",
    "bias_control",
    "aggregation.generic",
]
FIXED_RESOURCE_IDS = ["output_format"]
FIXED_RUNTIME_RESOURCE_IDS = ["tool.python_sandbox"]
SAMPLING_SEED = 0
SAMPLING_ALGORITHM = "random.Random(seed).shuffle over manifest eligible order; first half is A"

_shuffled_resource_ids = list(ELIGIBLE_RESOURCE_IDS)
random.Random(SAMPLING_SEED).shuffle(_shuffled_resource_ids)
_subset_a_ids = set(_shuffled_resource_ids[: len(_shuffled_resource_ids) // 2])

SUBSETS = {
    "subset_a": {
        "description": "seed-0 2-of-4 sample over optional generic judging resources",
        "kept_resource_ids": [item for item in ELIGIBLE_RESOURCE_IDS if item in _subset_a_ids],
    },
    "subset_b": {
        "description": "complement of subset_a over optional generic judging resources",
        "kept_resource_ids": [item for item in ELIGIBLE_RESOURCE_IDS if item not in _subset_a_ids],
    },
}

GENERATED_MARKER = "skill-rm-original-v6-rebuttal-subset-v1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True, width=1000)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def manifest_ids(manifest: dict[str, Any], group: str) -> list[str]:
    return [
        str(item["id"])
        for item in manifest.get(group, [])
        if isinstance(item, dict) and item.get("id")
    ]


def resource_paths(manifest: dict[str, Any]) -> list[str]:
    paths: set[str] = set()
    for group in ("resources", "runtime_resources"):
        for item in manifest.get(group, []):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.add(item["path"])
    return sorted(paths)


def copy_relative(source_root: Path, destination_root: Path, relative: str) -> None:
    source = source_root / relative
    destination = destination_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def remove_previous_generated_directory(path: Path) -> None:
    if not path.exists():
        return
    metadata_path = path / "SUBSET_METADATA.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Refusing to replace unmarked directory: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("generated_by") != GENERATED_MARKER:
        raise RuntimeError(f"Refusing to replace directory with an unknown marker: {path}")
    shutil.rmtree(path)


def filtered_manifest(
    source_manifest: dict[str, Any],
    optional_ids: list[str],
) -> dict[str, Any]:
    kept_resources = set(optional_ids) | set(FIXED_RESOURCE_IDS)
    kept_runtime = set(FIXED_RUNTIME_RESOURCE_IDS)
    return {
        "resources": [
            item
            for item in source_manifest.get("resources", [])
            if isinstance(item, dict) and item.get("id") in kept_resources
        ],
        "runtime_resources": [
            item
            for item in source_manifest.get("runtime_resources", [])
            if isinstance(item, dict) and item.get("id") in kept_runtime
        ],
    }


def prepare_subset_skill(
    repo_root: Path,
    subset_id: str,
    spec: dict[str, Any],
    source_tree_hash: str,
) -> tuple[Path, dict[str, Any]]:
    source = repo_root / SOURCE_SKILL
    destination = repo_root / "skills" / f"reward_judge_fair_50pct_{subset_id}"
    source_manifest = load_yaml(source / "resources.yaml")
    manifest = filtered_manifest(source_manifest, list(spec["kept_resource_ids"]))

    remove_previous_generated_directory(destination)
    destination.mkdir(parents=True)
    copy_relative(source, destination, "SKILL.md")
    write_yaml(destination / "resources.yaml", manifest)
    for relative in resource_paths(manifest):
        copy_relative(source, destination, relative)

    removed_ids = [
        resource_id
        for resource_id in ELIGIBLE_RESOURCE_IDS
        if resource_id not in spec["kept_resource_ids"]
    ]
    metadata = {
        "generated_by": GENERATED_MARKER,
        "subset_id": subset_id,
        "description": spec["description"],
        "sampling_seed": SAMPLING_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "sampling_granularity": "complete resources.yaml manifest entry",
        "source_skill": SOURCE_SKILL.as_posix(),
        "source_skill_tree_sha256": source_tree_hash,
        "eligible_resource_ids": ELIGIBLE_RESOURCE_IDS,
        "kept_resource_ids": spec["kept_resource_ids"],
        "removed_resource_ids": removed_ids,
        "fixed_resource_ids": FIXED_RESOURCE_IDS,
        "fixed_runtime_resource_ids": FIXED_RUNTIME_RESOURCE_IDS,
        "generated_manifest_resource_ids": manifest_ids(manifest, "resources"),
        "generated_manifest_runtime_resource_ids": manifest_ids(manifest, "runtime_resources"),
    }
    (destination / "SUBSET_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination, metadata


def assert_only_expected_config_changes(
    source: dict[str, Any],
    generated: dict[str, Any],
    allowed_changes: set[str],
) -> None:
    changed = {
        key
        for key in set(source) | set(generated)
        if source.get(key) != generated.get(key)
    }
    unexpected = changed - allowed_changes
    if unexpected:
        raise AssertionError(f"Unexpected generated config changes: {sorted(unexpected)}")


def prepare_config(
    repo_root: Path,
    bench: str,
    method: str,
    run_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source_path = repo_root / CONFIG_SOURCES[bench]
    source = load_yaml(source_path)
    generated = dict(source)

    if method == "full_fair":
        generated["output_dir"] = (run_root / method / bench).as_posix()
        allowed_changes = {"output_dir"}
    else:
        generated["skill_path"] = f"skills/reward_judge_fair_50pct_{method}"
        generated["output_dir"] = (run_root / method / bench).as_posix()
        generated["setting_name"] = f"skill_fair_50pct_{method}"
        generated["resource_subset_ablation"] = {
            "generated_by": GENERATED_MARKER,
            "subset_id": method,
            "sampling_seed": SAMPLING_SEED,
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "sampling_granularity": "complete resources.yaml manifest entry",
            "eligible_resource_ids": ELIGIBLE_RESOURCE_IDS,
            "kept_resource_ids": SUBSETS[method]["kept_resource_ids"],
            "fixed_resource_ids": FIXED_RESOURCE_IDS,
            "fixed_runtime_resource_ids": FIXED_RUNTIME_RESOURCE_IDS,
        }
        allowed_changes = {
            "skill_path",
            "output_dir",
            "setting_name",
            "resource_subset_ablation",
        }

    assert_only_expected_config_changes(source, generated, allowed_changes)
    output_path = (
        repo_root
        / "configs"
        / "rebuttal_original_v6"
        / f"{bench}_{method}.yaml"
    )
    write_yaml(output_path, generated)
    return output_path, {
        "benchmark": bench,
        "method": method,
        "source_config": source_path.relative_to(repo_root).as_posix(),
        "generated_config": output_path.relative_to(repo_root).as_posix(),
        "source_config_sha256": file_sha256(source_path),
        "generated_config_sha256": file_sha256(output_path),
        "changed_keys": sorted(allowed_changes),
    }


def validate_source_manifest(source_manifest: dict[str, Any]) -> None:
    optional = set(ELIGIBLE_RESOURCE_IDS)
    available = set(manifest_ids(source_manifest, "resources"))
    missing = optional - available
    if missing:
        raise ValueError(f"Source fair manifest is missing expected resources: {sorted(missing)}")
    if not set(FIXED_RESOURCE_IDS).issubset(available):
        raise ValueError("Source fair manifest is missing the fixed output contract.")
    runtime = set(manifest_ids(source_manifest, "runtime_resources"))
    if not set(FIXED_RUNTIME_RESOURCE_IDS).issubset(runtime):
        raise ValueError("Source fair manifest is missing the fixed Python sandbox.")


def audit_fair_setup(repo_root: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    resources = [item for item in source_manifest.get("resources", []) if isinstance(item, dict)]
    runtime = [item for item in source_manifest.get("runtime_resources", []) if isinstance(item, dict)]
    expected_resources = set(ELIGIBLE_RESOURCE_IDS + FIXED_RESOURCE_IDS)
    actual_resources = {str(item.get("id")) for item in resources}
    if actual_resources != expected_resources:
        raise ValueError(f"Unexpected fair resource IDs: {sorted(actual_resources)}")
    if {str(item.get("id")) for item in runtime} != set(FIXED_RUNTIME_RESOURCE_IDS):
        raise ValueError("Unexpected fair runtime resources.")

    manifest_entries = resources + runtime
    non_visible = [item.get("id") for item in manifest_entries if item.get("leakage_level") != "sample_visible"]
    non_fair = [
        item.get("id")
        for item in manifest_entries
        if "skill_fair" not in (item.get("allowed_setting") or [])
    ]
    suspicious_ids = [
        item.get("id")
        for item in manifest_entries
        if any(token in str(item.get("id", "")).lower() for token in ("benchmark", "ground_truth", "gold", "verifier"))
    ]
    if non_visible or non_fair or suspicious_ids:
        raise ValueError(
            f"Fair manifest audit failed: non_visible={non_visible}, non_fair={non_fair}, suspicious={suspicious_ids}"
        )

    config_checks: dict[str, Any] = {}
    for bench, relative in CONFIG_SOURCES.items():
        config = load_yaml(repo_root / relative)
        checks = {
            "skill_path_is_fair": config.get("skill_path") == "skills/reward_judge_fair",
            "allowed_setting_is_fair": config.get("skill_allowed_setting") == "skill_fair",
            "no_allowed_skill_scripts": not (config.get("allowed_skill_scripts") or []),
            "delegated_agents_disabled": not bool(config.get("enable_delegated_agents", False)),
        }
        if not all(checks.values()):
            raise ValueError(f"Fair config audit failed for {bench}: {checks}")
        config_checks[bench] = checks
    return {
        "passed": True,
        "resource_ids": [item["id"] for item in resources],
        "runtime_resource_ids": [item["id"] for item in runtime],
        "all_resources_sample_visible": True,
        "benchmark_specific_or_verifier_resources": [],
        "config_checks": config_checks,
        "interpretation": (
            "The fair path exposes only generic sample-visible judging guidance and a local Python sandbox; "
            "it does not load benchmark-specific references, gold labels, ground truth, or verifier resources."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare matched full/A/B resource-pool runs for the archived Skill-RM v6 code."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--run-root",
        default="outputs/original_v6_rebuttal/qwen35_27b",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root)
    source_skill = repo_root / SOURCE_SKILL
    if not source_skill.is_dir():
        raise FileNotFoundError(f"Run from the archived skill-rm-v6 root: {source_skill}")
    for relative in CONFIG_SOURCES.values():
        if not (repo_root / relative).is_file():
            raise FileNotFoundError(repo_root / relative)

    source_manifest = load_yaml(source_skill / "resources.yaml")
    validate_source_manifest(source_manifest)
    fair_leakage_audit = audit_fair_setup(repo_root, source_manifest)
    source_tree_hash_before = tree_sha256(source_skill)

    generated_skills: list[dict[str, Any]] = []
    for subset_id, spec in SUBSETS.items():
        path, metadata = prepare_subset_skill(
            repo_root,
            subset_id,
            spec,
            source_tree_hash_before,
        )
        generated_skills.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "tree_sha256": tree_sha256(path),
                "metadata": metadata,
            }
        )

    generated_configs: list[dict[str, Any]] = []
    for method in ("full_fair", "subset_a", "subset_b"):
        for bench in CONFIG_SOURCES:
            _, metadata = prepare_config(repo_root, bench, method, run_root)
            generated_configs.append(metadata)

    source_tree_hash_after = tree_sha256(source_skill)
    if source_tree_hash_before != source_tree_hash_after:
        raise AssertionError("The source reward_judge_fair package changed during preparation.")

    report = {
        "generated_by": GENERATED_MARKER,
        "repo_root": str(repo_root),
        "run_root": run_root.as_posix(),
        "source_skill_tree_sha256_before": source_tree_hash_before,
        "source_skill_tree_sha256_after": source_tree_hash_after,
        "source_skill_unchanged": True,
        "fair_leakage_audit": fair_leakage_audit,
        "generated_skills": generated_skills,
        "generated_configs": generated_configs,
    }
    report_path = repo_root / "configs" / "rebuttal_original_v6" / "PREPARATION_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
