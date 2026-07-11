from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from normalize_vllm_urls import normalize_vllm_url, normalize_vllm_urls
from prepare_original_v6_resource_subsets import SUBSETS, filtered_manifest
from prepare_rebuttal_data import validate_jsonl
from rebuttal_common import assert_text_sanitized, sha256_file
from verify_original_sources import verify


ROOT = Path(__file__).resolve().parents[1]


def test_url_normalization_accepts_bare_and_chinese_separators() -> None:
    urls = normalize_vllm_urls("10.0.0.1:8000， http://10.0.0.2:8001/v1；10.0.0.1:8000")
    assert urls == ["http://10.0.0.1:8000/v1", "http://10.0.0.2:8001/v1"]
    assert normalize_vllm_url("https://host.example/api") == "https://host.example/api/v1"


def test_url_normalization_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError):
        normalize_vllm_url("http://user:password@host.example:8000/v1")


def test_complementary_subsets_are_exact_and_keep_infrastructure() -> None:
    assert SUBSETS["subset_a"]["kept_resource_ids"] == ["rubric.generic_pairwise", "bias_control"]
    assert SUBSETS["subset_b"]["kept_resource_ids"] == ["principle.generic", "aggregation.generic"]
    source = yaml.safe_load((ROOT / "skills/reward_judge_fair/resources.yaml").read_text(encoding="utf-8"))
    a = filtered_manifest(source, SUBSETS["subset_a"]["kept_resource_ids"])
    b = filtered_manifest(source, SUBSETS["subset_b"]["kept_resource_ids"])
    assert [item["id"] for item in a["runtime_resources"]] == ["tool.python_sandbox"]
    assert [item["id"] for item in b["runtime_resources"]] == ["tool.python_sandbox"]
    assert "output_format" in {item["id"] for item in a["resources"]}
    assert "output_format" in {item["id"] for item in b["resources"]}


def test_original_source_hash_manifest_verifies() -> None:
    assert verify(ROOT, ROOT / "ORIGINAL_V6_SHA256SUMS") >= 50


def test_data_validator_checks_rows_json_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    validate_jsonl(path, expected_rows=2, expected_sha256=sha256_file(path))
    with pytest.raises(RuntimeError, match="row count"):
        validate_jsonl(path, expected_rows=3, expected_sha256=sha256_file(path))


def test_sensitive_scanner_rejects_all_required_classes() -> None:
    values = [
        "http://host.example/v1",
        "10.20.30.40:8000",
        r"C:\\company\\workspace\\run.json",
        "/" + "root/workspace/run.json",
        "Authorization: " + "Bear" + "er " + "abcdef123456",
        "api_" + "key=abcdef123456",
    ]
    for value in values:
        with pytest.raises(ValueError):
            assert_text_sanitized("injected", value)


def test_runner_declares_exactly_nine_dry_run_jobs() -> None:
    text = (ROOT / "scripts/run_original_v6_rebuttal.sh").read_text(encoding="utf-8")
    assert "run_one full_fair" in text
    assert "run_one subset_a" in text
    assert "run_one subset_b" in text
    assert "run_one baseline" not in text
    assert "run_one skill_operational" not in text
    assert "DRY_RUN method=" in text
