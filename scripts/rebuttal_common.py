from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


BENCHMARKS = ("rewardbench2", "rmbench", "judgebench")
METHODS = ("full_fair", "subset_a", "subset_b")
EXPECTED_PREDICTION_ROWS = {
    "rewardbench2": 1865,
    "rmbench": 11943,
    "judgebench": 1240,
}
EXPECTED_TRACE_ROWS = {
    "rewardbench2": 1763,
    "rmbench": 11943,
    "judgebench": 1240,
}

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
IP_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\d.])")
WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\r\n\"'<>|]*")
UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:root|home|mnt|workspace|data|tmp|opt|srv)/[^\s\"'<>]*")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|credential|password)\b\s*[:=]\s*[\"']?([^\s,;\"'}]{6,})"
)

NETWORK_KEYS = {"endpoint", "endpoints", "base_url", "base_urls", "url", "urls"}
PATH_KEYS = {
    "data_source",
    "data_sources",
    "output_dir",
    "repo_root",
    "run_root",
}
SECRET_KEY_PARTS = ("api_key", "access_token", "secret", "credential", "password")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_text(text: str) -> str:
    text = BEARER_RE.sub("Bearer <redacted-credential>", text)
    text = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted-credential>", text)
    text = URL_RE.sub("<redacted-url>", text)
    text = IP_RE.sub("<redacted-ip>", text)
    text = WINDOWS_PATH_RE.sub("<redacted-absolute-path>", text)
    text = UNIX_PATH_RE.sub("<redacted-absolute-path>", text)
    return text


def sanitize_value(value: Any, *, key: str = "") -> Any:
    lower_key = key.lower()
    if any(part in lower_key for part in SECRET_KEY_PARTS):
        return "<redacted-credential>"
    if lower_key in NETWORK_KEYS:
        if isinstance(value, list):
            return ["<redacted-network-value>" for _ in value]
        return "<redacted-network-value>"
    if lower_key in PATH_KEYS and isinstance(value, (str, list, dict)):
        if isinstance(value, str):
            return "<redacted-path>"
        if isinstance(value, list):
            return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(item_key): sanitize_value(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def sensitive_matches(text: str) -> list[str]:
    matches: list[str] = []
    checks = (
        ("url", URL_RE),
        ("ip", IP_RE),
        ("windows_absolute_path", WINDOWS_PATH_RE),
        ("unix_private_path", UNIX_PATH_RE),
        ("bearer_credential", BEARER_RE),
        ("credential_assignment", SECRET_ASSIGNMENT_RE),
    )
    for label, pattern in checks:
        if pattern.search(text):
            matches.append(label)
    return matches


def assert_text_sanitized(name: str, text: str) -> None:
    matches = sensitive_matches(text)
    if matches:
        raise ValueError(f"Sensitive patterns remained in {name}: {', '.join(matches)}")


def tree_hash(root: Path, *, include: Iterable[Path] | None = None) -> str:
    files = list(include) if include is not None else [path for path in root.rglob("*") if path.is_file()]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
