from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class ResourceError(ValueError):
    pass


class ResourceBank:
    def __init__(self, root: str | Path = "skills", setting: str = "blind") -> None:
        self.root = Path(root)
        self.setting = setting

    def view_skill(self, skill_name: str) -> dict[str, Any]:
        skill_path = self._skill_dir(skill_name) / "SKILL.md"
        text = self._read_text(skill_path)
        return {
            "skill_name": skill_name,
            "path": "SKILL.md",
            "content": text,
            "sha256": _sha256(text),
        }

    def list_resources(self, skill_name: str, resource_type: str | None = None) -> dict[str, Any]:
        resources = self._load_index(skill_name)
        filtered = []
        for item in resources:
            if resource_type and item.get("type") != resource_type:
                continue
            if not self._resource_allowed(item):
                continue
            filtered.append(item)
        return {"skill_name": skill_name, "resources": filtered}

    def view_resource(
        self,
        skill_name: str,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = 120,
    ) -> dict[str, Any]:
        item = self._lookup_resource(skill_name, path)
        if not self._resource_allowed(item):
            raise ResourceError(f"Resource blocked by setting={self.setting}: {path}")

        resource_path = self._safe_join(self._skill_dir(skill_name), path)
        text = self._read_text(resource_path)
        lines = text.splitlines()
        start = max(start_line, 1) - 1
        end = min(start + max_lines, len(lines))
        content = "\n".join(lines[start:end])
        return {
            "skill_name": skill_name,
            "path": path,
            "resource": item,
            "start_line": start + 1,
            "end_line": end,
            "content": content,
            "sha256": _sha256(text),
        }

    def _skill_dir(self, skill_name: str) -> Path:
        path = self._safe_join(self.root, skill_name)
        if not path.is_dir():
            raise ResourceError(f"Unknown skill: {skill_name}")
        return path

    def _load_index(self, skill_name: str) -> list[dict[str, Any]]:
        index_path = self._skill_dir(skill_name) / "resource_index.yaml"
        data = yaml.safe_load(self._read_text(index_path)) or {}
        resources = data.get("resources", [])
        if not isinstance(resources, list):
            raise ResourceError(f"Invalid resource index for skill: {skill_name}")
        return [dict(item) for item in resources]

    def _lookup_resource(self, skill_name: str, path: str) -> dict[str, Any]:
        for item in self._load_index(skill_name):
            if item.get("path") == path:
                return item
        raise ResourceError(f"Unregistered resource path: {path}")

    def _resource_allowed(self, item: dict[str, Any]) -> bool:
        leakage_level = str(item.get("leakage_level", "blind"))
        if self.setting in {"blind", "visible"} and leakage_level == "oracle":
            return False
        return True

    def _safe_join(self, base: Path, relative: str) -> Path:
        target = (base / relative).resolve()
        base_resolved = base.resolve()
        if target != base_resolved and base_resolved not in target.parents:
            raise ResourceError(f"Path escapes resource root: {relative}")
        return target

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ResourceError(f"Resource not found: {path}") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
