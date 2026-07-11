from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class RB2Example:
    sample_id: str
    prompt: str
    responses: dict[str, str]
    chosen_label: str
    subset: str | None = None
    visible_metadata: dict[str, Any] = field(default_factory=dict)
    hidden_metadata: dict[str, Any] = field(default_factory=dict)
    source_record: dict[str, Any] = field(default_factory=dict)

    def labels(self) -> list[str]:
        return list(self.responses.keys())

    def to_agent_payload(self) -> dict[str, Any]:
        """Return only fields the agent may see."""
        return {
            "id": self.sample_id,
            "prompt": self.prompt,
            "responses": self.responses,
            "visible_metadata": self.visible_metadata,
        }


@dataclass
class AgentDecision:
    sample_id: str
    best_label: str | None
    final: dict[str, Any]
    trace: list[dict[str, Any]]
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    sample_id: str
    subset: str | None
    chosen_label: str
    predicted_label: str | None
    correct: bool
    valid: bool
    error: str | None = None
