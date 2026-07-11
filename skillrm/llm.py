from __future__ import annotations

import json
import os
from typing import Any, Protocol


class ChatLLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class MockLLM:
    """Deterministic backend for smoke tests and framework debugging."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(message["content"] for message in messages)
        if "TOOL_RESULT view_skill" not in transcript:
            return json.dumps(
                {
                    "action": "view_skill",
                    "arguments": {"skill_name": "reward-judge-controller"},
                }
            )
        return json.dumps(
            {
                "final": {
                    "best_label": "A",
                    "ranking": ["A", "B", "C", "D"],
                    "confidence": "low",
                    "decision_basis": "mock",
                    "resources_used": [
                        {
                            "path": "SKILL.md",
                            "why": "Smoke-test mock controller was loaded.",
                        }
                    ],
                    "evidence": [],
                }
            }
        )


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        endpoint_index: int = 0,
    ) -> "OpenAICompatibleLLM":
        endpoints = [
            item.strip()
            for item in os.getenv("SKILLRM_VLLM_ENDPOINTS", "").split(",")
            if item.strip()
        ]
        resolved_base_url = base_url or (endpoints[endpoint_index] if endpoints else None)
        if not resolved_base_url:
            raise ValueError(
                "No vLLM endpoint configured. Pass --base-url or set SKILLRM_VLLM_ENDPOINTS."
            )
        return cls(
            base_url=resolved_base_url,
            model=model or os.getenv("SKILLRM_MODEL", "Qwen3.5-27B"),
            api_key=os.getenv("SKILLRM_API_KEY", "EMPTY"),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return content or ""
