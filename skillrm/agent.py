from __future__ import annotations

import json
import re
from typing import Any

from .llm import ChatLLM
from .resources import ResourceBank, ResourceError
from .types import AgentDecision, RB2Example


SYSTEM_PROMPT = """You are a Skill-RM RewardBench 2 judge.
The benchmark subset and gold labels are hidden from you.
You may use the mock reward-judge-controller skill through JSON actions.

Allowed action format:
{"action": "view_skill", "arguments": {"skill_name": "reward-judge-controller"}}
{"action": "list_resources", "arguments": {"skill_name": "reward-judge-controller", "type": null}}
{"action": "view_resource", "arguments": {"skill_name": "reward-judge-controller", "path": "references/output_schema.md"}}

Final answer format:
{"final": {"best_label": "A", "ranking": ["A", "B", "C", "D"], "confidence": "low|medium|high", "decision_basis": "mock|rubric|verifier|mixed", "resources_used": [], "evidence": []}}

Return JSON only. Do not mention hidden metadata or use benchmark subset routing.
"""


class SkillJudgeAgent:
    def __init__(
        self,
        *,
        llm: ChatLLM,
        resources: ResourceBank,
        max_steps: int = 4,
    ) -> None:
        self.llm = llm
        self.resources = resources
        self.max_steps = max_steps

    def judge(self, example: RB2Example) -> AgentDecision:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(example.to_agent_payload(), ensure_ascii=False),
            },
        ]
        trace: list[dict[str, Any]] = []

        for step in range(self.max_steps):
            raw = self.llm.complete(messages)
            parsed = parse_json_object(raw)
            trace.append({"step": step, "assistant_raw": raw, "parsed": parsed})

            final = _extract_final(parsed)
            if final is not None:
                return self._decision_from_final(example, final, trace)

            if parsed.get("action"):
                tool_name = str(parsed["action"])
                args = dict(parsed.get("arguments") or {})
                tool_result = self._execute_tool(tool_name, args)
                trace[-1]["tool_result"] = tool_result
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT {tool_name}: "
                        + json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                continue

            return AgentDecision(
                sample_id=example.sample_id,
                best_label=None,
                final={},
                trace=trace,
                valid=False,
                error="assistant returned neither action nor final",
            )

        return AgentDecision(
            sample_id=example.sample_id,
            best_label=None,
            final={},
            trace=trace,
            valid=False,
            error=f"max_steps exceeded: {self.max_steps}",
        )

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name == "view_skill":
                return {
                    "ok": True,
                    "result": self.resources.view_skill(str(args["skill_name"])),
                }
            if tool_name == "list_resources":
                resource_type = args.get("type")
                return {
                    "ok": True,
                    "result": self.resources.list_resources(
                        str(args["skill_name"]),
                        str(resource_type) if resource_type else None,
                    ),
                }
            if tool_name == "view_resource":
                return {
                    "ok": True,
                    "result": self.resources.view_resource(
                        str(args["skill_name"]),
                        str(args["path"]),
                        start_line=int(args.get("start_line", 1)),
                        max_lines=int(args.get("max_lines", 120)),
                    ),
                }
            if tool_name == "python_sandbox":
                return {
                    "ok": False,
                    "error": "python_sandbox is allowed by policy but not implemented in this first framework pass.",
                }
            return {"ok": False, "error": f"unknown tool: {tool_name}"}
        except (KeyError, ResourceError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def _decision_from_final(
        self,
        example: RB2Example,
        final: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> AgentDecision:
        best_label = final.get("best_label")
        if best_label not in example.responses:
            return AgentDecision(
                sample_id=example.sample_id,
                best_label=str(best_label) if best_label is not None else None,
                final=final,
                trace=trace,
                valid=False,
                error=f"invalid best_label: {best_label}",
            )
        return AgentDecision(
            sample_id=example.sample_id,
            best_label=str(best_label),
            final=final,
            trace=trace,
            valid=True,
        )


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_final(parsed: dict[str, Any]) -> dict[str, Any] | None:
    final = parsed.get("final")
    if isinstance(final, dict):
        return final
    if "best_label" in parsed:
        return parsed
    return None
