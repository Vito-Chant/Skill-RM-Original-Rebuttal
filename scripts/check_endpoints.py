#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests

from normalize_vllm_urls import normalize_vllm_urls


def request_json(session: requests.Session, url: str, *, timeout: float) -> tuple[bool, Any]:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, type(exc).__name__


def post_completion(
    session: requests.Session,
    base_url: str,
    *,
    model: str,
    timeout: float,
    require_tool: bool,
) -> tuple[bool, str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Call final_answer when a tool is available; otherwise reply exactly OK."},
            {"role": "user", "content": "Submit now."},
        ],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if require_tool:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Submit the verdict.",
                    "parameters": {
                        "type": "object",
                        "properties": {"verdict": {"type": "string", "enum": ["OK"]}},
                        "required": ["verdict"],
                    },
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "final_answer"}}
    try:
        response = session.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            payload.pop("chat_template_kwargs", None)
            response = session.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                timeout=timeout,
            )
        response.raise_for_status()
        message = ((response.json().get("choices") or [{}])[0].get("message") or {})
        if require_tool:
            calls = message.get("tool_calls") or []
            name = (((calls[0] if calls else {}).get("function") or {}).get("name"))
            return name == "final_answer", "tool-call missing" if name != "final_answer" else "ok"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI-compatible Skill-RM endpoints.")
    parser.add_argument("--base-urls", default=os.environ.get("SKILLRM_BASE_URLS", ""))
    parser.add_argument("--model", default=os.environ.get("SKILLRM_MODEL", "Qwen3.5-27B"))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--redact-urls", action="store_true")
    parser.add_argument("--trust-env", action="store_true")
    args = parser.parse_args()

    try:
        urls = normalize_vllm_urls(args.base_urls)
    except ValueError as exc:
        print(f"Endpoint configuration is invalid: {exc}", file=sys.stderr)
        return 2
    if not urls:
        print("No endpoints configured.", file=sys.stderr)
        return 2

    session = requests.Session()
    session.trust_env = bool(args.trust_env)
    failures = 0
    for index, base_url in enumerate(urls, start=1):
        label = f"endpoint[{index}]" if args.redact_urls else base_url
        ok, detail = request_json(session, f"{base_url}/models", timeout=args.timeout)
        print(f"{label}/models: {'ok' if ok else 'failed'}{'' if ok else ' <redacted-error>'}")
        if not ok:
            failures += 1
            continue
        if args.chat:
            chat_ok, _ = post_completion(session, base_url, model=args.model, timeout=args.timeout, require_tool=False)
            print(f"{label}/chat: {'ok' if chat_ok else 'failed <redacted-error>'}")
            failures += 0 if chat_ok else 1
        if args.tools:
            tool_ok, _ = post_completion(session, base_url, model=args.model, timeout=args.timeout, require_tool=True)
            print(f"{label}/tool-call: {'ok' if tool_ok else 'failed <redacted-error>'}")
            failures += 0 if tool_ok else 1
    print(f"checked={len(urls)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
