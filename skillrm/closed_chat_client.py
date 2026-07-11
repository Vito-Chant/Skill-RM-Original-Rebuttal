from __future__ import annotations

import json
import os
import time
from itertools import count
from typing import Any

import requests
import urllib3


DEFAULT_CLOSED_API_URL = ""
_TOOL_ID_COUNTER = count(1)


def call_closed_with_retries(
    base_url: str,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    response_meta: dict[str, Any] = {}
    error = None
    retries = int(config.get("retries", 2))
    for attempt in range(1, retries + 2):
        try:
            response_meta = call_closed_chat_completion(
                messages,
                config,
                tools=tools,
                tool_choice=tool_choice,
            )
            break
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt > retries:
                break
            time.sleep(min(2**attempt, 8))
        except RuntimeError as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt > retries:
                break
            time.sleep(min(2**attempt, 8))
    reasoning = response_meta.get("reasoning")
    return {
        "latency_sec": time.time() - started_at,
        "content": response_meta.get("content", ""),
        "reasoning": reasoning,
        "thinking_field_sent": None,
        "reasoning_len": len(reasoning) if isinstance(reasoning, str) else 0,
        "finish_reason": response_meta.get("finish_reason"),
        "tool_calls": response_meta.get("tool_calls") or [],
        "error": error,
        "closed_api_model": response_meta.get("model"),
        "closed_api_uuid": response_meta.get("uuid"),
    }


def call_closed_chat_completion(
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    credentials = closed_api_credentials()
    params: dict[str, Any] = {
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 4096)),
    }
    if config.get("top_p") is not None:
        params["top_p"] = float(config.get("top_p"))
    if tools is not None:
        params["tools"] = tools
        params["tool_choice"] = tool_choice if tool_choice is not None else config.get("tool_choice", "auto")

    payload = {
        "business_unit": str(config.get("closed_api_business_unit", "")),
        "app": str(config.get("closed_api_app") or os.environ.get("CLOSED_API_APP") or ""),
        "quota_id": credentials["quota_id"],
        "model": str(config.get("model") or "gemini-3-flash-preview"),
        "prompt": normalize_closed_messages(messages),
        "params": params,
        "cache": int(config.get("closed_api_cache", 0)),
        "tag": str(config.get("closed_api_tag") or os.environ.get("CLOSED_API_TAG") or ""),
        "user_id": credentials["user_id"],
        "access_key": credentials["access_key"],
    }
    headers = {
        "content-type": "application/json",
        "token": credentials["token"],
    }
    if not bool(config.get("closed_api_verify_ssl", False)):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.trust_env = False
    api_url = str(config.get("closed_api_url") or os.environ.get("CLOSED_API_URL") or DEFAULT_CLOSED_API_URL).strip()
    if not api_url:
        raise RuntimeError("missing closed API URL; set closed_api_url in config or CLOSED_API_URL in the environment")
    response = session.post(
        api_url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        verify=bool(config.get("closed_api_verify_ssl", False)),
        timeout=float(config.get("timeout", 120)),
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") not in (0, "0", None):
        raise RuntimeError(f"closed API returned code={body.get('code')} message={body.get('message')}")
    completion = body.get("data", {}).get("completion")
    message: dict[str, Any] = {}
    finish_reason = body.get("data", {}).get("finish_reason")
    if isinstance(completion, dict) and completion.get("choices"):
        choice = completion["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason") or finish_reason
    elif isinstance(completion, dict):
        message = {"content": completion.get("content", "")}
    elif isinstance(completion, str):
        message = {"content": completion}
    return {
        "content": message.get("content") or "",
        "tool_calls": normalize_tool_calls(message.get("tool_calls") or []),
        "finish_reason": finish_reason,
        "model": body.get("data", {}).get("model"),
        "uuid": body.get("uuid"),
    }


def closed_api_credentials() -> dict[str, str]:
    required = {
        "token": "CLOSED_API_TOKEN",
        "access_key": "CLOSED_API_ACCESS_KEY",
        "quota_id": "CLOSED_API_QUOTA_ID",
        "user_id": "CLOSED_API_USER_ID",
    }
    values = {key: os.environ.get(env, "").strip() for key, env in required.items()}
    missing = [env for key, env in required.items() if not values[key]]
    if missing:
        raise RuntimeError(f"missing closed API environment variables: {', '.join(missing)}")
    return values


def normalize_closed_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("tool_calls"):
            item["tool_calls"] = normalize_tool_calls(item["tool_calls"])
        normalized.append(item)
    return normalized


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for tool_call in tool_calls:
        item = dict(tool_call)
        if not item.get("id"):
            item["id"] = f"closed_call_{next(_TOOL_ID_COUNTER)}"
        item.setdefault("type", "function")
        normalized.append(item)
    return normalized
