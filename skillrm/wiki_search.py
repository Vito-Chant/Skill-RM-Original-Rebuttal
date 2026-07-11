from __future__ import annotations

import hashlib
import time
from typing import Any

import requests


DEFAULT_WIKI_SEARCH_URL = "http://localhost:7070/retrieve"


def wiki_search_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enable_wiki_search") or config.get("enable_knowledge_search"))


def wiki_search_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": (
                "Search the local Wikipedia retrieval service for factual evidence. Use this for factual claims, "
                "named entities, dates, scientific facts, historical facts, and other externally checkable questions. "
                "The tool returns short passages from the local Wiki corpus with scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more complete semantic search queries.",
                    },
                    "topk": {
                        "type": ["integer", "null"],
                        "description": "Optional number of passages per query.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why Wikipedia evidence is needed for this judgment.",
                    },
                },
                "required": ["query_list", "reason"],
                "additionalProperties": False,
            },
        },
    }


def run_wiki_search_tool(args: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not wiki_search_enabled(config):
        return {"ok": False, "tool": "wiki_search", "error": "wiki_search disabled by config"}

    queries = normalize_queries(args)
    if not queries:
        return {"ok": False, "tool": "wiki_search", "error": "query_list is empty"}

    max_queries = int(config.get("wiki_search_max_queries_per_call", config.get("knowledge_max_queries_per_call", 3)))
    max_query_chars = int(config.get("wiki_search_max_query_chars", 512))
    queries = [query[:max_query_chars] for query in queries[:max_queries]]

    default_topk = int(config.get("wiki_search_topk", config.get("knowledge_topk", 3)))
    max_topk = int(config.get("wiki_search_max_topk", config.get("knowledge_max_topk", 5)))
    topk = args.get("topk", default_topk)
    try:
        topk = max(1, min(int(topk or default_topk), max_topk))
    except (TypeError, ValueError):
        topk = default_topk

    url = str(config.get("wiki_search_url") or config.get("knowledge_search_url") or DEFAULT_WIKI_SEARCH_URL)
    timeout = float(config.get("wiki_search_timeout_sec", config.get("knowledge_timeout_sec", 5.0)))
    payload: dict[str, Any] = {
        "queries": queries,
        "topk": topk,
        "return_scores": True,
    }
    collection = config.get("wiki_search_collection") or config.get("knowledge_collection")
    if collection:
        payload["collection"] = str(collection)

    started = time.time()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "tool": "wiki_search",
            "provider": "local_wiki",
            "url": url,
            "queries": queries,
            "topk": topk,
            "latency_sec": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }

    max_chars_per_result = int(config.get("wiki_search_max_chars_per_result", 1200))
    max_total_chars = int(config.get("wiki_search_max_total_chars", 6000))
    normalized, result_count, returned_chars = normalize_retrieval_response(
        raw,
        queries=queries,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
    )
    return {
        "ok": True,
        "tool": "wiki_search",
        "provider": "local_wiki",
        "url": url,
        "reason": str(args.get("reason") or ""),
        "queries": queries,
        "topk": topk,
        "latency_sec": time.time() - started,
        "result_count": result_count,
        "returned_chars": returned_chars,
        "result": normalized,
    }


def normalize_queries(args: dict[str, Any]) -> list[str]:
    raw = args.get("query_list")
    if raw is None:
        raw = args.get("queries")
    if raw is None:
        raw = args.get("query")
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_retrieval_response(
    raw: Any,
    *,
    queries: list[str],
    max_chars_per_result: int,
    max_total_chars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    result_lists = []
    if isinstance(raw, dict):
        value = raw.get("result", raw.get("results", []))
        if isinstance(value, list):
            result_lists = value
    elif isinstance(raw, list):
        result_lists = raw

    if result_lists and all(isinstance(item, dict) for item in result_lists):
        result_lists = [result_lists]

    normalized: list[dict[str, Any]] = []
    result_count = 0
    returned_chars = 0
    for query_index, query in enumerate(queries):
        hits = result_lists[query_index] if query_index < len(result_lists) else []
        query_hits: list[dict[str, Any]] = []
        if not isinstance(hits, list):
            hits = []
        for rank, hit in enumerate(hits, start=1):
            if returned_chars >= max_total_chars:
                break
            doc, score = split_doc_score(hit)
            if not isinstance(doc, dict):
                continue
            text = extract_doc_text(doc)
            remaining = max(0, max_total_chars - returned_chars)
            text = text[: min(max_chars_per_result, remaining)]
            returned_chars += len(text)
            result_count += 1
            query_hits.append(
                {
                    "rank": rank,
                    "score": score,
                    "doc_id": extract_doc_id(doc, text),
                    "title": extract_doc_title(doc),
                    "source": doc.get("source") or doc.get("source_path") or doc.get("url"),
                    "text": text,
                    "truncated": len(extract_doc_text(doc)) > len(text),
                }
            )
        normalized.append({"query": query, "hits": query_hits})
    return normalized, result_count, returned_chars


def split_doc_score(hit: Any) -> tuple[Any, Any]:
    if not isinstance(hit, dict):
        return hit, None
    if "document" in hit:
        return hit.get("document"), hit.get("score")
    if "doc" in hit:
        return hit.get("doc"), hit.get("score")
    return hit, hit.get("score")


def extract_doc_text(doc: dict[str, Any]) -> str:
    for key in ("text", "contents", "content", "passage", "body"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = extract_doc_title(doc)
    return title


def extract_doc_title(doc: dict[str, Any]) -> str:
    for key in ("title", "name", "doc_title"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    contents = doc.get("contents")
    if isinstance(contents, str) and contents.strip():
        return contents.splitlines()[0].strip().strip('"')
    return ""


def extract_doc_id(doc: dict[str, Any], text: str) -> str:
    for key in ("doc_id", "id", "_id", "document_id"):
        value = doc.get(key)
        if value is not None:
            return str(value)
    digest = hashlib.sha1((extract_doc_title(doc) + "\n" + text[:200]).encode("utf-8")).hexdigest()
    return digest[:16]
