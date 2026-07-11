#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from urllib.parse import urlsplit, urlunsplit


def split_raw_urls(raw: str) -> list[str]:
    normalized = raw.translate(str.maketrans({"，": ",", "；": ",", ";": ",", "、": ","}))
    return [item.strip() for item in re.split(r"[\s,]+", normalized) if item.strip()]


def normalize_vllm_url(value: str) -> str:
    text = value.strip().rstrip("/")
    if not text:
        raise ValueError("empty endpoint")
    if "://" not in text:
        text = "http://" + text
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid endpoint")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("endpoint must not contain credentials, query, or fragment")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def normalize_vllm_urls(raw: str) -> list[str]:
    return list(dict.fromkeys(normalize_vllm_url(item) for item in split_raw_urls(raw)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize vLLM OpenAI-compatible endpoints.")
    parser.add_argument("raw", nargs="?", default=os.environ.get("SKILLRM_BASE_URLS", ""))
    args = parser.parse_args()
    urls = normalize_vllm_urls(args.raw)
    if not urls:
        raise SystemExit("No endpoints supplied.")
    print(",".join(urls))


if __name__ == "__main__":
    main()
