#!/usr/bin/env python3
"""Measure actual TTFT and streamed decode separately through an OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    unit = "benchmark context datum "
    prompt = (
        f"unique request {time.time_ns()} "
        + unit * max(1, args.prompt_tokens // 3)
        + "\nReturn exactly 128 numbered lowercase English words, then stop."
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )

    started = time.monotonic()
    first_wire = first_token = finished = None
    usage = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            now = time.monotonic()
            first_wire = first_wire or now
            if line == "data: [DONE]":
                finished = now
                break
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if any(
                delta.get(key)
                for key in ("content", "reasoning", "reasoning_content", "tool_calls")
            ):
                first_token = first_token or now

    if first_token is None or finished is None:
        raise SystemExit("stream ended without a model token or [DONE]")
    prompt_tokens = (usage or {}).get("prompt_tokens", args.prompt_tokens)
    completion_tokens = (usage or {}).get("completion_tokens", args.max_tokens)
    report = {
        "first_wire_s": first_wire - started,
        "ttft_s": first_token - started,
        "elapsed_s": finished - started,
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": prompt_tokens / (first_token - started),
        "completion_tokens": completion_tokens,
        "decode_tok_s": completion_tokens / max(0.001, finished - first_token),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
