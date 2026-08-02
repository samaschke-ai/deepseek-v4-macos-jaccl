#!/usr/bin/env python3
"""Focused tests for incremental DeepSeek DSML stream normalization."""

from __future__ import annotations

import json
import unittest

from openai_router import PUBLIC_MODEL, _SSEStreamNormalizer


def event(payload: dict, delimiter: bytes = b"\n\n") -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + delimiter


def payloads(chunks: list[bytes]) -> list[dict]:
    decoded = []
    for chunk in chunks:
        for block in chunk.replace(b"\r\n", b"\n").split(b"\n\n"):
            if not block.startswith(b"data: ") or block == b"data: [DONE]":
                continue
            decoded.append(json.loads(block[6:]))
    return decoded


class SSEStreamNormalizerTests(unittest.TestCase):
    def test_normal_text_streams_and_public_model_is_rewritten(self) -> None:
        normalizer = _SSEStreamNormalizer(normalize_tools=True)
        first = {
            "id": "chatcmpl-1",
            "model": "default_model",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        output = normalizer.feed(event(first))
        self.assertEqual(payloads(output)[0]["model"], PUBLIC_MODEL)
        self.assertEqual(payloads(output)[0]["choices"][0]["delta"]["content"], "Hello")

    def test_split_dsml_is_hidden_and_emitted_as_tool_calls(self) -> None:
        normalizer = _SSEStreamNormalizer(normalize_tools=True)
        parts = [
            {"reasoning": "I will check."},
            {"content": "\n\n<"},
            {"content": "｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"},
            {"content": "<｜DSML｜parameter name=\"city\" string=\"true\">Paris"},
            {"content": "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"},
        ]
        output: list[bytes] = []
        for part in parts:
            output.extend(
                normalizer.feed(
                    event(
                        {
                            "id": "chatcmpl-2",
                            "model": "default_model",
                            "choices": [{"index": 0, "delta": part, "finish_reason": None}],
                        }
                    )
                )
            )
        output.extend(
            normalizer.feed(
                event(
                    {
                        "id": "chatcmpl-2",
                        "model": "default_model",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
            )
        )
        output.extend(normalizer.feed(b"data: [DONE]\n\n"))

        wire = b"".join(output).decode()
        self.assertNotIn("DSML", wire)
        parsed = payloads(output)
        self.assertEqual(parsed[0]["choices"][0]["delta"]["reasoning"], "I will check.")
        tool_delta = next(
            item["choices"][0]["delta"]
            for item in parsed
            if item.get("choices") and item["choices"][0].get("delta", {}).get("tool_calls")
        )
        call = tool_delta["tool_calls"][0]
        self.assertEqual(call["index"], 0)
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(parsed[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(b"data: [DONE]\n\n" in output)

    def test_incomplete_dsml_is_recovered_only_after_parse_failure(self) -> None:
        normalizer = _SSEStreamNormalizer(normalize_tools=True)
        start = {
            "id": "chatcmpl-3",
            "model": "default_model",
            "choices": [{"index": 0, "delta": {"content": "<｜DSML｜tool_calls>broken"}, "finish_reason": None}],
        }
        final = {
            "id": "chatcmpl-3",
            "model": "default_model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.assertEqual(normalizer.feed(event(start)), [])
        output = normalizer.feed(event(final))
        self.assertIn("broken", b"".join(output).decode())
        self.assertEqual(payloads(output)[-1]["choices"][0]["finish_reason"], "stop")

    def test_done_flushes_a_partial_non_dsml_prefix(self) -> None:
        normalizer = _SSEStreamNormalizer(normalize_tools=True)
        partial = {
            "id": "chatcmpl-partial",
            "model": "default_model",
            "choices": [{"index": 0, "delta": {"content": "literal <"}, "finish_reason": None}],
        }
        output = normalizer.feed(event(partial))
        output.extend(normalizer.feed(b"data: [DONE]\n\n"))
        text = "".join(
            (choice.get("delta") or {}).get("content", "")
            for item in payloads(output)
            for choice in item.get("choices", [])
        )
        self.assertEqual(text, "literal <")

    def test_crlf_and_arbitrary_byte_splits(self) -> None:
        normalizer = _SSEStreamNormalizer(normalize_tools=False)
        raw = event(
            {
                "id": "chatcmpl-4",
                "model": "default_model",
                "choices": [{"index": 0, "delta": {"content": "Grüße"}, "finish_reason": None}],
            },
            b"\r\n\r\n",
        )
        output = []
        for byte in raw:
            output.extend(normalizer.feed(bytes([byte])))
        parsed = payloads(output)
        self.assertEqual(parsed[0]["choices"][0]["delta"]["content"], "Grüße")
        self.assertEqual(parsed[0]["model"], PUBLIC_MODEL)


if __name__ == "__main__":
    unittest.main()
