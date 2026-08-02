#!/usr/bin/env python3
"""Expose one stable OpenAI model ID for the distributed DeepSeek server."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TARGET = os.environ.get("DEEPSEEK_TARGET_BASE", "http://127.0.0.1:8888/v1").rstrip("/")
PUBLIC_MODEL = os.environ.get("DEEPSEEK_PUBLIC_MODEL", "deepseek-v4-flash")
BACKEND_MODEL = os.environ.get("DEEPSEEK_BACKEND_MODEL", "deepseek-v4-flash-0731")
MODEL_NAMESPACE = os.environ.get("DEEPSEEK_MODEL_NAMESPACE", "").strip("/")
TIMEOUT = float(os.environ.get("DEEPSEEK_ROUTER_TIMEOUT", "7200"))
HEARTBEAT_INTERVAL = float(os.environ.get("DEEPSEEK_STREAM_HEARTBEAT_INTERVAL", "30"))
READY_FILE = Path(os.environ.get("DEEPSEEK_READY_FILE", "/tmp/deepseek-v4/ready"))

app = FastAPI(title="DeepSeek V4 router")
_BACKGROUND_DRAINS: set[asyncio.Task[Any]] = set()

_INVOKE_RE = re.compile(
    r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</｜DSML｜invoke>',
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r'<｜DSML｜parameter\s+name="(?P<key>[^"]+)"\s+'
    r'string="(?P<string>true|false)"\s*>(?P<value>.*?)</｜DSML｜parameter>',
    re.DOTALL,
)
_DSML_START = "<｜DSML｜tool_calls>"
_DSML_END = "</｜DSML｜tool_calls>"


def _decode_dsml_value(raw: str, is_string: bool) -> Any:
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    if is_string:
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def _dsml_tool_calls(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, str) or "<｜DSML｜tool_calls>" not in content:
        return []
    calls = []
    for invoke in _INVOKE_RE.finditer(content):
        arguments = {}
        for parameter in _PARAM_RE.finditer(invoke.group("body")):
            arguments[parameter.group("key")] = _decode_dsml_value(
                parameter.group("value"), parameter.group("string") == "true"
            )
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": invoke.group("name"),
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    return calls


def _normalize_response(content: dict[str, Any]) -> dict[str, Any]:
    content["model"] = PUBLIC_MODEL
    for choice in content.get("choices") or []:
        message = choice.get("message") or {}
        calls = _dsml_tool_calls(message.get("content"))
        if calls:
            message["content"] = None
            message["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"
    return content


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _heartbeat_chunk(request_id: str, created: int) -> bytes:
    """Emit a valid empty OpenAI chunk so clients know slow prefill is alive."""
    return _sse(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": PUBLIC_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }
    )


class _DSMLStreamState:
    """Withhold only a possible DSML envelope while normal text keeps flowing."""

    def __init__(self) -> None:
        self.pending = ""
        self.tool_text = ""
        self.in_tool = False
        self.complete = False
        self.calls: list[dict[str, Any]] = []
        self.calls_emitted = False

    @staticmethod
    def _partial_start_len(text: str) -> int:
        maximum = min(len(text), len(_DSML_START) - 1)
        for size in range(maximum, 0, -1):
            if text.endswith(_DSML_START[:size]):
                return size
        return 0

    def feed(self, text: str) -> str:
        if not text or self.complete:
            return ""
        if self.in_tool:
            self.tool_text += text
            self._finish_tool_if_complete()
            return ""

        self.pending += text
        start = self.pending.find(_DSML_START)
        if start >= 0:
            visible = self.pending[:start]
            self.tool_text = self.pending[start:]
            self.pending = ""
            self.in_tool = True
            self._finish_tool_if_complete()
            return visible

        keep = self._partial_start_len(self.pending)
        if keep:
            visible = self.pending[:-keep]
            self.pending = self.pending[-keep:]
            return visible
        visible = self.pending
        self.pending = ""
        return visible

    def _finish_tool_if_complete(self) -> None:
        end = self.tool_text.find(_DSML_END)
        if end < 0:
            return
        self.tool_text = self.tool_text[: end + len(_DSML_END)]
        self.complete = True
        self.calls = _dsml_tool_calls(self.tool_text)

    def take_calls(self) -> list[dict[str, Any]]:
        if not self.calls or self.calls_emitted:
            return []
        self.calls_emitted = True
        return [{"index": index, **call} for index, call in enumerate(self.calls)]

    def finish_visible(self) -> str:
        if self.calls:
            self.pending = ""
            self.tool_text = ""
            return ""
        visible = self.pending
        if self.in_tool:
            # Preserve malformed output only after structured parsing has failed.
            visible += self.tool_text
        self.pending = ""
        self.tool_text = ""
        return visible


class _SSEStreamNormalizer:
    """Incrementally rewrite upstream SSE without exposing DSML control markup."""

    def __init__(self, normalize_tools: bool) -> None:
        self.normalize_tools = normalize_tools
        self.buffer = b""
        self.states: dict[int, _DSMLStreamState] = {}
        self.last_payload: dict[str, Any] | None = None
        self.done = False

    @staticmethod
    def _event_boundary(data: bytes) -> tuple[int, int] | None:
        boundaries = []
        for delimiter in (b"\n\n", b"\r\n\r\n"):
            index = data.find(delimiter)
            if index >= 0:
                boundaries.append((index, len(delimiter)))
        return min(boundaries) if boundaries else None

    @staticmethod
    def _data(event: bytes) -> bytes | None:
        lines = []
        for line in event.splitlines():
            if line.startswith(b"data:"):
                lines.append(line[5:].lstrip())
        return b"\n".join(lines) if lines else None

    @staticmethod
    def _choice_chunk(payload: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any]:
        chunk = {key: copy.deepcopy(value) for key, value in payload.items() if key != "choices"}
        chunk["choices"] = [choice]
        return chunk

    def _normalize_payload(self, payload: dict[str, Any]) -> list[bytes]:
        payload["model"] = PUBLIC_MODEL
        self.last_payload = copy.deepcopy(payload)
        if not self.normalize_tools:
            return [_sse(payload)]

        before: list[bytes] = []
        emit_choices = []
        for choice in payload.get("choices") or []:
            index = int(choice.get("index", 0))
            state = self.states.setdefault(index, _DSMLStreamState())
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str):
                visible = state.feed(text)
                if visible:
                    delta["content"] = visible
                else:
                    delta.pop("content", None)

            calls = state.take_calls()
            if calls:
                delta["tool_calls"] = calls

            if choice.get("finish_reason") is not None:
                tail = state.finish_visible()
                if tail:
                    before.append(
                        _sse(
                            self._choice_chunk(
                                payload,
                                {"index": index, "delta": {"content": tail}, "finish_reason": None},
                            )
                        )
                    )
                if state.calls:
                    choice["finish_reason"] = "tool_calls"
                    choice["delta"] = {}

            choice["delta"] = delta if choice.get("finish_reason") is None else choice.get("delta", {})
            if delta or choice.get("finish_reason") is not None:
                emit_choices.append(choice)

        if emit_choices:
            payload["choices"] = emit_choices
            before.append(_sse(payload))
        elif not payload.get("choices"):
            before.append(_sse(payload))
        return before

    def _flush_states(self) -> list[bytes]:
        if self.last_payload is None:
            return []
        output = []
        for index, state in self.states.items():
            tail = state.finish_visible()
            if tail:
                output.append(
                    _sse(
                        self._choice_chunk(
                            self.last_payload,
                            {"index": index, "delta": {"content": tail}, "finish_reason": None},
                        )
                    )
                )
        return output

    def _normalize_event(self, event: bytes, delimiter: bytes) -> list[bytes]:
        data = self._data(event)
        if data is None:
            return [event + delimiter]
        if data == b"[DONE]":
            output = self._flush_states()
            self.done = True
            output.append(b"data: [DONE]\n\n")
            return output
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return [event + delimiter]
        if not isinstance(payload, dict):
            return [event + delimiter]
        return self._normalize_payload(payload)

    def feed(self, data: bytes) -> list[bytes]:
        if self.done:
            return []
        self.buffer += data
        output = []
        while (boundary := self._event_boundary(self.buffer)) is not None:
            index, size = boundary
            event = self.buffer[:index]
            delimiter = self.buffer[index : index + size]
            self.buffer = self.buffer[index + size :]
            output.extend(self._normalize_event(event, delimiter))
        return output

    def finish(self) -> list[bytes]:
        if self.done:
            self.buffer = b""
            return []
        output = self._flush_states()
        if self.buffer:
            output.append(self.buffer)
            self.buffer = b""
        return output


def _retain_background(coro: Any) -> None:
    """Keep a disconnected distributed request alive until all ranks finish."""
    task = asyncio.create_task(coro)
    _BACKGROUND_DRAINS.add(task)
    task.add_done_callback(_BACKGROUND_DRAINS.discard)


async def _drain_stream_request(
    iterator: Any,
    next_chunk: asyncio.Task[bytes] | None,
    response: httpx.Response,
    client: httpx.AsyncClient,
) -> None:
    try:
        if next_chunk is not None:
            try:
                await next_chunk
            except StopAsyncIteration:
                return
        async for _ in iterator:
            pass
    except Exception:
        pass
    finally:
        await response.aclose()
        await client.aclose()


async def _finish_pending_stream_request(
    send_task: asyncio.Task[httpx.Response],
    client: httpx.AsyncClient,
) -> None:
    """Finish a request abandoned before upstream response headers arrived."""
    try:
        response = await send_task
    except Exception:
        await client.aclose()
        return
    iterator = response.aiter_bytes().__aiter__()
    await _drain_stream_request(iterator, None, response, client)


async def upstream_model() -> str:
    if not READY_FILE.is_file():
        raise RuntimeError("DeepSeek is warming up")
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        response = await client.get(f"{TARGET}/models")
        response.raise_for_status()
        models = response.json().get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError("DeepSeek upstream returned no model ID")
    return str(models[0]["id"])


@app.get("/v1/models")
async def models() -> JSONResponse:
    try:
        await upstream_model()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"DeepSeek upstream unhealthy: {type(exc).__name__}: {exc}"}},
        )
    return JSONResponse(
        content={
            "object": "list",
            "data": [{"id": PUBLIC_MODEL, "object": "model", "owned_by": "deepseek-ai"}],
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    try:
        native_model = await upstream_model()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "target": TARGET, "error": f"{type(exc).__name__}: {exc}"},
        )
    return JSONResponse(
        content={"ok": True, "target": TARGET, "public_model": PUBLIC_MODEL, "native_model": native_model}
    )


def valid_model(value: Any) -> bool:
    requested = str(value or "")
    if MODEL_NAMESPACE:
        requested = requested.removeprefix(f"{MODEL_NAMESPACE}/")
    return requested in {PUBLIC_MODEL, BACKEND_MODEL}


async def proxy(request: Request, endpoint: str):
    payload = await request.json()
    if not valid_model(payload.get("model")):
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"unknown model: {payload.get('model')}"}},
        )
    try:
        # Every rank maps this shared logical key to its own local symlink.
        # Forwarding rank 0's resolved /Volumes path would trigger an asymmetric
        # model reload because the two Macs use different physical volumes.
        await upstream_model()
        payload["model"] = "default_model"
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"DeepSeek upstream unavailable: {type(exc).__name__}: {exc}"}},
        )

    timeout = httpx.Timeout(TIMEOUT, connect=10.0)
    requested_stream = bool(payload.get("stream"))
    normalize_tool_stream = (
        requested_stream and endpoint == "chat/completions" and bool(payload.get("tools"))
    )

    if not requested_stream:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            try:
                response = await client.post(f"{TARGET}/{endpoint}", json=payload)
            except Exception as exc:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": f"DeepSeek request failed: {type(exc).__name__}: {exc}"}},
                )
        try:
            content = response.json() if response.content else {}
        except Exception:
            content = {"error": {"message": response.text[:2000]}}
        if response.is_success and isinstance(content, dict):
            content = _normalize_response(content)
        return JSONResponse(status_code=response.status_code, content=content)

    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    upstream_request = client.build_request("POST", f"{TARGET}/{endpoint}", json=payload)

    async def chunks():
        response: httpx.Response | None = None
        iterator: Any = None
        next_chunk: asyncio.Task[bytes] | None = None
        send_task = asyncio.create_task(client.send(upstream_request, stream=True))
        normalizer = _SSEStreamNormalizer(normalize_tool_stream)
        request_id = f"chatcmpl-heartbeat-{uuid.uuid4().hex}"
        created = int(time.time())
        handed_off = False
        try:
            # Start the downstream SSE response before oMLX finishes prefill and
            # sends its response headers. Otherwise no heartbeat can reach the
            # client during a long prompt.
            yield _heartbeat_chunk(request_id, created)
            while not send_task.done():
                done, _ = await asyncio.wait({send_task}, timeout=HEARTBEAT_INTERVAL)
                if not done:
                    yield _heartbeat_chunk(request_id, created)
            try:
                response = send_task.result()
            except Exception as exc:
                yield _sse(
                    {"error": {"message": f"DeepSeek stream failed: {type(exc).__name__}: {exc}"}}
                )
                yield b"data: [DONE]\n\n"
                return
            if not response.is_success:
                detail = (await response.aread()).decode(errors="replace")[:2000]
                yield _sse(
                    {
                        "error": {
                            "message": f"DeepSeek upstream returned HTTP {response.status_code}: {detail}"
                        }
                    }
                )
                yield b"data: [DONE]\n\n"
                return

            iterator = response.aiter_bytes().__aiter__()
            next_chunk = asyncio.create_task(iterator.__anext__())
            while True:
                done, _ = await asyncio.wait({next_chunk}, timeout=HEARTBEAT_INTERVAL)
                if not done:
                    yield _heartbeat_chunk(request_id, created)
                    continue
                try:
                    for chunk in normalizer.feed(next_chunk.result()):
                        yield chunk
                except StopAsyncIteration:
                    for chunk in normalizer.finish():
                        yield chunk
                    break
                next_chunk = asyncio.create_task(iterator.__anext__())
        except (asyncio.CancelledError, GeneratorExit):
            if response is None:
                _retain_background(_finish_pending_stream_request(send_task, client))
            else:
                _retain_background(_drain_stream_request(iterator, next_chunk, response, client))
            handed_off = True
            raise
        finally:
            if not handed_off:
                if next_chunk is not None and not next_chunk.done():
                    next_chunk.cancel()
                if not send_task.done():
                    send_task.cancel()
                if response is not None:
                    await response.aclose()
                await client.aclose()

    return StreamingResponse(chunks(), status_code=200, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy(request, "chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy(request, "completions")
