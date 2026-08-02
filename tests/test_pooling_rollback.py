#!/usr/bin/env python3
"""Exhaustive ratio-4 rollback regression test for patched oMLX caches."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import mlx.core as mx


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("dsv4_cache_patch_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_array(values):
    return mx.array(values, dtype=mx.float32).reshape(1, -1, 1)


def compress_and_update(cache, values, *, batch: bool) -> None:
    values = list(values)
    kv = token_array(values)
    gate = token_array([value + 1000 for value in values])
    if batch:
        cache.prepare(lengths=[len(values)])
        ready_kv, ready_gate, _ = cache.accumulate_windows(
            kv, gate, mx.array([cache._processed[0]])
        )
    else:
        offset = cache.offset * cache.ratio + cache.remainder
        ready_kv, ready_gate, _ = cache.accumulate_windows(kv, gate, offset)

    count = ready_kv.shape[1] // cache.ratio
    if count:
        window_kv = mx.unflatten(ready_kv, 1, (-1, cache.ratio))
        window_gate = mx.unflatten(ready_gate, 1, (-1, cache.ratio))
        old_kv, old_gate = cache.prev_for_prepend()
        dropped = 0
        if old_kv is not None:
            extended_kv = mx.concatenate([old_kv, window_kv], axis=1)
            extended_gate = mx.concatenate([old_gate, window_gate], axis=1)
            dropped = 1
        else:
            extended_kv, extended_gate = window_kv, window_gate
        current = extended_kv[:, dropped:].sum(axis=(2, 3))
        previous = (
            mx.zeros_like(current)
            if dropped == 0
            else extended_kv[:, :-1].sum(axis=(2, 3))
        )
        pooled = (current + previous)[..., None]
        cache.store_prev(extended_kv, extended_gate, dropped)
    else:
        pooled = mx.zeros((1, 0, 1), dtype=mx.float32)
    cache.update_and_fetch(pooled)
    if batch:
        cache.finalize()
    mx.eval(
        *[
            value
            for value in (
                cache.buf_kv,
                cache.buf_gate,
                cache.pooled,
                cache.prev_win_kv,
                cache.prev_win_gate,
            )
            if isinstance(value, mx.array)
        ]
    )


def normalized_state(cache, *, batch: bool):
    remainder = cache.remainder[0] if batch else cache.remainder
    pool_length = (
        cache._pool_lengths[0]
        if batch
        else (0 if cache.pooled is None else cache.pooled.shape[1])
    )
    pooled = [] if cache.pooled is None else cache.pooled[0, :pool_length].tolist()
    buffer = (
        []
        if cache.buf_kv is None or remainder == 0
        else cache.buf_kv[0, :remainder].tolist()
    )
    previous = [] if cache.prev_win_kv is None else cache.prev_win_kv[0].tolist()
    state = (remainder, pool_length, pooled, buffer, previous)
    if batch:
        state += (cache._processed[0], cache._prev_valid[0])
    return state


def build(cache_class, *, batch: bool, prefix, verify=None):
    cache = cache_class(4, [0]) if batch else cache_class(4)
    compress_and_update(cache, prefix, batch=batch)
    if verify is not None:
        compress_and_update(cache, verify, batch=batch)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_extras", type=Path)
    args = parser.parse_args()
    module = load_module(args.cache_extras.resolve())

    verify = list(range(100, 106))
    checks = 0
    for batch, cache_class in (
        (False, module.PoolingCache),
        (True, module.BatchPoolingCache),
    ):
        for remainder in range(4):
            prefix = list(range(10, 14 + remainder))
            for accepted in range(5):
                actual = build(cache_class, batch=batch, prefix=prefix, verify=verify)
                expected = build(
                    cache_class,
                    batch=batch,
                    prefix=prefix,
                    verify=verify[: accepted + 1],
                )
                trim_count = 5 - accepted
                assert actual._can_undo(trim_count), (
                    batch,
                    remainder,
                    accepted,
                    "cannot undo",
                )
                assert actual.trim(trim_count) == trim_count
                assert normalized_state(actual, batch=batch) == normalized_state(
                    expected, batch=batch
                ), (batch, remainder, accepted)
                checks += 1
    print(f"POOLING_ROLLBACK_MATRIX_OK checks={checks}")


if __name__ == "__main__":
    main()
