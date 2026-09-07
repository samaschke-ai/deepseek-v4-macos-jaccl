#!/usr/bin/env python3
"""Skip non-local DeepSeek EP routes in native prefill MoE kernels."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PATCHES = (
    (
        """def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order
""",
        """def _gather_sort(x, indices, route_mask=None):
    *_, M = indices.shape
    indices = indices.flatten()
    if route_mask is None:
        order = mx.argsort(indices)
    else:
        route_mask = route_mask.flatten()
        # Keep active routes first within each expert. The native block builder
        # can then omit the inactive suffix without changing tensor shapes.
        sort_key = indices * 2 + (~route_mask).astype(indices.dtype)
        order = mx.argsort(sort_key)
    inv_order = mx.argsort(order)
    result = (x.flatten(0, -3)[order // M], indices[order], inv_order)
    if route_mask is None:
        return result
    return (*result, route_mask[order])
""",
    ),
    (
        """        const int end = lo;

        for (int row = start; row < end; row += BM) {
""",
        """        int end = lo;
        // Active routes are sorted before inactive routes for each expert.
        // Trim the masked suffix so no native GEMM block is dispatched for
        // an expert owned by the other EP rank.
        while (end > start && route_mask[end - 1] == 0) {
            --end;
        }

        for (int row = start; row < end; row += BM) {
""",
    ),
    (
        """        input_names=[\"indices\"],
""",
        """        input_names=[\"indices\", \"route_mask\"],
""",
    ),
    (
        """def _build_mxfp4_blocks(indices: mx.array, num_experts: int, bm: int):
    indices = indices.astype(mx.int32)
    max_blocks = (indices.size + bm - 1) // bm + num_experts
    builder = _mxfp4_block_builder(num_experts, bm)
    return builder(
        inputs=[indices],
""",
        """def _build_mxfp4_blocks(
    indices: mx.array,
    num_experts: int,
    bm: int,
    route_mask: mx.array | None = None,
):
    indices = indices.astype(mx.int32)
    if route_mask is None:
        route_mask = mx.ones(indices.shape, dtype=mx.bool_)
    else:
        route_mask = route_mask.astype(mx.bool_)
    max_blocks = (indices.size + bm - 1) // bm + num_experts
    builder = _mxfp4_block_builder(num_experts, bm)
    return builder(
        inputs=[indices, route_mask],
""",
    ),
    (
        """        original_dtype = x.dtype

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
""",
        """        original_dtype = x.dtype

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        route_mask = scores != 0 if scores is not None else None
        if do_sort:
            if route_mask is None:
                x, idx, inv_order = _gather_sort(x, indices)
            else:
                x, idx, inv_order, route_mask = _gather_sort(
                    x, indices, route_mask
                )
""",
        """        route_mask = scores != 0 if scores is not None else None
        if do_sort:
            if route_mask is None:
""",
    ),
    (
        """                block_meta, block_count = _build_mxfp4_blocks(
                    idx,
                    self.up_proj.num_experts,
                    block_bm,
                )
""",
        """                block_meta, block_count = _build_mxfp4_blocks(
                    idx,
                    self.up_proj.num_experts,
                    block_bm,
                    route_mask=route_mask,
                )
""",
    ),
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} SWITCH_LAYERS_PY")
    path = Path(sys.argv[1]).resolve()
    source = path.read_text()
    changed = False
    for old, new, *present_markers in PATCHES:
        if new in source or any(marker in source for marker in present_markers):
            continue
        if source.count(old) != 1:
            raise SystemExit(
                f"refusing unexpected oMLX source near: {old.splitlines()[0]!r}"
            )
        source = source.replace(old, new)
        changed = True

    backup = path.with_suffix(path.suffix + ".pre-dsv4-ep-route-mask")
    if changed:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(source)
        print(f"omlx_ep_route_mask=installed path={path}")
    else:
        print(f"omlx_ep_route_mask=present path={path}")


if __name__ == "__main__":
    main()
