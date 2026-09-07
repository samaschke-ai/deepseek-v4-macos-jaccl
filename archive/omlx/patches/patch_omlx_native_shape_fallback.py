#!/usr/bin/env python3
"""Make oMLX DeepSeek native MoE kernels fall back on unsupported shapes."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PATCHES = (
    (
        """        if native_kind is not None:
            if block_plan is None:
                block_bm, block_variant = _mxfp4_block_config(indices.size)
                block_meta, block_count = _build_mxfp4_blocks(
                    indices,
                    self.num_experts,
                    block_bm,
                )
            else:
                block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                    block_plan
                )
            if native_kind == "mxfp4":
                x = glm_fast.deepseek_mxfp4_gather_qmm_blocks(
                    x,
                    self["weight"],
                    self["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
            else:
                x = glm_fast.deepseek_affine_gather_qmm_blocks(
                    x,
                    self["weight"],
                    self["scales"],
                    self["biases"],
                    block_meta,
                    block_count,
                    self.group_size,
                    self.bits,
                    block_variant,
                )
        else:
            x = mx.gather_qmm(
""",
        """        if native_kind is not None:
            try:
                if block_plan is None:
                    block_bm, block_variant = _mxfp4_block_config(indices.size)
                    block_meta, block_count = _build_mxfp4_blocks(
                        indices,
                        self.num_experts,
                        block_bm,
                    )
                else:
                    block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                        block_plan
                    )
                if native_kind == "mxfp4":
                    x = glm_fast.deepseek_mxfp4_gather_qmm_blocks(
                        x,
                        self["weight"],
                        self["scales"],
                        block_meta,
                        block_count,
                        block_variant,
                    )
                else:
                    x = glm_fast.deepseek_affine_gather_qmm_blocks(
                        x,
                        self["weight"],
                        self["scales"],
                        self["biases"],
                        block_meta,
                        block_count,
                        self.group_size,
                        self.bits,
                        block_variant,
                    )
            except ValueError as exc:
                if "unsupported shape" not in str(exc):
                    raise
                native_kind = None
        if native_kind is None:
            x = mx.gather_qmm(
""",
    ),
    (
        """        if use_pair_proj:
            block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                block_plan
            )
            if glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_concat_blocks"):
                x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                    x,
                    self.up_proj["weight"],
                    self.up_proj["scales"],
                    self.gate_proj["weight"],
                    self.gate_proj["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
                hidden_dims = self.up_proj.output_dims
                x_up = x_pair[..., :hidden_dims]
                x_gate = x_pair[..., hidden_dims:]
            else:
                x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_blocks(
                    x,
                    self.up_proj["weight"],
                    self.up_proj["scales"],
                    self.gate_proj["weight"],
                    self.gate_proj["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
                x_up = x_pair[0]
                x_gate = x_pair[1]
""",
        """        if use_pair_proj:
            block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                block_plan
            )
            try:
                if glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_concat_blocks"):
                    x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                        x,
                        self.up_proj["weight"],
                        self.up_proj["scales"],
                        self.gate_proj["weight"],
                        self.gate_proj["scales"],
                        block_meta,
                        block_count,
                        block_variant,
                    )
                    hidden_dims = self.up_proj.output_dims
                    x_up = x_pair[..., :hidden_dims]
                    x_gate = x_pair[..., hidden_dims:]
                else:
                    x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_blocks(
                        x,
                        self.up_proj["weight"],
                        self.up_proj["scales"],
                        self.gate_proj["weight"],
                        self.gate_proj["scales"],
                        block_meta,
                        block_count,
                        block_variant,
                    )
                    x_up = x_pair[0]
                    x_gate = x_pair[1]
            except ValueError as exc:
                if "unsupported shape" not in str(exc):
                    raise
                # Cancellation and ragged-prefill compaction can produce route
                # counts outside the native kernel's compiled shape set. Keep
                # correctness by using MLX's general gather_qmm for that call.
                x_up = self.up_proj(x, idx, sorted_indices=False)
                x_gate = self.gate_proj(x, idx, sorted_indices=False)
""",
        "Cancellation and ragged-prefill compaction can produce route",
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
            raise SystemExit(f"refusing unexpected oMLX source near: {old.splitlines()[0]!r}")
        source = source.replace(old, new)
        changed = True

    backup = path.with_suffix(path.suffix + ".pre-dsv4-shape-fallback")
    if changed:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(source)
        print(f"omlx_native_shape_fallback=installed path={path}")
    else:
        print(f"omlx_native_shape_fallback=present path={path}")


if __name__ == "__main__":
    main()
