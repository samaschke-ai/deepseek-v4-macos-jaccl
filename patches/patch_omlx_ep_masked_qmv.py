#!/usr/bin/env python3
"""Enable the EP-only masked small-M MXFP4 QMV kernel in editable oMLX."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path


FAST_PATCHES = (
    (
        '    "deepseek_mxfp4_gather_qmm_pair_concat_blocks",\n'
        '    "deepseek_mxfp4_gather_qmm_expert",\n',
        '    "deepseek_mxfp4_gather_qmm_pair_concat_blocks",\n'
        '    "deepseek_mxfp4_gather_qmm_masked_row",\n'
        '    "deepseek_mxfp4_gather_qmm_expert",\n',
    ),
    (
        "def deepseek_mxfp4_gather_qmm_expert(\n",
        '''def deepseek_mxfp4_gather_qmm_masked_row(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    indices: mx.array,
    route_mask: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(
        _ext, "deepseek_mxfp4_gather_qmm_masked_row"
    ):
        return _ext.deepseek_mxfp4_gather_qmm_masked_row(
            x,
            weight,
            scales,
            indices,
            route_mask,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "deepseek_mxfp4_gather_qmm_masked_row native kernel is unavailable"
    )


def deepseek_mxfp4_gather_qmm_expert(
''',
    ),
)

SWITCH_PATCHES = (
    (
        "def _should_sort_experts(x: mx.array, indices: mx.array) -> bool:\n",
        '''def _masked_ep_qmv(proj, x, indices, route_mask):
    flat_x = x.flatten(0, -3)
    flat_indices = indices.flatten().astype(mx.int32)
    flat_mask = route_mask.flatten().astype(mx.bool_)
    out = glm_fast.deepseek_mxfp4_gather_qmm_masked_row(
        flat_x,
        proj["weight"],
        proj["scales"],
        flat_indices,
        flat_mask,
        0,
    )
    return out.reshape(*indices.shape, 1, proj.output_dims)


def _should_sort_experts(x: mx.array, indices: mx.array) -> bool:
''',
    ),
    (
        '''        block_plan = None
        native_kinds = None
        use_f16_affine_moe = False
        projections = (self.up_proj, self.gate_proj, self.down_proj)
''',
        '''        block_plan = None
        native_kinds = None
        use_f16_affine_moe = False
        projections = (self.up_proj, self.gate_proj, self.down_proj)
        use_masked_ep_qmv = (
            not do_sort
            and route_mask is not None
            and getattr(self, "_ep_masked_qmv", False)
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_masked_row")
            and all(
                isinstance(p, QuantizedSwitchLinear)
                and p.group_size == 32
                and p.bits == 4
                and p.mode == "mxfp4"
                and p.get("biases") is None
                and "bias" not in p
                and p["weight"].dtype == mx.uint32
                and p["scales"].dtype == mx.uint8
                and p.input_dims % 512 == 0
                and p.output_dims % 8 == 0
                for p in projections
            )
        )
''',
    ),
    (
        "        if use_pair_proj:\n",
        '''        if use_masked_ep_qmv:
            x_up = _masked_ep_qmv(self.up_proj, x, idx, route_mask)
            x_gate = _masked_ep_qmv(self.gate_proj, x, idx, route_mask)
        elif use_pair_proj:
''',
    ),
    (
        '''        if getattr(self.down_proj, "_tp_fp32_partial", False):
            x = x.astype(mx.float32)
        x = self.down_proj(
            x,
            idx,
            sorted_indices=do_sort,
            block_plan=block_plan,
        )
''',
        '''        if getattr(self.down_proj, "_tp_fp32_partial", False):
            x = x.astype(mx.float32)
        if use_masked_ep_qmv:
            x = _masked_ep_qmv(self.down_proj, x, idx, route_mask)
        else:
            x = self.down_proj(
                x,
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
            )
''',
    ),
)


def apply(path: Path, patches: tuple[tuple[str, str], ...]) -> str:
    text = path.read_text()
    changed = False
    for old, new in patches:
        if new in text:
            continue
        if text.count(old) != 1:
            raise SystemExit(
                f"patch anchor count for {path}: expected 1, got {text.count(old)}"
            )
        text = text.replace(old, new)
        changed = True
    if changed:
        path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    return "installed" if changed else "present"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} FAST_PY SWITCH_LAYERS_PY")
    fast_path, switch_path = map(Path, sys.argv[1:])
    fast_state = apply(fast_path, FAST_PATCHES)
    switch_state = apply(switch_path, SWITCH_PATCHES)
    print(f"omlx_ep_masked_qmv={fast_state}/{switch_state}")


if __name__ == "__main__":
    main()
