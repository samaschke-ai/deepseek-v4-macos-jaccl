#!/usr/bin/env python3
"""Install the official oMLX DeepSeek native artifacts into an editable runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ARTIFACTS = {
    "_ext.cpython-313-darwin.so": ("791738fb2d3a648cb123e86155fb29fe5c9a6631cbc8bfb76a11eb1a42bb6608", 0o755),
    "libomlx_glm_kernel_ops.dylib": ("e75ca7c7d2db504a2eb09add81239f1b15fb0c33d2e94320fe11a47d5527c36f", 0o755),
    "omlx_glm_kernels.metallib": ("ae3f15b73f834e3cddae48cc12d065a3b5ba8b21fdcc95dadd8012cc4486fd52", 0o644),
}
REQUIRED = (
    "deepseek_mxfp4_gather_qmm_blocks",
    "deepseek_mxfp4_gather_qmm_pair_blocks",
    "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
    "deepseek_v4_sparse_attention",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} STAGE_DIR DEST_DIR")
    stage, destination = map(Path, sys.argv[1:])
    destination.mkdir(parents=True, exist_ok=True)

    changed = False
    for name, (expected, mode) in ARTIFACTS.items():
        source = stage / name
        if digest(source) != expected:
            raise SystemExit(f"unexpected checksum for {source}")
        target = destination / name
        if target.is_file() and digest(target) == expected:
            continue
        temporary = destination / f".{name}.tmp"
        shutil.copyfile(source, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        changed = True

    from omlx.custom_kernels.glm_moe_dsa import fast

    missing = fast.missing_symbols(REQUIRED)
    receipt = {
        "changed": changed,
        "extension": str(fast._ext.__file__) if fast._ext else None,
        "native_symbols": fast.native_symbols(),
        "missing_required": missing,
    }
    print(json.dumps(receipt, indent=2))
    if not fast.is_native_available() or missing:
        raise SystemExit("DeepSeek native-kernel verification failed")


if __name__ == "__main__":
    main()
