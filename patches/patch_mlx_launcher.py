#!/usr/bin/env python3
"""Patch MLX's remote launcher to normalize pseudo-TTY CRLF pidfile output."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

OLD_PIDFILE = "            self._pidfile = pidfile\n"
NEW_PIDFILE = '            self._pidfile = pidfile.rstrip("\\r")\n'
OLD_LOG_LEVEL = '            cmd = f"ssh -tt -o LogLevel=QUIET {shlex.quote(host)} {shlex.quote(cmd)}"\n'
NEW_LOG_LEVEL = '            cmd = f"ssh -tt -o LogLevel=ERROR {shlex.quote(host)} {shlex.quote(cmd)}"\n'


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} MLX_LAUNCH_PY")
    target = Path(sys.argv[1])
    text = target.read_text()
    replacements = []
    for old, new, label in (
        (OLD_PIDFILE, NEW_PIDFILE, "pidfile_crlf"),
        (OLD_LOG_LEVEL, NEW_LOG_LEVEL, "ssh_error_logging"),
    ):
        if new in text:
            continue
        if text.count(old) != 1:
            raise SystemExit(f"refusing unexpected MLX launcher source for {label}: {target}")
        replacements.append((old, new, label))
    if not replacements:
        print(f"mlx_launcher_patch=present path={target}")
        return

    backup = target.with_name(target.name + ".pre-dsv4-crlf-patch")
    if not backup.exists():
        shutil.copy2(target, backup)

    updated = text
    for old, new, _label in replacements:
        updated = updated.replace(old, new)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    labels = ",".join(label for _old, _new, label in replacements)
    print(f"mlx_launcher_patch=installed changes={labels} path={target} backup={backup}")


if __name__ == "__main__":
    main()
