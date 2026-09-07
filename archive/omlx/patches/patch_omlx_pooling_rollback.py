#!/usr/bin/env python3
"""Enable exact DeepSeek DSpark rollback across PoolingCache window seams."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

CACHE_PATCHES = (
    (
        "class PoolingCache(_BaseCache):\n",
        "class PoolingCache(_BaseCache):\n    _omlx_arbitrary_rollback = True\n",
    ),
    (
        """                self.prev_win_kv,
                self.prev_win_gate,
            )
        else:
            self._undo = None

        # Prompt mode
""",
        """                self.prev_win_kv,
                self.prev_win_gate,
                None,  # compressed outputs from update_and_fetch
            )
        else:
            self._undo = None

        # Prompt mode
""",
    ),
    (
        """    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
""",
        """    def update_and_fetch(self, px: mx.array):
        if self._undo is not None:
            self._undo = (*self._undo[:-1], px)
        if px.shape[1] == 0:
""",
    ),
    (
        """    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[4].shape[1] - n
        # The replayed confirmed prefix must stay inside the buffer (the
        # original per-token inputs of a completed window are gone, so a
        # replay that pools again cannot be reconstructed).
        return k >= 0 and undo[2] + k < self.ratio

    def trim(self, n):
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        buf_kv, buf_gate, rem_prev, pooled_prev, kv, gate, prev_kv, prev_gate = (
            self._undo
        )
        self._undo = None
        k = kv.shape[1] - n
        self.pooled = pooled_prev
        self.remainder = rem_prev
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        if buf_kv is not None:
            self.buf_kv[:, :rem_prev] = buf_kv
            self.buf_gate[:, :rem_prev] = buf_gate
        if k > 0:
            # Replay the confirmed prefix; _can_undo guarantees it stays in
            # the buffer, so no window is recompressed.
            self.accumulate_windows(kv[:, :k], gate[:, :k], 0)
            self._undo = None
        return n
""",
        """    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[4].shape[1] - n
        # update_and_fetch records the already-compressed outputs for this
        # verify window. Any retained prefix can therefore restore completed
        # pool windows without rerunning model projections.
        return k >= 0 and len(undo) == 9 and undo[8] is not None

    def trim(self, n):
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        (
            buf_kv,
            buf_gate,
            rem_prev,
            pooled_prev,
            kv,
            gate,
            prev_kv,
            prev_gate,
            pooled_update,
        ) = self._undo
        self._undo = None
        k = kv.shape[1] - n
        self.pooled = pooled_prev
        self.remainder = rem_prev
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        if buf_kv is not None:
            self.buf_kv[:, :rem_prev] = buf_kv
            self.buf_gate[:, :rem_prev] = buf_gate
        if k > 0:
            ready_kv, ready_gate, _ = self.accumulate_windows(
                kv[:, :k], gate[:, :k], 0
            )
            new_windows = ready_kv.shape[1] // self.ratio
            if new_windows:
                kept = pooled_update[:, :new_windows]
                self.pooled = (
                    kept
                    if pooled_prev is None
                    else mx.concatenate([pooled_prev, kept], axis=1)
                )
                if self.ratio == 4:
                    win_kv = mx.unflatten(ready_kv, 1, (-1, self.ratio))
                    win_gate = mx.unflatten(ready_gate, 1, (-1, self.ratio))
                    old_kv, old_gate = self.prev_for_prepend()
                    dropped = 0
                    if old_kv is not None:
                        win_kv = mx.concatenate([old_kv, win_kv], axis=1)
                        win_gate = mx.concatenate([old_gate, win_gate], axis=1)
                        dropped = 1
                    self.store_prev(win_kv, win_gate, dropped)
            self._undo = None
        return n
""",
    ),
    (
        "class BatchPoolingCache(_BaseCache):\n",
        "class BatchPoolingCache(_BaseCache):\n    _omlx_arbitrary_rollback = True\n",
    ),
    (
        """                self.prev_win_gate,
                list(self._prev_valid),
            )
        else:
            self._undo = None
""",
        """                self.prev_win_gate,
                list(self._prev_valid),
                None,  # compressed outputs from update_and_fetch
            )
        else:
            self._undo = None
""",
    ),
    (
        """    def update_and_fetch(self, px: mx.array):
        B, N, D = px.shape
""",
        """    def update_and_fetch(self, px: mx.array):
        if self._undo is not None:
            self._undo = (*self._undo[:-1], px)
        B, N, D = px.shape
""",
    ),
    (
        """    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[5].shape[1] - n
        # The replayed confirmed prefix must stay inside the buffer for
        # every row (a replay that pools again cannot be reconstructed).
        return k >= 0 and all(r + k < self.ratio for r in undo[2])

    def trim(self, n):
        if n <= min(self.remainder):
            for i in range(len(self.remainder)):
                self.remainder[i] -= n
                self._processed[i] -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        (
            buf_kv,
            buf_gate,
            remainder,
            pool_lengths,
            processed,
            kv,
            gate,
            prev_kv,
            prev_gate,
            prev_valid,
        ) = self._undo
        self._undo = None
        k = kv.shape[1] - n
        # The undo path only triggers when some row completed a window,
        # which rebinds self.buf_* to fresh arrays — the stashed objects
        # still hold the pre-update contents. The pooled tensor keeps any
        # extra written rows; restoring _pool_lengths masks them out.
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self.remainder = list(remainder)
        self._pool_lengths = list(pool_lengths)
        self._processed = list(processed)
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        self._prev_valid = list(prev_valid)
        if k > 0:
            # Replay the confirmed prefix; _can_undo guarantees it stays in
            # the buffer, so no window is recompressed.
            self.accumulate_windows(kv[:, :k], gate[:, :k], 0)
            self._undo = None
        return n
""",
        """    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[5].shape[1] - n
        return k >= 0 and len(undo) == 11 and undo[10] is not None

    def trim(self, n):
        if n <= min(self.remainder):
            for i in range(len(self.remainder)):
                self.remainder[i] -= n
                self._processed[i] -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        (
            buf_kv,
            buf_gate,
            remainder,
            pool_lengths,
            processed,
            kv,
            gate,
            prev_kv,
            prev_gate,
            prev_valid,
            pooled_update,
        ) = self._undo
        self._undo = None
        k = kv.shape[1] - n
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self.remainder = list(remainder)
        self._pool_lengths = list(pool_lengths)
        self._processed = list(processed)
        self.prev_win_kv = prev_kv
        self.prev_win_gate = prev_gate
        self._prev_valid = list(prev_valid)
        if k > 0:
            ready_kv, ready_gate, _ = self.accumulate_windows(
                kv[:, :k], gate[:, :k], 0
            )
            new_counts = [
                (self._processed[i] - self.remainder[i]) // self.ratio
                - self._pool_lengths[i]
                for i in range(len(self.remainder))
            ]
            max_new = max(new_counts)
            if max_new:
                self.update_and_fetch(pooled_update[:, :max_new])
                if self.ratio == 4:
                    win_kv = mx.unflatten(ready_kv, 1, (-1, self.ratio))
                    win_gate = mx.unflatten(ready_gate, 1, (-1, self.ratio))
                    old_kv, old_gate = self.prev_for_prepend()
                    dropped = 0
                    if old_kv is not None:
                        win_kv = mx.concatenate([old_kv, win_kv], axis=1)
                        win_gate = mx.concatenate([old_gate, win_gate], axis=1)
                        dropped = 1
                    self.store_prev(win_kv, win_gate, dropped)
            self._undo = None
        return n
""",
    ),
)

MODEL_PATCHES = (
    (
        """            if remainder is None or not ratio:
                continue
            values = (
""",
        """            if remainder is None or not ratio:
                continue
            if getattr(entry, "_omlx_arbitrary_rollback", False):
                continue
            values = (
""",
    ),
)


def apply_patches(path: Path, patches, label: str) -> None:
    source = path.read_text()
    changed = False
    for old, new in patches:
        if new in source:
            continue
        if source.count(old) != 1:
            raise SystemExit(
                f"refusing unexpected {label} source near: {old.splitlines()[0]!r}"
            )
        source = source.replace(old, new)
        changed = True
    backup = path.with_suffix(path.suffix + ".pre-dsv4-pooling-rollback")
    if changed:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(source)
    print(f"{label}={'installed' if changed else 'present'} path={path}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} CACHE_EXTRAS_PY DSV4_MTP_MODEL_PY"
        )
    apply_patches(Path(sys.argv[1]).resolve(), CACHE_PATCHES, "pooling_rollback")
    apply_patches(Path(sys.argv[2]).resolve(), MODEL_PATCHES, "mtp_depth5")


if __name__ == "__main__":
    main()
