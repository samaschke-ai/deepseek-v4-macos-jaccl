#!/usr/bin/env python3
"""Install rank-0 control synchronization hooks in oMLX MTP generation."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PATCHES = (
    (
        """        host = mx.concatenate(
            [m_arr, targets, state.drafts.astype(mx.int32)]
        ).tolist()
        m = int(host[0])
""",
        """        host = mx.concatenate(
            [m_arr, targets, state.drafts.astype(mx.int32)]
        ).tolist()
        rank_sync = globals().get("_rank_sync_host")
        if rank_sync is not None:
            host = rank_sync(host)
        m = int(host[0])
""",
    ),
    (
        """        host = mx.concatenate(
            [
                m_arr.astype(mx.int32),
                state.drafts.astype(mx.int32),
                res_samples.astype(mx.int32),
                bonus_tok.astype(mx.int32),
            ]
        ).tolist()
        m = int(host[0])
""",
        """        host = mx.concatenate(
            [
                m_arr.astype(mx.int32),
                state.drafts.astype(mx.int32),
                res_samples.astype(mx.int32),
                bonus_tok.astype(mx.int32),
            ]
        ).tolist()
        rank_sync = globals().get("_rank_sync_host")
        if rank_sync is not None:
            host = rank_sync(host)
        m = int(host[0])
""",
    ),
    (
        """                emit_last_id = draft_ids[m]
                emit_last_lp = combined_lp[m]

    # --- stats ---
""",
        """                emit_last_id = draft_ids[m]
                emit_last_lp = combined_lp[m]

    rank_sync = globals().get("_rank_sync_host")
    if rank_sync is not None:
        m, emit_last_id = rank_sync([m, emit_last_id])
        m = int(m)
        emit_last_id = int(emit_last_id)
        emit_last_lp = combined_lp[m if m < k else k]

    # --- stats ---
""",
    ),
    (
        """    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    hidden_at_confirmed = hidden[:, 0:1, :]
""",
        """    rank_sync = globals().get("_rank_sync_host")
    if rank_sync is not None:
        accept = bool(rank_sync([int(accept)])[0])
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    hidden_at_confirmed = hidden[:, 0:1, :]
""",
    ),
    (
        """        emit_id, _ = _residual_sample(verify_accept_lp, draft_accept_lp)
        emit_lp = verify_lp_2d.squeeze(0)

    emit_tok = mx.array([emit_id], dtype=mx.uint32)
""",
        """        emit_id, _ = _residual_sample(verify_accept_lp, draft_accept_lp)
        emit_lp = verify_lp_2d.squeeze(0)

    rank_sync = globals().get("_rank_sync_host")
    if rank_sync is not None:
        emit_id = int(rank_sync([emit_id])[0])
    emit_tok = mx.array([emit_id], dtype=mx.uint32)
""",
    ),
    (
        """    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    if depth <= 0:
""",
        """    sampler = _resolve_sampler(gen_batch)
    # Build DSpark's full lazy chain locally on both ranks. Synchronizing each
    # sampler call would force five GPU/CPU fences and serialize the chain.
    sampler = getattr(sampler, "rank_local_sampler", sampler)
    procs = _proc_list(gen_batch)

    if depth <= 0:
""",
    ),
    (
        """    state.draft_accept_lps = draft_accept_lps
    mx.async_eval(state.drafts)


# ---------------------------------------------------------------------------
""",
        """    state.draft_accept_lps = draft_accept_lps
    rank_sync_drafts = globals().get("_rank_sync_draft_array")
    if rank_sync_drafts is not None:
        state.drafts = rank_sync_drafts(state.drafts)
    else:
        mx.async_eval(state.drafts)


# ---------------------------------------------------------------------------
""",
    ),
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} BATCH_GENERATOR_PY")
    path = Path(sys.argv[1]).resolve()
    source = path.read_text()
    changed = False
    for old, new in PATCHES:
        if new in source:
            continue
        if source.count(old) != 1:
            raise SystemExit(f"refusing unexpected oMLX source near: {old.splitlines()[0]!r}")
        source = source.replace(old, new)
        changed = True

    backup = path.with_suffix(path.suffix + ".pre-dsv4-rank-control")
    if changed:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(source)
        print(f"omlx_mtp_rank_control=installed path={path}")
    else:
        print(f"omlx_mtp_rank_control=present path={path}")


if __name__ == "__main__":
    main()
