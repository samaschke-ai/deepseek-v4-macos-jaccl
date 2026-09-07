#!/usr/bin/env python3
"""Serve official DeepSeek V4 with EP2 backbone and replicated DSpark."""

from __future__ import annotations

import os
import signal
import socket
import struct
import sys
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
from mlx.nn.layers.distributed import shard_inplace

READY_FILE = os.environ.get("DSV4_READY_FILE", "/tmp/deepseek-v4/ready")
CONTROL_HOST = os.environ.get("DSV4_CONTROL_HOST")
if not CONTROL_HOST:
    raise RuntimeError("DSV4_CONTROL_HOST must be set to rank 0's direct-link address")
CONTROL_PORT = int(os.environ.get("DSV4_CONTROL_PORT", "29650"))


def model_path(argv: list[str]) -> str:
    return argv[argv.index("--model") + 1]


path = model_path(sys.argv)
from omlx.utils.model_loading import maybe_apply_pre_load_patches

settings = SimpleNamespace(
    mtp_enabled=os.environ.get("OMLX_MTP_ENABLED", "1") == "1",
    mtp_num_draft_tokens=int(os.environ.get("OMLX_MTP_DRAFT_TOKENS", "5")),
)
maybe_apply_pre_load_patches(path, model_settings=settings)

import mlx_lm.models.deepseek_v4 as dsv4

# Split complete routed experts 128/128. The target trunk, shared experts, and
# integrated three-stage DSpark module remain replicated on both ranks.
moe_cls = dsv4.DeepseekV4MoE
_original_moe_call = moe_cls.__call__
_RANK_CONTROL = None
_FATAL_GENERATION_HOOK_INSTALLED = False
_ORIGINAL_THREADING_EXCEPTHOOK = threading.excepthook


def _install_fatal_generation_thread_hook(rank: int) -> None:
    """Exit the rank if mlx-lm's generation worker dies behind a live HTTP API."""
    global _FATAL_GENERATION_HOOK_INSTALLED
    if _FATAL_GENERATION_HOOK_INSTALLED:
        return

    def fatal_generation_thread_exception(args) -> None:
        _ORIGINAL_THREADING_EXCEPTHOOK(args)
        if "_generate" not in args.thread.name:
            return
        print(
            f"[ep2] fatal generation thread exit rank={rank}; terminating rank process",
            flush=True,
        )
        if rank == 0:
            try:
                os.unlink(READY_FILE)
            except FileNotFoundError:
                pass
        os._exit(70)

    threading.excepthook = fatal_generation_thread_exception
    _FATAL_GENERATION_HOOK_INSTALLED = True


class _RankControlChannel:
    """Rank-0 control decisions over a dedicated direct-Thunderbolt socket."""

    def __init__(
        self,
        rank: int,
        host: str = CONTROL_HOST,
        port: int = CONTROL_PORT,
    ):
        self.rank = rank
        self.lock = threading.Lock()
        if rank == 0:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(300)
            self.sock, _ = listener.accept()
            listener.close()
        else:
            deadline = time.monotonic() + 300
            while True:
                try:
                    self.sock = socket.create_connection((host, port), timeout=10)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
        self.sock.settimeout(300)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        while size:
            chunk = self.sock.recv(size)
            if not chunk:
                raise ConnectionError("rank-control channel closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def sync_ints(self, values) -> list[int]:
        local = [int(value) for value in values]
        with self.lock:
            if self.rank == 0:
                payload = struct.pack(f"!I{len(local)}q", len(local), *local)
                self.sock.sendall(payload)
                return local
            (count,) = struct.unpack("!I", self._recv_exact(4))
            if count != len(local):
                raise RuntimeError(
                    f"rank-control shape mismatch: rank0={count}, rank1={len(local)}"
                )
            return list(struct.unpack(f"!{count}q", self._recv_exact(count * 8)))


def _sync_control(values) -> list[int]:
    if _RANK_CONTROL is None:
        raise RuntimeError("rank-control channel is not initialized")
    return _RANK_CONTROL.sync_ints(values)


def _sync_draft_array_rank0(drafts):
    """Synchronize one complete DSpark draft chain without per-draft fences."""
    shape = tuple(int(size) for size in drafts.shape)
    count = 1
    for size in shape:
        count *= size
    if _RANK_CONTROL is None:
        raise RuntimeError("rank-control channel is not initialized")
    if _RANK_CONTROL.rank == 0:
        _RANK_CONTROL.sync_ints(drafts.reshape(-1).tolist())
        return drafts
    # Evaluate the replicated DSpark graph so rank 1 keeps identical cache
    # progress, then replace only its control tokens with rank 0's chain.
    mx.eval(drafts)
    values = _RANK_CONTROL.sync_ints([0] * count)
    return mx.array(values, dtype=drafts.dtype).reshape(shape)


class _RankSynchronizedSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    @property
    def rank_local_sampler(self):
        return self.sampler

    def __call__(self, logprobs):
        token = self.sampler(logprobs)
        values = _sync_control(token.reshape(-1).tolist())
        return mx.array(values, dtype=token.dtype).reshape(token.shape)

    def __getattr__(self, name):
        return getattr(self.sampler, name)


def _ep_moe_call(self, x, input_ids):
    if not hasattr(self, "_ep_start"):
        return _original_moe_call(self, x, input_ids)

    input_dtype = x.dtype
    inds, scores = self.gate(x, input_ids)
    local = (inds >= self._ep_start) & (inds < self._ep_end)
    local_inds = mx.clip(inds - self._ep_start, 0, self._ep_count - 1)
    weights = scores.astype(mx.float32) * local.astype(mx.float32)
    # The patched native block builder uses zero scores as an ownership mask:
    # remote routes retain fixed shapes but dispatch no routed-expert GEMMs.
    y = self.switch_mlp(x, local_inds, scores=weights)
    if y.ndim == scores.ndim + 1:
        # Skipped native rows are intentionally unwritten. Select explicit
        # zeros before weighting so their contents cannot contaminate the sum.
        y = mx.where(local[..., None], y, mx.zeros((), dtype=y.dtype))
        y = (y.astype(mx.float32) * weights[..., None]).sum(-2)
    y = mx.distributed.all_sum(y, group=self._ep_group).astype(input_dtype)
    return y + self.shared_experts(x)


moe_cls.__call__ = _ep_moe_call


def _expert_shard(self, group=None):
    global _RANK_CONTROL
    group = group or mx.distributed.init()
    size = group.size()
    rank = group.rank()
    _install_fatal_generation_thread_hook(rank)
    if _RANK_CONTROL is None:
        _RANK_CONTROL = _RankControlChannel(rank)
    if size != 2:
        raise ValueError(f"DeepSeek EP runtime requires exactly 2 ranks, got {size}")

    start = end = 0
    for layer in self.model.layers:
        moe = layer.ffn
        total = moe.switch_mlp.gate_proj.num_experts
        if total % size:
            raise ValueError(f"{total} experts not divisible by {size}")
        count = total // size
        start, end = rank * count, (rank + 1) * count
        moe._ep_start = start
        moe._ep_end = end
        moe._ep_count = count
        moe._ep_group = group
        # Only EP target modules opt into the masked QMV path; replicated
        # DSpark/MTP modules retain their stock numerical path.
        moe.switch_mlp._ep_masked_qmv = True
        for proj in (
            moe.switch_mlp.gate_proj,
            moe.switch_mlp.up_proj,
            moe.switch_mlp.down_proj,
        ):
            shard_inplace(proj, lambda _path, _value: 0, group=group)

    print(
        f"[ep2] rank={rank}/{size} main experts={start}..{end - 1}; "
        "DSpark replicated",
        flush=True,
    )


dsv4.Model.shard = _expert_shard


def _report_native_dsa_state(_signum, _frame) -> None:
    gib = 1024**3
    print(
        "[ep2] native DSA state "
        f"indexer_disabled={dsv4._DEEPSEEK_V4_INDEXER_NATIVE_DISABLED} "
        f"sparse_attention_disabled="
        f"{dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED} "
        f"metal_active_gib={mx.get_active_memory() / gib:.2f} "
        f"metal_peak_gib={mx.get_peak_memory() / gib:.2f} "
        f"metal_cache_gib={mx.get_cache_memory() / gib:.2f}",
        flush=True,
    )
    mx.reset_peak_memory()


signal.signal(signal.SIGUSR1, _report_native_dsa_state)

# The stock adaptive draft controller uses rank-local timings. Different choices
# on the two ranks produce incompatible collective shapes. Keep both ranks at
# native depth 5; the model's rollback-safety hook may cap a verify cycle to 3.
from omlx.patches.mlx_lm_mtp import batch_generator as mtp_batch


class _FixedDepthController:
    def __init__(self, max_depth, **_kwargs):
        self.max_depth = max(1, int(max_depth))
        self.cur = self.max_depth
        self.t = {}

    def observe(self, *_args, **_kwargs):
        self.cur = self.max_depth

    def should_exit(self):
        return False


mtp_batch._DepthController = _FixedDepthController
mtp_batch._rank_sync_host = _sync_control
mtp_batch._rank_sync_draft_array = _sync_draft_array_rank0

import mlx_lm.server as mlx_server

_original_make_sampler = mlx_server._make_sampler


def _make_rank_synchronized_sampler(args, tokenizer):
    return _RankSynchronizedSampler(_original_make_sampler(args, tokenizer))


mlx_server._make_sampler = _make_rank_synchronized_sampler
print("[ep2] rank-0 synchronized DSpark/MTP control depth=5", flush=True)

mlx_server.main()
