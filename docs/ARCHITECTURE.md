# Architecture

## Data plane

```text
OpenAI client
    │
    ▼
optional compatibility router
    │
    ▼
rank-0 mlx-lm HTTP server
    │
    ├── replicated trunk / attention / shared experts
    ├── routed experts 0..127 ─────────────── rank 0
    ├── routed experts 128..255 ───────────── rank 1
    └── per-layer partial output ── JACCL all-sum over Thunderbolt RDMA
```

Both ranks execute the same target tokens. Rank 0 owns API output and
speculative decisions.

## Control plane

MLX/JACCL collectives carry model tensors only. Speculative tokens, acceptance,
rollback, and stop decisions are sent over a dedicated rank-0-authoritative
socket on the direct-link IP. This avoids inserting application messages into
JACCL's request-sharing/protocol sequence.

The complete five-token draft chain is built lazily and synchronized once.
Synchronizing each draft sampler separately introduces five GPU/CPU fences and
substantially reduces generation throughput.

## Expert parallel route masking

Each rank stores 128 complete routed experts. A naïve wrapper clips remote
expert IDs to a local ID, computes all six routes, and masks remote outputs only
afterward. Prefill therefore performs nearly full routed-expert work twice.

The route-mask patch sorts active routes first within each expert and passes an
ownership mask to the native MXFP4 block builder. Inactive suffixes emit no
GEMM blocks. Output tensors and JACCL collectives retain fixed shapes.

Small decode/verify windows remain on generic gathered QMM. Tests forcing the
native block path for a six-row/36-route verify were slower because block-plan
and kernel-launch overhead exceeded the saved arithmetic.

## Pooling rollback

DeepSeek V4's ratio-4 compressor carries a previous raw window and appends
compressed outputs to a pool. A six-row target verify can cross one or more
pooling seams. On rejection, restoring only the remainder length is
insufficient because accepted rows may themselves complete a compressed
window.

The exact rollback patch records:

- the pre-update remainder buffers and pooled state;
- raw KV/gate inputs for the verify window;
- compressed outputs produced by that window;
- previous overlap-window state.

Rollback restores the pre-update snapshot, replays the accepted raw prefix,
then appends the corresponding prefix of already-computed compressed outputs.
All four remainders and five rejection positions are regression-tested for both
single and batched cache classes.

## Failure handling

- Unsupported native MoE shapes fall back for that call only.
- An uncaught generation-thread exception removes readiness and exits the rank,
  allowing the service supervisor to restart the pair.
- The router starts SSE before upstream response headers and emits heartbeats
  during long prefill.
- Disconnected distributed requests are drained rather than cancelling one
  rank and stranding the other.

## Transport alternatives

JACCL is the native MLX RDMA path used here. NCCL and RCCL do not support Apple
GPUs. UCX, libfabric, or MPI would require a new MLX transport/backend and do
not currently offer a proven Apple-GPU zero-copy path. Practical optimization
work therefore centers on fewer/fused collectives, overlap, and better
partitioning within JACCL.
