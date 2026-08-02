# DeepSeek V4 Flash on two Macs with oMLX + JACCL

A reproducible, experimental recipe for serving the official
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint across two Apple Silicon Macs
with:

- oMLX's official DeepSeek V4 and integrated DSpark/MTP implementation;
- expert parallelism across two ranks;
- MLX JACCL collectives over Apple Thunderbolt RDMA;
- five-position speculative verification with exact pooling-cache rollback;
- native MXFP4 route masking for locally owned experts;
- six logical serving slots and the checkpoint's 1,048,576-token ceiling;
- incremental OpenAI-compatible streaming and DSML tool conversion.

No model weights, credentials, private hostnames, or internal deployment values
are included.

> **Status:** research/advanced operations. The patches are intentionally kept
> small and reviewable while upstream submissions are prepared. Pin and test the
> exact oMLX/MLX revisions you deploy.

## Measured reference system

Two M4 Max Macs with 128 GB unified memory each, connected by direct
Thunderbolt RDMA. Rank 1 used an Apple 96W USB-C adapter negotiating 94W and
macOS AC High Power mode (`powermode 2`).

### Headline improvement versus original production

| Metric | Original production | Final accepted | Improvement |
|---|---:|---:|---:|
| 32K prefill (PP) | 178.24 tok/s | 287.15 tok/s | **+61.1%** |
| Visible decode (TG) | ~25.4 tok/s | 36.45–37.77 tok/s | **+43–49%** |

The original figures are pre-remediation production measurements. The
stage-level figures below isolate individual route-mask, power, and rollback
changes and therefore use different immediate baselines.

| Measurement | Result |
|---|---:|
| 8K prefill before EP route masking | 259 tok/s |
| 8K prefill after EP route masking | 312 tok/s |
| 32K prefill, final configuration | 287 tok/s |
| Decode before five-position rollback | 30.75 visible tok/s |
| Decode after five-position rollback | 36.45–37.77 visible tok/s |
| DSpark yield after rollback repair | 3.76 tokens/cycle |

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md),
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md), and the machine-readable
[`results/dual-m4-max-20260802.json`](results/dual-m4-max-20260802.json).

## Requirements

- Two Apple Silicon Macs with enough unified memory for the selected EP split.
- Matching macOS and MLX versions on both ranks.
- A direct Thunderbolt network whose RDMA devices report active.
- MLX with the JACCL backend.
- Editable oMLX source containing the DeepSeek V4 and mlx-lm MTP patches.
- The official DeepSeek V4 Flash 0731 checkpoint on both ranks.
- The same Python sources and native artifacts on both ranks.

JACCL is the practical native MLX RDMA transport for this topology. NCCL and
RCCL do not support Apple GPUs. UCX, libfabric, or MPI would require a custom
MLX transport/backend and do not currently offer a better established
Apple-GPU path.

## Patch order

Set paths for your editable oMLX checkout:

```bash
OMLX=/path/to/omlx
PYTHON=/path/to/venv/bin/python
SWITCH=$OMLX/omlx/patches/deepseek_v4/switch_layers.py
CACHE=$OMLX/omlx/patches/deepseek_v4/cache_extras.py
MTP_MODEL=$OMLX/omlx/patches/mlx_lm_mtp/deepseek_v4_model.py
MTP_BATCH=$OMLX/omlx/patches/mlx_lm_mtp/batch_generator.py

$PYTHON patches/patch_omlx_native_shape_fallback.py "$SWITCH"
$PYTHON patches/patch_omlx_ep_route_mask.py "$SWITCH"
$PYTHON patches/patch_omlx_pooling_rollback.py "$CACHE" "$MTP_MODEL"
$PYTHON patches/patch_omlx_mtp_rank_control.py "$MTP_BATCH"
```

Every patcher is idempotent and refuses an unexpected source layout.

Run the rollback matrix before deployment:

```bash
$PYTHON tests/test_pooling_rollback.py "$CACHE"
# POOLING_ROLLBACK_MATRIX_OK checks=40
```

## Launch outline

1. Install matching native DeepSeek artifacts on both ranks.
2. Apply the patch set on both ranks.
3. Copy `runtime/server_ep2.py` to both ranks.
4. Adapt `examples/hosts-jaccl.json` to your LAN and RDMA device names.
5. Set `DSV4_CONTROL_HOST` to rank 0's direct-link address. This compact TCP
   channel carries only rank-0-authoritative speculative control; model tensors
   and collectives remain on JACCL RDMA.
6. Launch through `mlx.launch` with the JACCL hostfile.
7. Publish only the rank-0 API, optionally through `runtime/openai_router.py`.

`examples/run-rank-server.sh` documents the measured serving flags. Do not copy
placeholder addresses unchanged.

## Why the patches exist

- **EP route masking:** an ordinary clipped EP wrapper computes remote expert
  routes and multiplies them by zero afterward. The block-plan mask prevents
  those native GEMMs while preserving fixed shapes.
- **Exact pooling rollback:** ratio-4 pooling previously allowed only three
  safe verify positions. The patch records compressed verify outputs and raw
  overlap state, allowing exact rollback to every accepted prefix of a
  six-token target verify.
- **Native shape fallback:** cancellation/ragged compaction can produce native
  block shapes outside compiled support. Those calls fall back to general
  `gather_qmm` instead of killing generation.
- **Rank control:** stochastic speculative decisions must be identical before
  both ranks enter their next JACCL collective. A separate direct-link socket
  avoids inserting control messages into MLX's collective framing.

## Validated and rejected settings

- Keep `--prefill-step-size 512` on the reference M4 Max pair. Retested 1,024
  was slower.
- Do not force native block kernels for the six-row/36-route verify. It reduced
  decode from 37.0 to 29.9 tok/s.
- Shared-expert tensor sharding improved prefill but reduced decode from 30.75
  to 23.46 tok/s on this backend and was rejected.
- Do not use TCP fallback for model tensors when validating this recipe.

## Repository map

- `runtime/` — EP2 server and optional OpenAI compatibility router.
- `patches/` — idempotent oMLX/MLX source patchers.
- `tests/` — pooling rollback and router stream tests.
- `examples/` — sanitized hostfile and rank launcher.
- `scripts/` — OpenAI endpoint benchmark.
- `docs/` — architecture, quantization, performance methodology, and upstream plan.
- `results/` — machine-readable benchmark receipt.

## License

Apache-2.0. The repository contains integration code and patches, not DeepSeek
weights. Follow the checkpoint's own license and terms separately.
