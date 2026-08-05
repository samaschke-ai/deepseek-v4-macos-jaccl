# DeepSeek V4 Flash on two Macs with oMLX + JACCL

A reproducible integration recipe for serving the official
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint across two Apple Silicon Macs
with oMLX, expert parallelism, MLX JACCL, and integrated DSpark/MTP.

This repository contains patches, launch examples, tests, and measurements. It
does **not** contain model weights, credentials, or a turnkey installer.

## Current truth

The deployed reference is canonical EP2: 128 routed experts on each rank,
replicated trunk/shared experts/DSpark, JACCL collectives, five-position
DSpark/MTP, six prompt slots, six decode slots, dynamic caches, and the full
1,048,576-token context ceiling.

The active performance objective is **at least 100 visible tok/s for one warmed
request**. Aggregate concurrency throughput is a separate measurement.

### Current exact `count300` baseline

| Metric | Canonical EP2 result |
|---|---:|
| Visible decode, three matched repeats | **60.88 tok/s median** |
| Best visible decode repeat | **61.21 tok/s** |
| Median target cycle | **95.99 ms** |
| Speculative yield | **5.84 tokens/cycle** |
| DSpark acceptance | **527/535 (98.5%)** |
| Output and rank control | identical |

At 5.84 tokens/cycle, 100 tok/s requires approximately **58.4 ms/cycle**.
The canonical runtime remains below that gate; this repository does not claim
100 tok/s has been reached.

The measurement uses the exact DGX-compatible `count300` prompt, one warm-up,
three matched repeats, no `seed` parameter, and a parser that includes both
`delta.reasoning`/`delta.reasoning_content` and `delta.content`.

### Latest candidate decisions

Two correctness-first attention layouts were tested and not promoted:

- **Post-attention head gather:** exact through eight layers across attention
  compression ratios 0, 4, and 128, but diverged after layer 9 attention at
  full depth. Rejected for exactness.
- **Pre-attention Q gather:** bit-identical through all 43 layers, including
  final hidden state and logits, but measured 57.17 tok/s median because one
  additional collective per layer cost more than the saved projection work.
  Rejected for performance.

No candidate was deployed. Full details are in
[`results/deepseek-100tps-phase-20260805.json`](results/deepseek-100tps-phase-20260805.json).

## What is accepted in the reference recipe

- Official DeepSeek V4 Flash 0731 checkpoint lineage.
- EP2 with routed experts split 128/128; trunk, shared experts, and DSpark
  replicated.
- MLX JACCL over direct Thunderbolt RDMA. Do not silently substitute TCP model
  collectives for production validation.
- Five-position DSpark/MTP with exact pooling-cache rollback and rank-0
  speculative control.
- Native EP route masking and the accepted masked small-M MXFP4 QMV path.
- Six prompt slots, six decode slots, dynamic caches, and full context.
- SSE streaming, disconnect draining, readiness handling, and OpenAI-compatible
  DSML tool conversion.

## Quick start

### 1. Prepare both ranks

Install matching macOS, Python, MLX, oMLX, native kernel artifacts, and the
same official checkpoint on both Macs. Set these variables independently on
each rank:

```bash
export OMLX=/path/to/your/omlx-checkout
export PYTHON=$OMLX/venv/bin/python
export MODEL=/path/to/DeepSeek-V4-Flash-0731
```

The checkpoint must be the official 0731 lineage. Do not point the server at a
second copy under another path: that can load a second model and exhaust
unified memory.

### 2. Apply the patches in order

The patchers are idempotent and refuse unexpected source layouts.

```bash
SWITCH=$OMLX/omlx/patches/deepseek_v4/switch_layers.py
FAST=$OMLX/omlx/custom_kernels/glm_moe_dsa/fast.py
CACHE=$OMLX/omlx/patches/deepseek_v4/cache_extras.py
MTP_MODEL=$OMLX/omlx/patches/mlx_lm_mtp/deepseek_v4_model.py
MTP_BATCH=$OMLX/omlx/patches/mlx_lm_mtp/batch_generator.py

$PYTHON patches/patch_omlx_native_shape_fallback.py "$SWITCH"
$PYTHON patches/patch_omlx_ep_route_mask.py "$SWITCH"
$PYTHON patches/patch_omlx_ep_masked_qmv.py "$FAST" "$SWITCH"
$PYTHON patches/patch_omlx_pooling_rollback.py "$CACHE" "$MTP_MODEL"
$PYTHON patches/patch_omlx_mtp_rank_control.py "$MTP_BATCH"
```

Install the checksum-pinned native artifacts before enabling masked QMV.
The published artifact is linked from the
[`deepseek-ep-masked-qmv-v1` oMLX release](https://github.com/samaschke-ai/omlx/releases/tag/deepseek-ep-masked-qmv-v1).

```bash
$PYTHON patches/install_native_kernels.py /path/to/staged/glm_moe_dsa \
  "$OMLX/omlx/custom_kernels/glm_moe_dsa"
```

Run the rollback test before starting distributed serving:

```bash
$PYTHON tests/test_pooling_rollback.py "$CACHE"
```

### 3. Configure the sanitized JACCL hostfile

Copy [`examples/hosts-jaccl.json`](examples/hosts-jaccl.json) and replace its
placeholder SSH hostnames, direct-link addresses, and RDMA device names with
the values for your two Macs. Keep the JACCL backend and ensure both ranks use
the same model/runtime sources.

The separate rank-control socket is only for rank-0-authoritative speculative
control. Model tensors and model collectives must remain on JACCL.

### 4. Launch and verify

Set the values consumed by
[`examples/run-rank-server.sh`](examples/run-rank-server.sh) on the launch
machine:

```bash
export DSV4_WORK="$OMLX"
export DSV4_MODEL="$MODEL"
export DSV4_SERVER_EP="$PWD/runtime/server_ep2.py"

$OMLX/venv/bin/mlx.launch --verbose \
  --hostfile examples/hosts-jaccl.json \
  --cwd "$OMLX" \
  --python "$OMLX/venv/bin/python" \
  "$PWD/examples/run-rank-server.sh"
```

The hostfile supplies the rank-specific SSH/RDMA topology; the rank launcher
supplies the six-slot serving flags. Publish only the rank-0 API endpoint.

Before benchmarking, verify:

1. both ranks report active RDMA devices;
2. both ranks pass the checkpoint and native-artifact checksums;
3. the readiness file appears only after model warm-up;
4. `/v1/models` reports the expected public model;
5. no stale EXO, llama.cpp, distributed-smoke, or old benchmark process exists.

## Benchmarking correctly

Use identical clean restart, warm-up, request ordering, and output-hash checks
for every candidate. Long DSpark runs are invalid without MTP telemetry. Do not
send `seed=1` in this path: it can silently disable MTP activation.

The canonical `count300` result is the primary predictable-workload comparison.
For public API testing, also exercise ordinary generation, reasoning, JSON, and
OpenAI tool calls. Parse streamed reasoning and content separately when doing
application-level accounting.

## Historical accepted results

These are useful milestones from earlier matched workloads, not substitutes for
the current `count300` baseline above:

| Workload or stage | Result |
|---|---:|
| 32K prefill after accepted EP/power remediation | 287.15 tok/s |
| Five-position rollback decode | 36.45–37.77 tok/s |
| Masked small-M MXFP4 QMV matched run | 40.36 tok/s |
| DSpark yield after rollback repair | 3.76 tokens/cycle |

The old 36–40 tok/s figures used a different short counting workload and
measurement stage. They must not be presented as the current `count300`
throughput or as evidence that the 100 tok/s objective has been reached.

## Known rejected approaches

- Generic `mx.compile` around routed EP-MoE: transient gains changed output or
  degraded under sustained graph/cache growth.
- Shared-expert tensor sharding: prefill gains did not survive decode.
- Native block kernels for the six-row/36-route verification shape: decode
  regressed.
- Current EXO and llama.cpp RPC paths: incorrect placement/output or
  insufficient memory on this topology.
- Post-attention head gathering: full-depth numerical drift.
- Pre-attention Q gathering: exact but slower due to per-layer collectives.
- TCP fallback for production model collectives.

## Documentation map

- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — methodology, historical
  measurements, and current phase checkpoint.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — EP2, DSpark, cache, and
  transport design.
- [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) — MXFP4/MXFP8 boundaries and
  numerical constraints.
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — upstream candidates and provenance.
- [`results/`](results/) — machine-readable benchmark receipts.
- [`patches/`](patches/) — source patchers.
- [`runtime/`](runtime/) — sanitized server/router examples.
- [`tests/`](tests/) — rollback, router, and integration checks.

## License and checkpoint terms

Apache-2.0 for this repository. The repository contains integration code and
patches, not DeepSeek weights. Follow the checkpoint's own license and terms
separately.
