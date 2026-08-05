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
| Measured median cycle latency | **95.99 ms** |
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
same official checkpoint on both Macs. The compact launcher below requires the
oMLX checkout, this integration repository, and the checkpoint to exist at
identical absolute paths on both ranks. If your paths differ, adapt the launcher
and per-rank environment instead of using these commands unchanged.

Export the same values in the setup shell on **each** rank, including the
launch machine. These exports are for the setup commands; section 3 configures
the environment inherited by the SSH-launched processes.

```bash
export OMLX=/same/path/on/both-ranks/omlx-checkout
export REPO=/same/path/on/both-ranks/deepseek-v4-macos-jaccl
export PYTHON=$OMLX/venv/bin/python
export MODEL=/same/path/on/both-ranks/DeepSeek-V4-Flash-0731
```

Before continuing, run this preflight on both ranks:

```bash
test -x "$PYTHON"
test -d "$MODEL"
test -f "$REPO/runtime/server_ep2.py"
test -x "$REPO/examples/run-rank-server.sh"
```

The checkpoint must be the official 0731 lineage. Do not point the server at a
second copy under another path: that can load a second model and exhaust
unified memory.

Use the tested source and ABI pins on both ranks:

- oMLX commit [`e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f`](https://github.com/samaschke-ai/omlx/commit/e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f)
  (tag `deepseek-ep-masked-qmv-v1`);
- MLX `0.32.0`;
- mlx-lm commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc`;
- CPython 3.13 for the published `cp313` native extension.

Verify the checkout and environment before patching:

```bash
test "$(git -C "$OMLX" rev-parse HEAD)" = \
  "e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f"
test "$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = \
  "3.13"
test "$("$PYTHON" -c 'import mlx; print(mlx.__version__)')" = "0.32.0"
"$PYTHON" -m pip freeze | grep -F \
  'mlx-lm @ git+https://github.com/ml-explore/mlx-lm@ab1806e8f5d6aa035973af194a1b9198ab4754dc'
```

### 2. Apply the patches in order on both ranks

Run this entire section separately on **each** rank. Identical absolute paths do
not imply a shared filesystem. The patchers are idempotent and refuse
unexpected source layouts.

```bash
SWITCH=$OMLX/omlx/patches/deepseek_v4/switch_layers.py
FAST=$OMLX/omlx/custom_kernels/glm_moe_dsa/fast.py
CACHE=$OMLX/omlx/patches/deepseek_v4/cache_extras.py
MTP_MODEL=$OMLX/omlx/patches/mlx_lm_mtp/deepseek_v4_model.py
MTP_BATCH=$OMLX/omlx/patches/mlx_lm_mtp/batch_generator.py

$PYTHON "$REPO/patches/patch_omlx_native_shape_fallback.py" "$SWITCH"
$PYTHON "$REPO/patches/patch_omlx_ep_route_mask.py" "$SWITCH"
$PYTHON "$REPO/patches/patch_omlx_ep_masked_qmv.py" "$FAST" "$SWITCH"
$PYTHON "$REPO/patches/patch_omlx_pooling_rollback.py" "$CACHE" "$MTP_MODEL"
$PYTHON "$REPO/patches/patch_omlx_mtp_rank_control.py" "$MTP_BATCH"
```

Install the checksum-pinned native artifacts on each rank before enabling
masked QMV. Use
`omlx-0.5.4-cp313-cp313-macosx_15_0_universal2.whl` from the
[`deepseek-ep-masked-qmv-v1` oMLX release](https://github.com/samaschke-ai/omlx/releases/tag/deepseek-ep-masked-qmv-v1),
whose SHA-256 is
`359858e91989e0ac6a8ff7551a4475b710c125a0f74a4c4e13586284dc1319c4`.
The installer independently verifies every extracted artifact checksum.

```bash
$PYTHON "$REPO/patches/install_native_kernels.py" \
  /path/to/staged/glm_moe_dsa \
  "$OMLX/omlx/custom_kernels/glm_moe_dsa"
```

Run the rollback test on each rank, then compare the `shasum` output between
ranks before starting distributed serving:

```bash
$PYTHON "$REPO/tests/test_pooling_rollback.py" "$CACHE"
shasum "$FAST" "$SWITCH" "$CACHE" "$MTP_MODEL" "$MTP_BATCH" \
  "$REPO/runtime/server_ep2.py"
```

### 3. Configure the sanitized JACCL hostfile

Copy [`examples/hosts-jaccl.json`](examples/hosts-jaccl.json) and replace its
placeholder SSH hostnames, direct-link addresses, and RDMA device names with
the values for your two Macs. Keep the JACCL backend and ensure both ranks use
the same model/runtime sources.

Add the common absolute launch paths to the hostfile's `envs` array so every
new SSH process receives them; exports from an earlier remote shell do not
persist into `mlx.launch`:

```json
{
  "envs": [
    "OMLX_MTP_ENABLED=1",
    "OMLX_MTP_DRAFT_TOKENS=5",
    "OMLX_MTP_ROWWISE_BATCH=0",
    "DSV4_CONTROL_HOST=10.0.0.1",
    "DSV4_CONTROL_PORT=29650",
    "DSV4_READY_FILE=/tmp/deepseek-v4/ready",
    "DSV4_WORK=/same/path/on/both-ranks/omlx-checkout",
    "DSV4_MODEL=/same/path/on/both-ranks/DeepSeek-V4-Flash-0731",
    "DSV4_SERVER_EP=/same/path/on/both-ranks/deepseek-v4-macos-jaccl/runtime/server_ep2.py"
  ]
}
```

The separate rank-control socket is only for rank-0-authoritative speculative
control. Model tensors and model collectives must remain on JACCL.

### 4. Launch and verify

With `OMLX` and `REPO` set on the launch machine and the three `DSV4_*` paths
in the hostfile, launch [`examples/run-rank-server.sh`](examples/run-rank-server.sh):

```bash
$OMLX/venv/bin/mlx.launch --verbose \
  --hostfile "$REPO/examples/hosts-jaccl.json" \
  --cwd "$OMLX" \
  --python "$OMLX/venv/bin/python" \
  "$REPO/examples/run-rank-server.sh"
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
