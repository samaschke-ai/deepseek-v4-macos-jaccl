# DeepSeek V4 Flash on two Apple Silicon Macs

Serve the official `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint across two
Apple Silicon Macs with oMLX, expert parallelism, MLX JACCL, and DSpark/MTP.

This repository provides the integration patches, launch examples, API router,
tests, and measurement receipts. Model weights are not included.

[Quick start](#quick-start) · [Measurements](#measurements) ·
[Configuration](#configuration) · [Documentation](#documentation)

## What it provides

- Expert-parallel inference with 128 routed experts on each rank.
- JACCL collectives over a direct Thunderbolt RDMA link.
- Replicated trunk, shared experts, and DSpark/MTP control.
- Five-position speculative verification with dynamic prompt and decode caches.
- Six prompt slots, six decode slots, and the checkpoint's full context limit.
- OpenAI-compatible streaming and DSML tool-call conversion.
- Checksum-checked native kernel installation and rollback tests.

## Reference system

The published measurements used two M4 Max Macs with 128 GB unified memory
each and direct Thunderbolt RDMA. The reference serving configuration used:

- official DeepSeek V4 Flash 0731 weights;
- MLX `0.32.0`;
- Python 3.13;
- six prompt slots and six decode slots;
- 512-token prefill steps;
- a 1,048,576-token API context ceiling.

## Measurements

### Exact `count300` workload — 2026-08-05

| Metric | Measurement |
|---|---:|
| Visible decode, three matched repeats | **60.88 tok/s median** |
| Best visible decode repeat | **61.21 tok/s** |
| Median cycle latency | **95.99 ms** |
| Speculative yield | **5.84 tokens/cycle** |
| Verification acceptance | **527/535 (98.5%)** |

The workload uses one warm-up and three matched repeats. Stream accounting
includes reasoning and content deltas separately. See the machine-readable
[measurement receipt](results/deepseek-100tps-phase-20260805.json).

### Additional published measurements

These measurements use different workloads or stages and are not one combined
before/after series:

| Workload or stage | Result |
|---|---:|
| 32K prefill | **287.15 tok/s** |
| Five-position rollback decode | **36.45–37.77 tok/s** |
| Masked small-M MXFP4 QMV workload | **40.36 tok/s** |
| DSpark speculative yield after rollback repair | **3.76 tokens/cycle** |

More methodology and receipts are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Requirements

- Two Apple Silicon Macs with sufficient unified memory for the checkpoint and
  EP2 placement.
- Matching macOS, Python, MLX, oMLX, and native artifacts on both ranks.
- A direct Thunderbolt network with active RDMA devices.
- The official DeepSeek V4 Flash 0731 checkpoint on both ranks.
- Full Xcode/Metal tooling when building native artifacts from source.
- An authenticated, private rank-0 API endpoint for serving clients.

The model-control socket is separate from JACCL: it carries rank-0 speculative
control decisions only. Model tensors and model collectives remain on JACCL.

## Quick start

The commands below assume the oMLX checkout, this repository, and the model
exist at the same absolute paths on both Macs. Run setup and patch commands on
both ranks.

### 1. Set paths

```bash
export OMLX=/same/path/on-both-ranks/omlx
export REPO=/same/path/on-both-ranks/deepseek-v4-macos-jaccl
export MODEL=/same/path/on-both-ranks/DeepSeek-V4-Flash-0731
export PYTHON=$OMLX/venv/bin/python
```

Use these tested source pins:

- oMLX commit [`e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f`](https://github.com/samaschke-ai/omlx/commit/e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f);
- MLX `0.32.0`;
- mlx-lm commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc`;
- CPython 3.13.

Check the paths and versions on both ranks before patching:

```bash
test -x "$PYTHON" && test -d "$MODEL" && test -f "$REPO/runtime/server_ep2.py"
test "$(git -C "$OMLX" rev-parse HEAD)" = \
  "e0121d511bb3ab7d38a9e6b7d6e5ffc6e4f0c96f"
test "$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.13"
test "$("$PYTHON" -c 'import mlx; print(mlx.__version__)')" = "0.32.0"
"$PYTHON" -m pip freeze | grep -F \
  'mlx-lm @ git+https://github.com/ml-explore/mlx-lm@ab1806e8f5d6aa035973af194a1b9198ab4754dc'
```

### 2. Apply the integration patches

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

Download `omlx-0.5.4-cp313-cp313-macosx_15_0_universal2.whl` from the
[`deepseek-ep-masked-qmv-v1` oMLX release](https://github.com/samaschke-ai/omlx/releases/tag/deepseek-ep-masked-qmv-v1).
Its SHA-256 is
`359858e91989e0ac6a8ff7551a4475b710c125a0f74a4c4e13586284dc1319c4`.
Verify the download, extract the `glm_moe_dsa` artifacts, and run the installer
on both ranks:

```bash
echo '359858e91989e0ac6a8ff7551a4475b710c125a0f74a4c4e13586284dc1319c4  /path/to/omlx-0.5.4-cp313-cp313-macosx_15_0_universal2.whl' | shasum -a 256 -c -
$PYTHON "$REPO/patches/install_native_kernels.py" \
  /path/to/staged/glm_moe_dsa \
  "$OMLX/omlx/custom_kernels/glm_moe_dsa"

$PYTHON "$REPO/tests/test_pooling_rollback.py" "$CACHE"
```

### 3. Configure JACCL

Copy [`examples/hosts-jaccl.json`](examples/hosts-jaccl.json) and replace the
`.example.invalid` hostnames, RDMA addresses, and device names with the values
for your two Macs. Do not use the placeholders unchanged.

Set these environment variables in the hostfile for the launched ranks:

```text
OMLX_MTP_ENABLED=1
OMLX_MTP_DRAFT_TOKENS=5
OMLX_MTP_ROWWISE_BATCH=0
DSV4_CONTROL_HOST=<rank-0-direct-address>
DSV4_CONTROL_PORT=<private-control-port>
DSV4_READY_FILE=/tmp/deepseek-v4/ready
DSV4_WORK=<oMLX-path>
DSV4_MODEL=<checkpoint-path>
DSV4_SERVER_EP=<repository-path>/runtime/server_ep2.py
```

Keep `backend` set to `jaccl`. The example hostfile documents the required
shape and fields without containing a usable network topology.

### 4. Launch

From the rank-0 setup shell:

```bash
$OMLX/venv/bin/mlx.launch --verbose \
  --hostfile "$REPO/examples/hosts-jaccl.json" \
  --cwd "$OMLX" \
  --python "$OMLX/venv/bin/python" \
  "$REPO/examples/run-rank-server.sh"
```

Publish only the rank-0 API endpoint. Check readiness and the model list before
sending client traffic:

```bash
export BASE_URL=https://<rank-0-api-host>/v1
curl -fsS "$BASE_URL/models"
```

A minimal streaming request is:

```bash
curl -N "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain JACCL in one sentence."}],
    "stream": true,
    "max_tokens": 64
  }'
```

## Benchmarking

The repository includes a small OpenAI-compatible streaming benchmark:

```bash
$PYTHON "$REPO/scripts/benchmark_openai.py" \
  --base-url "$BASE_URL" \
  --model deepseek-v4-flash
```

For comparable measurements, keep the model, prompt, warm-up, request ordering,
and stream accounting constant. Exercise ordinary generation, reasoning, JSON,
and tool calls separately. Do not publish credentials, host addresses, or local
filesystem paths in measurement receipts.

## Configuration and source map

- [`examples/hosts-jaccl.json`](examples/hosts-jaccl.json) — sanitized JACCL
  topology shape.
- [`examples/run-rank-server.sh`](examples/run-rank-server.sh) — serving
  defaults and six-slot launch parameters.
- [`runtime/server_ep2.py`](runtime/server_ep2.py) — EP2 server integration.
- [`runtime/openai_router.py`](runtime/openai_router.py) — optional API router.
- [`patches/`](patches/) — idempotent source and native-artifact patchers.
- [`tests/`](tests/) — rollback and API tests.
- [`results/`](results/) — machine-readable measurements.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — placement, collectives, and
  runtime design.
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — measurement methodology.
- [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) — quantization boundaries.
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — upstream provenance.

## License and checkpoint terms

Apache-2.0 for this repository. The repository contains integration code and
patches, not DeepSeek weights. Follow the checkpoint's own license and terms
separately.
