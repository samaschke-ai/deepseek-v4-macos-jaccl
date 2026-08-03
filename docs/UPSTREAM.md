# Upstream plan

Keep changes separated by concern so they can be reviewed and benchmarked
independently.

## 1. EP-aware native block planning

Target: `omlx/patches/deepseek_v4/switch_layers.py`

- Add optional route ownership masks.
- Sort active routes before inactive routes per expert.
- Trim inactive suffixes in the block builder.
- Explicitly zero skipped output rows before weighted reduction.

This is a prefill optimization and preserves fixed distributed shapes.

## 2. Exact arbitrary PoolingCache rollback

Targets:

- `omlx/patches/deepseek_v4/cache_extras.py`
- `omlx/patches/mlx_lm_mtp/deepseek_v4_model.py`

Include the 40-case matrix test covering single/batched caches, all ratio-4
remainders, and every rejection point in a five-draft verify.

## 3. Shape-specific native fallback

Target: `omlx/patches/deepseek_v4/switch_layers.py`

Catch only the native extension's `unsupported shape` error and fall back to
`gather_qmm`. Unrelated exceptions must continue to propagate.

## 4. Generation liveness

Target: mlx-lm/oMLX server worker lifecycle.

An uncaught generation-thread failure must make readiness fail or terminate the
server process. `/v1/models` is metadata health, not generation health.
Deployment-specific readiness-file removal belongs in the service wrapper.

## 5. Small-M EP-aware MXFP4 verification kernel — implemented

Five-position verification has 36 routed rows. Generic gathered QMM computes
remote routes; the existing native block path is slower at this size because
sorting and block-plan overhead dominate.

The implementation is available on:

- branch: [`samaschke-ai/omlx:perf/deepseek-masked-small-m-mxfp4-main`](https://github.com/samaschke-ai/omlx/tree/perf/deepseek-masked-small-m-mxfp4-main)
- wheel: [`deepseek-ep-masked-qmv-v1`](https://github.com/samaschke-ai/omlx/releases/tag/deepseek-ep-masked-qmv-v1)

It dispatches MLX's exact MXFP4 QMV reduction over fixed route rows and checks
the EP ownership mask before loading expert weights. It requires no sorting,
data-dependent shapes, or host synchronization. Focused FP16/BF16 tests are
bit-identical to generic QMM and masked rows are exactly zero.

Measured results:

- real 36-route projection: 0.7172 → 0.4332 ms (1.655×);
- matched visible decode: 36.60 → 40.36 tok/s (+10.3%);
- acceptance unchanged at 190/235 (80.9%), 3.76 tokens/cycle;
- per-depth counts unchanged: 59/68, 45/59, 36/45, 27/36, 23/27.

A fused gate+up variant was correct but slower than two masked calls (0.7101
versus 0.7038 ms) and was removed in a follow-up commit. Further work should
focus on replicated trunk/shared-expert cost and collective scheduling rather
than reintroducing block planning for small M.
