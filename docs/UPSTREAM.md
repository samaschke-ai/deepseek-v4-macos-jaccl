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

## 5. Small-M EP-aware MXFP4 verification kernel

This is the remaining performance project. Five-position verification has 36
routed rows. Generic gathered QMM computes remote routes; the existing native
block path is slower at this size because planning/launch overhead dominates.
A useful upstream kernel must:

- fuse route compaction/block discovery with gate+up execution;
- skip remote routes without host synchronization or dynamic shapes;
- preserve decode-identical reduction order so DSpark acceptance does not fall;
- support M=2..6 and both BF16/FP16 inputs;
- benchmark the full target verify, not only isolated GEMM time.

Measured rejected alternatives:

- all-native 36-route verify: 29.88 tok/s;
- native gate+up with generic down: 28.12 tok/s;
- MoE-only tensor partitioning: 34.72 tok/s due acceptance drift;
- four-position verification: 34.88 tok/s;
- partial/full oMLX v0.5.4 grafts onto this EP2 runtime: 28.74/33.28 tok/s.

The accepted generic path reaches 36.45–37.77 tok/s with exact five-position
rollback. Do not upstream a microbenchmark-only change that regresses this
end-to-end result.
