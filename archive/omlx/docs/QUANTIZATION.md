# Quantization comparison

The official checkpoint and the two serving recipes use related weight
precision, but different runtime layouts and cache formats.

## Official checkpoint metadata

`config.json` declares dynamic FP8 weights:

- format: E4M3
- scale format: E8M0
- weight block size: 128 × 128
- model compute dtype: BF16

Direct tensor inspection shows that routed expert weights are packed as 4-bit
values with E8M0 scales, while attention and shared-expert tensors use E4M3
weights with E8M0 block scales.

## This Apple/oMLX recipe

- Routed experts: MXFP4, group size 32, native Metal block kernels
- Attention/indexer/shared/MTP projections: MXFP8/FP8 paths where selected by
  oMLX's model quantization map
- Model activations: BF16/FP16 according to the kernel path
- Local and compressed attention caches: BF16-derived runtime arrays

## Two-DGX-Spark recipes

The shared DGX recipes retain the same official weight families:

- Routed experts: MXFP4 through B12X or DeepGEMM W4A16 kernels
- Dense/attention weights: FP8 Tensor Core paths
- MLA KV cache: `nvfp4_ds_mla` in the 1M-context profile
- Historical diagnostic profile: FP8 KV cache

NVFP4 in those repository names primarily describes the MLA KV-cache profile;
it does not mean every model weight was converted from the official checkpoint
to NVFP4. Their own measurements report FP8 versus NVFP4 KV as a capacity
lever rather than a material decode-speed lever.

## Comparison implication

The DGX decode advantage cannot be explained by universally lower-bit model
weights. Both paths execute MXFP4 routed experts and FP8-class dense weights.
The important differences are NVIDIA Tensor Core kernels, CUDA graph/scheduler
behavior, distributed partitioning, KV-cache representation, and speculative
yield per target verification.
