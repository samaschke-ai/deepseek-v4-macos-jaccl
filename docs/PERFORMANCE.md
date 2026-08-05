# Performance results and methodology

## Reference configuration

- Model: official `deepseek-ai/DeepSeek-V4-Flash-0731`
- Runtime: oMLX DeepSeek V4 + integrated DSpark/MTP
- Hosts: two Apple M4 Max systems, 128 GB unified memory each; rank 1 on AC High Power mode
- Transport: MLX JACCL over direct Thunderbolt RDMA
- Parallelism: 128 routed experts per rank; trunk/shared experts/DSpark replicated
- Serving: six prompt slots, six decode slots, dynamic caches
- Prefill step: 512 tokens
- API ceiling: 1,048,576 tokens

## Headline result versus original production

| Metric | Original production | Final accepted | Improvement |
|---|---:|---:|---:|
| 32K prefill | 178.24 tok/s | 287.15 tok/s | **+61.1%** |
| Visible decode | ~25.4 tok/s | 36.45–37.77 tok/s | **+43–49%** |

These original figures predate the remediation stages below. Stage-level rows
use their immediate experiment baselines and should not be substituted for the
full production-to-production comparison.

## 100 tok/s phase checkpoint (2026-08-05)

The active target is at least 100 visible tok/s for one warmed request. The
exact DGX `count300` workload produced 637 completion tokens over 109 cycles,
with 5.84 tokens/cycle, 527/535 accepted speculative tokens (98.5%), and
rank-identical control decisions.

| Configuration | Median visible decode | Median cycle | Decision |
|---|---:|---:|---|
| Canonical EP2, three matched repeats | **60.88 tok/s** | **95.99 ms** | Current reference |
| Pre-attention Q gather, exact through 43 layers | 57.17 tok/s | 102.22 ms | Reject: collective overhead |

At the measured 5.84-token yield, 100 tok/s requires approximately 58.4 ms per
cycle. The pre-attention Q-gather candidate was numerically exact through all
43 layers, but its additional collective per layer outweighed the saved Q
projection work. A post-attention head-gather candidate was exact through eight
layers across compression ratios 0, 4, and 128, then diverged after layer 9
attention and was rejected for correctness. Neither candidate was deployed.
See [`results/deepseek-100tps-phase-20260805.json`](../results/deepseek-100tps-phase-20260805.json)
for the machine-readable receipt.

## Prefill

The benchmark prompt repeats `benchmark context datum` with a unique nonce to
avoid prefix-cache hits. TTFT is measured at the first non-empty model delta;
heartbeats do not count as model output.

| State | Prompt | TTFT | Throughput |
|---|---:|---:|---:|
| Baseline before native EP route masking | 8,217 | 31.715 s | 259.0 tok/s |
| Native EP route masking | 8,217 | 26.343 s | 311.93 tok/s |
| Route mask with rank 1 on 60W supply | 32,799 | 177.442 s | 184.81 tok/s |
| Final, rank 1 negotiating 94W | 32,801 | 114.216 s | 287.15 tok/s |

The 60W run developed 5–9 second late chunks. With the higher-capacity supply,
late chunks stayed mostly around 1.9–2.4 seconds. This is a controlled
system-level A/B observation, not a general claim that adapter metadata alone
proves thermal throttling.

## Decode and DSpark yield

Matched 256-token counting streams separate TTFT from sustained streamed decode.

| Verification | Visible decode | Tokens/cycle | Cycles |
|---|---:|---:|---:|
| Three positions | 30.75 tok/s | 2.42 | 106 |
| Five positions with exact rollback | 36.45–37.77 tok/s | 3.76 | 68 |
| Five positions plus EP masked QMV | 40.36 tok/s | 3.76 | 68 |

The masked QMV result uses an immediate matched baseline of 36.60 tok/s, for a
10.3% gain. Target-backbone telemetry fell from approximately 6.13 seconds to
5.47–5.50 seconds per 256-token run. Acceptance remained 190/235 (80.9%). That
20–25% target was an intermediate milestone; the active objective is now the
single-request 100 tok/s gate described above.

Final per-position acceptance:

| Position | Accepted / drafted |
|---|---:|
| d1 | 59 / 68 |
| d2 | 45 / 59 |
| d3 | 36 / 45 |
| d4 | 27 / 36 |
| d5 | 23 / 27 |

The target backbone completed approximately 11 target verifies/s at depth five.
The remaining gap to high-yield CUDA recipes is therefore a combination of
verify-row kernel efficiency and accepted tokens per cycle—not simply serial
model decode bandwidth.

## Rejected experiments

| Experiment | Result | Decision |
|---|---:|---|
| Prefill step 1,024 after EP fix | 302.31 tok/s at 8K | Retain 512 |
| Prefill step 640 | 310.18 tok/s versus 320.41 at 512 | Reject |
| Shared-expert tensor sharding | 329.51 tok/s prefill; 23.46 tok/s decode | Reject |
| Shape-gated shared-expert sharding | 325.8 tok/s prefill; 32.0 tok/s decode | Reject |
| Native block kernel for 36-route verify | 29.88 tok/s decode | Reject |
| Fused masked gate+up QMV | 0.7101 ms versus 0.7038 ms for two calls | Reject |
| Two-rank JACCL ring | 40.09 tok/s versus 40.36 mesh | Retain mesh |
| Rowwise MTP batching | 40.33 tok/s | Neutral; do not persist |

## Comparison boundary

Two-DGX-Spark reports around 2,488 tok/s prefill at 32K and approximately
55 tok/s mean decode for the official 0731 checkpoint. The platforms are not
compute-equivalent: NVIDIA uses native FP4 Tensor Cores, CUDA graphs, and a
different distributed schedule. This repository reports matched checkpoint and
prompt shapes but does not claim hardware-normalized parity.

## Reproduce

```bash
export OPENAI_API_KEY=...
python scripts/benchmark_openai.py \
  --base-url https://your-endpoint.example/v1 \
  --model your-model-id \
  --prompt-tokens 32768 \
  --max-tokens 128
```

Use server-side tokenizer usage when available. The repeated unit approximates
the target before tokenization; machine-readable receipts record actual usage.
