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
| Shared-expert tensor sharding | 329.51 tok/s prefill; 23.46 tok/s decode | Reject |
| Native block kernel for 36-route verify | 29.88 tok/s decode | Reject |

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
