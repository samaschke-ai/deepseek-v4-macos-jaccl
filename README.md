# DeepSeek V4 Flash Vision Q4 on two Apple Silicon Macs

A reproducible **llama.cpp Metal + layer-split RPC** recipe for
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`, `UD-Q4_K_XL`, with the tested BF16
Vision projector. Two M4 Max Macs with 128 GiB unified memory each, external
SSDs and a private Thunderbolt RDMA link are the reference hardware.

The repository name is historical: **JACCL, oMLX expert parallelism, DSpark/MTP,
and tensor-parallel launch variants are not supported recipes here.** Their
source/tests/receipts are preserved together in [archive/omlx](archive/omlx).
The corrected TP experiment was much slower; DSpark did not establish a gain.
This is layer placement (`1,1`), not tensor parallelism.

## Scope and evidence

The recorded Q4 service passed real text, image (left red/right blue), and four
concurrent marker requests after restoration. This publication was prepared
without restarting that service or running new inference. Offline recipe tests
and source/build checks are separate from those historical serving receipts.
A full-context allocation is not a million-token workload benchmark, and reboot
recovery is **not tested or automated**. Raw chat content can begin with
`</think>`; clients must handle this known parser limitation explicitly.

## 1. Prerequisites on both Macs

- Reference OS: macOS 26.6.2; Apple RPC RDMA requires macOS 26.2 or newer,
  supported hardware, enabled RDMA and an SDK with `librdma`.
- Full Xcode and its Metal toolchain, Git, CMake, Python 3.9+; select the Xcode
  developer directory if `xcrun metal` is unavailable. Install the Metal
  toolchain via Xcode if requested. `brew install cmake python git` is sufficient
  for CLI dependencies, not a substitute for the Metal compiler.
- Direct Thunderbolt cable, separately configured private static IPv4 addresses,
  RDMA enabled on both hosts. Follow Apple's RDMA setup for your OS; enabling it
  can require Recovery and a reboot. This recipe never changes it automatically.
- Keep ordinary clients and untrusted machines off the RPC subnet. **RPC has no
  authentication and is not a safe public service.** Firewall the worker port to
  the coordinator's Thunderbolt address. Private IP validation is not a firewall.
- Reserve external SSD capacity: coordinator weights total **155,095,288,672
  bytes** plus 934,455,392-byte projector, download cache and build space. Worker
  `-c` maintains its own RPC tensor cache; allow ample external SSD space too.
  Both nodes need memory for their weights, KV, compute and OS; total RAM alone
  is not an allocation proof. Reference wired limit was 114688 MiB; inspect
  `sysctl iogpu.wired_limit_mb` and memory pressure rather than changing it blindly.

Use paths appropriate to each machine; they need not be identical:

```bash
mkdir -p "$HOME/ai"
export REPO="$HOME/ai/deepseek-v4-macos-jaccl"
git clone https://github.com/samaschke-ai/deepseek-v4-macos-jaccl.git "$REPO"
export SOURCE="$HOME/ai/llama-vision-q4"
export RUNTIME="$SOURCE/build-recipe"
export MODELS="/Volumes/YourExternalSSD/models/deepseek-vision"
export LLAMA_CACHE="/Volumes/YourExternalSSD/llama-cache"
export RPC_HOST="192.168.0.2"  # REPLACE with worker's private Thunderbolt IPv4
mkdir -p "$LLAMA_CACHE"
```

## 2. Rebuild the exact patched source on both Macs

Do not substitute an arbitrary latest llama.cpp or copy only a server binary.
The public base is **95ef7fc16054e63b427a3ef00188e055ef7586d8**, plus the complete
[patch](patches/metal-rpc-scheduler.patch). It includes RPC connection/lifetime
fixes, the Metal MoE matrix crossover (`ne21 >= 8`), bounded prefill scheduling,
and their C++ regression sources (including originally untracked tests).
Both ends must use the same patched source and build configuration.

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$SOURCE"
git -C "$SOURCE" checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
test -z "$(git -C "$SOURCE" status --porcelain)"
shasum -a 256 "$REPO/patches/metal-rpc-scheduler.patch"
# Expected: d3441f8d065300aa41090026e4db6ae4040ce1b268a1f3345bfcb9a5ae69ccdb
git -C "$SOURCE" apply --check "$REPO/patches/metal-rpc-scheduler.patch"
git -C "$SOURCE" apply "$REPO/patches/metal-rpc-scheduler.patch"
cmake -S "$SOURCE" -B "$RUNTIME" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=ON -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF
cmake --build "$RUNTIME" --parallel 6 --target llama-server ggml-rpc-server \
  test-server-prefill-budget test-server-prefill-fairness \
  test-rpc-connect-failure test-rpc-reconnect
ctest --test-dir "$RUNTIME" --output-on-failure -R 'test-server-prefill-(budget|fairness)'
"$RUNTIME/bin/llama-server" --version
"$RUNTIME/bin/ggml-rpc-server" --help
```

CMake must report `RDMA transport enabled (Apple RDMA-over-Thunderbolt, UC)`.
Record the base SHA, patch SHA and CMake cache on each node. Binary hashes may
differ with toolchain/link paths; keep the build tree's dependent libraries, not
just its executables. See [runtime provenance](docs/RUNTIME.md).

## 3. Download and verify exact weights (coordinator only)

Use the modern Hugging Face CLI and Xet transport. Repeating the same command
resumes/reuses its cache; do not delete `.cache/huggingface`, force download, or
use `HF_HUB_DISABLE_XET=1`. The worker receives tensors via RPC, not a second
model checkout.

```bash
python3 -m venv "$HOME/.venvs/hf-download"
source "$HOME/.venvs/hf-download/bin/activate"
python -m pip install 'huggingface_hub[hf_xet]'
unset HF_HUB_DISABLE_XET
hf download unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF \
  --revision b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0 \
  --include 'UD-Q4_K_XL/*.gguf' --local-dir "$MODELS"
hf download ggml-org/DeepSeek-V4-Flash-Vision-Exp-GGUF \
  mmproj-DeepSeek-V4-Flash-Vision-Exp-BF16.gguf \
  --revision 5d829f12416e76edf53500c54cedf72c94e8c2e8 --local-dir "$MODELS"
python3 "$REPO/scripts/preflight.py" --models "$MODELS"
```

Require **six verified files**. [model-manifest.json](model-manifest.json) pins
all sizes and SHA-256 values. The first shard is mostly metadata: its presence
is not a completed model. The BF16 projector comes from a **different repository
and revision**; Unsloth's `mmproj-BF16.gguf` is not the tested file.

## 4. Start worker, then coordinator

First inspect existing processes, ports and memory on both hosts. Do not start
alongside another loaded model. The launch helper refuses existing same-name
processes and occupied ports; this is a best-effort guard, not a multi-user
lifecycle controller. Allow only one operator to launch. Resolve live traffic
and stop the old service with its owner before proceeding.

Worker (its own terminal):

```bash
python3 "$REPO/scripts/recipe.py" worker --runtime "$RUNTIME" \
  --rpc-host "$RPC_HOST" --dry-run
python3 "$REPO/scripts/recipe.py" worker --runtime "$RUNTIME" --rpc-host "$RPC_HOST"
```

On the worker, inspect its listener without opening an RPC connection:

```bash
lsof -nP -iTCP:50052 -sTCP:LISTEN
```

Never probe the RPC port with curl, nc or a generic health check. Coordinator
(separate terminal):

```bash
python3 "$REPO/scripts/recipe.py" coordinator --runtime "$RUNTIME" \
  --models "$MODELS" --rpc-host "$RPC_HOST" --dry-run
python3 "$REPO/scripts/recipe.py" coordinator --runtime "$RUNTIME" \
  --models "$MODELS" --rpc-host "$RPC_HOST"
```

Startup can take several minutes. Defaults reproduce: `MTL0,RPC0`, layer `1,1`,
context 1048576, four slots, f16 K/V, flash on, batch 2048, ubatch 1024,
prefill-budget 1024, spec none, cache-ram 0, load-mode none, alias
`deepseek-v4-flash`, port 8241. HTTP binds **loopback**, intentionally safer than
the recorded production wildcard. Use an SSH tunnel or authenticated private
reverse proxy for clients; none is provisioned here.

For detachment use `screen -S vision-worker` / `screen -S vision-coordinator`,
run the same foreground command inside, then Ctrl-A D. Reattach with
`screen -r vision-worker`. No nested `nohup`, orphan-producing shell pipelines,
auto-restart or reboot-persistence claim. `LLAMA_CACHE` must be exported in the
worker's launch shell; RPC adds `rpc/` beneath it.

## 5. Verify health, identity, slots, transport, text and Vision

Read-only checks (do not generate tokens):

```bash
python3 "$REPO/scripts/preflight.py" --url http://127.0.0.1:8241
sysctl iogpu.wired_limit_mb vm.swapusage
memory_pressure
```

Require health `ok`, alias `deepseek-v4-flash` and four idle slots. Also inspect
the actual process arguments/model load line; an alias alone cannot identify a
quant. Context is shared according to server slot allocation: inspect `/slots`
for actual per-slot capacity, do not advertise 1M for each of four requests.

An RDMA-enabled binary or a Thunderbolt IP does **not** prove RDMA negotiation.
Require `RDMA(Apple/UC) activated` in the new coordinator/worker connection logs;
this marker was verified in the recorded production worker log. A TCP bootstrap
socket is expected. Unset `GGML_RPC_NO_RDMA` on both nodes (even a value of `0`
disables negotiation because this variable is presence-based). Provider logs
are supplementary diagnostics, not a substitute for the application marker:

```bash
log show --last 10m --info --debug --style compact \
  --predicate 'subsystem == "com.apple.AppleThunderboltRDMA"'
```

Correlate active serving PIDs and QP transitions through RTS. Provider activation
is not a payload throughput benchmark. If activation cannot be established,
report working RPC with **RDMA unverified**, not RDMA performance.

Only when authorized to send new inference, run the semantic checks below.
They generate a deterministic red/blue image locally and submit it as image
bytes, not a textual description. Require the left/right colors to match and
inspect the returned raw content:

```bash
python3 "$REPO/scripts/verify.py" --url http://127.0.0.1:8241 --inference
```

This is a smoke test, not a quality benchmark. It also sends four concurrent
marker requests. Keep raw `</think>` handling visible; do not claim clean parser
output or tool-call correctness from these tests.

## 6. Stop safely / troubleshooting

Drain clients first. Read `/slots` repeatedly for at least 120 seconds with no
processing and unchanged task IDs, check once more immediately, and coordinate
with callers (a tool-call gap is not completion). Stop **coordinator first** with
Ctrl-C in its terminal/screen, verify it exited, then stop the worker. Never
`pkill` all model processes, delete model files, or kill the worker under active
RPC traffic. Recover by starting the matching worker then coordinator; reboot
recovery and automatic failover remain untested.

- Missing `--prefill-budget`: wrong binary or unapplied patch.
- Metal compiler failure: select/install full Xcode Metal tooling; do not silently
  disable Metal and claim this recipe passed.
- Allocation failure: inspect per-node weight/KV/compute allocations, pressure,
  swap deltas, wired limit and competing apps. Lowering precision/context is a
  different unverified configuration, not a demonstrated fix.
- RPC malformed response/disconnect: check both patched builds, worker cache disk
  space, private addressing and negotiated transport. Stop coordinator before
  restarting worker. An HTTP listener after backend errors is not a pass.
- Bad image output: verify exact BF16 SHA and a real image request; advertised
  multimodal support or projector loading alone is insufficient.

## Measurements and repository gates

Median of three repeats per workload, recorded with the same patched runtime,
f16 KV, four slots and full configured context:

| Variant | Cold 32768 prefill tok/s | Warm decode tok/s | 4x8192 end-to-end output tok/s |
|---|---:|---:|---:|
| Q2 distributed | 348.608 | 22.649 | 7.675 |
| Q3 distributed | 348.045 | 22.168 | 7.712 |
| Q4 distributed | 356.140 | 22.176 | 7.767 |
| Q2 single | 281.399 | 25.110 | 7.246 |

[Sanitized receipts](results/vision-quant-comparison.json) retain 72 requests,
input/prompt hashes, server timings, cache counts, group wall times and summary.
Cold requests reused zero tokens; warm requests reused **32764**, reevaluating
four tail tokens; every request generated 256 tokens. Concurrent throughput is
1024 output tokens / group wall seconds, **including prefill**, not aggregate
decode speed. Q3's revision differs from Q2/Q4: quantization is **not the sole
variable**. Q4 means less quantization, not measured superior answer quality;
no quality benchmark was performed. These workloads do not exercise 1M tokens.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_results.py
python3 -m compileall -q scripts tests
 git diff --check
```

Historical oMLX tests remain intact under the retired component, with their
original external runtime dependencies. They are not substitutes for the active
recipe gate or a new live serving test.
