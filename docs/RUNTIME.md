# Runtime provenance and operational boundaries

Public source: https://github.com/ggml-org/llama.cpp at
`95ef7fc16054e63b427a3ef00188e055ef7586d8`.
Apply `patches/metal-rpc-scheduler.patch`, SHA-256
`d3441f8d065300aa41090026e4db6ae4040ce1b268a1f3345bfcb9a5ae69ccdb`.
The patch reproduces the production tracked diff and four additional regression
sources. No unpublished source commit or machine-local binary path is required.
The full patch is the compatibility unit; do not mix ends from different trees.

Changes cover RPC shared connection ownership/lifecycle, disconnect/reconnect
handling and regression tests; Metal MoE matrix crossover from 32 to 8; and
`--prefill-budget`. That flag is opt-in (default zero), rejects negative values,
and limits text prefill while generation is active. It does not bound Vision
encoding or promise a wall-clock fairness deadline.

## SDK and link dependencies

Use full recent Xcode if Command Line Tools lack RDMA or Metal components:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun --find metal
SDK="$(xcrun --sdk macosx --show-sdk-path)"
test -f "$SDK/usr/include/infiniband/verbs.h"
test -f "$SDK/usr/lib/librdma.tbd"
rdma_ctl status
unset GGML_RPC_NO_RDMA
```

`rdma_ctl status` is read-only. RDMA activation/Recovery is an administrator
prerequisite, not something this recipe runs. CMake requires RDMA, rather than
silently accepting TCP-only. Production used Release, Metal embedded library,
RPC/RDMA ON, BLAS OFF and explicit `-lrdma` executable/shared link flags. If the
SDK weak-link configuration fails to resolve verbs, reconfigure with:

```bash
cmake -S "$SOURCE" -B "$RUNTIME" \
  -DCMAKE_EXE_LINKER_FLAGS=-lrdma -DCMAKE_SHARED_LINKER_FLAGS=-lrdma
```

HTTP build dependencies include OpenSSL when detected (`brew install openssl@3`).
Server multimodal support links `libmtmd`; do not strip it. Inspect dependent
libraries and rpaths after building or relocating:

```bash
otool -L "$RUNTIME/bin/llama-server"
otool -L "$RUNTIME/bin/ggml-rpc-server"
otool -L "$RUNTIME/bin/libggml-rpc.dylib"
```

Production node binary hashes differ, including signing/relocation differences;
this recipe claims matching source/configuration, not byte-identical production
binaries. Keep the complete build bundle and required libraries on each node.

## API access and identity

For a client machine, open an authenticated SSH tunnel (replace account/host):

```bash
ssh -N -L 127.0.0.1:8241:127.0.0.1:8241 user@coordinator.example.invalid
```

Send HTTP to local port 8241; never tunnel or probe the RPC port as an HTTP API.
The alias is stable across quants and cannot prove model identity. On the
coordinator inspect the running command, correlate its PID with the terminal,
and check the model-loading log against the verified manifest:

```bash
ps -ww -p "$(pgrep -x llama-server)" -o pid=,command=
```

Slots must be four idle entries with `n_ctx=262144` each. The preflight enforces
that capacity, not merely the array length. This does not prove caller sessions
are complete: drain with their owner before stopping anything.

## Publication build verification

A fresh public clone at the pinned SHA accepted the patch with `git apply
--check`. On the publication workstation, AppleClang 21.0.0.21000101 compiled
`llama-server`, `ggml-rpc-server`, both prefill tests and both RPC regression
executables with Release / Metal / RPC / RDMA enabled and BLAS disabled.
CMake's SDK weak RDMA linking succeeded without additional `-lrdma` flags on
this machine. Both prefill CTest cases passed; `llama-server --version` and
`ggml-rpc-server --help` executed successfully. This is compile/CLI evidence,
not an inference or negotiated-RDMA run. The independent review may report
additional isolated RPC tests separately.

## Local-only regression tests

The scheduler C++ tests do not load a model. RPC tests start isolated temporary
workers and should run only on a test machine, never pointing at production:

```bash
ctest --test-dir "$RUNTIME" --output-on-failure \
  -R 'test-(server-prefill-(budget|fairness)|rpc-(connect-failure|reconnect))'
```

A successful offline gate is not a new Vision/RDMA model-load validation. The
recorded serving text/image/concurrency evidence remains separate. No reboot
recovery, quality superiority, unrestricted public serving or automatic failover
is claimed.
