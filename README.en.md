[简体中文](README.md) | **English**

# vLLM-Kestrel — Edge-Cloud Collaboration · Exact-Attention LLM Runtime

> Axiom-Guided Exact-Attention edge LLM runtime

A self-contained LLM inference engine written in pure C, targeting edge boards
(aarch64-Linux / RISC-V) as the primary platform. It provides an OpenAI-compatible
HTTP server, a proprietary VQF quantized model format, continuous batching,
tiered KV residency, MoE expert parallelism, multimodal support (Qwen3-VL / YOLO),
and embedding retrieval.

The RISC-V runtime has been tested on a SpacemiT K1 board. x86-64 hosts can build
and run numerical regression checks; x86 builds are not used as an edge-performance
benchmark. On-board measurements apply only to their stated model, configuration
and workload (see "Honest Notes" below).

## MiniCPM5-1B: Q4_0 and K1 validation

`OpenBMB/MiniCPM5-1B` is supported, including tokenization, chat formatting and
termination tokens. The validated configuration uses **Q4_0 weights + FP32 KV**,
with F16 embeddings and F32 norms. Model weights are not distributed in this repository.

```bash
# Run from the repository; MODEL_DIR contains original weights and tokenizer files.
./vqf_convert/vqf_conv --model "$MODEL_DIR" \
  --convert-vqf "$MODEL_DIR/model-q4_0.vqf" --wmode q4
# Also prepare vocab.bin, config.json and the applicable merges.bin in that directory.
VLLM_WMODE=q4 VLLM_KV_F32=1 VLLM_KV_NOF32=0 \
VLLM_Q4_INT=1 VLLM_Q4_RVV=3 \
./build-riscv/vllm_kestrel --model "$MODEL_DIR/model-q4_0.vqf" \
  --threads 4 --serve --auto-load --port 8080
```

K1 checks cover short generation, consistency across thread counts and API finish
reasons: reaching `max_tokens` returns `finish_reason="length"`; a normal termination
token returns `"stop"`. Q4_0 is lossy and these checks do not establish equivalence
to the original F32 model.

RVV prefill shares activation quantization across QKV and Gate/Up. Direct vector
conversion of scales removes temporary array accesses while preserving quantization
and floating-point accumulation order. Before/after scale optimization, hashes of
50 full-vocabulary logits vectors and 24 freely generated token IDs match.
Existing register spills have not been eliminated.

Independent K1 measurements on 2026-11-19 used four threads on CPUs 0–3, FP32 KV,
context capacity 512, 32 input tokens and 24 fixed replay steps. Each process warmed
up fully, then cleared logical KV. BAABABBA order provided four samples per version.
Timing covers forward calls only, excluding loading, tokenization, sampling and HTTP.

| Metric | Before scale optimization | After | Mean time reduction |
|---|---:|---:|---:|
| Prefill | 1.075563 s | 0.990086 s | 7.95% |
| Decode, 24 steps | 3.597447 s | 3.587518 s | 0.28% (essentially unchanged) |
| Total forward | 4.673009 s | 4.577604 s | 2.04% |

**These are observations, not a strict acceptance pass.** The measurement script
incorrectly applied the starting-temperature limit of 65°C throughout each run;
one sample also started monitoring at 66°C. All samples were retained. Post-hoc
sensitivity analysis excluding that entire pair gives 7.42% lower prefill time and
1.88% lower total forward time across three pairs. Sampled frequency was always
1.6 GHz, which does not exclude transients between samples. Results do not establish
long-context performance, other-model performance or end-to-end service throughput.

---

## 1. Layout

```
kestrel_vllm_riscv/
├── include/                  # Headers (common / core / serve / model / media)
├── src/                      # Engine sources (entry: main.c)
│   ├── core/                 # Attention / scheduler / transformer / matmul / TP / EP / L3 / KV
│   ├── model/                # safetensors parser / VQF / tokenizer / vision / laya / yolo
│   ├── serve/                # HTTP server / continuous batching / admin / session / audit / rulebook / embedded web
│   ├── media/                # RTSP / H264 / MP4 video sources
│   └── common/               # Device detection / Chinese cryptography (SM2/SM3/SM4)
├── tools/
│   ├── build/                # Build + regression gate scripts (x64 / riscv)
│   ├── client/               # vllm_client.py (OpenAI-compatible client, zero deps)
│   ├── preproc/              # Data preprocessing scripts (for the FHE layer chain, see its README)
│   ├── bench/                # Kernel-level probes / benchmark scripts
│   ├── yolo/                 # YOLO quantization / benchmark tools
│   └── ops/                  # vllm_mgr.py ops script
├── vqf_convert/              # Standalone converter: safetensors -> VQF
│   ├── build.bat / build.sh
│   └── vqf_conv(.exe)        # Converter artifact
└── build-riscv/              # RISC-V build output (default dir)
```

Model weights are **not distributed with the source**; keep them outside the repo
(see "Model Preparation").

---

## 2. Model Format: VQF

The engine **loads only its proprietary VQF format**. Raw safetensors / GGUF have been
moved out of the engine entirely and are handled by the standalone tool `vqf_convert/`
(`vqf_conv`).

A **loadable model directory** must contain:

| File | Required | Description |
|---|---|---|
| `model.vqf` | yes | Quantized weights (or point `--model` directly at a single `.vqf` file) |
| `vocab.bin` | yes | Binary vocabulary (generated by `tools/build/build_vocab_bin.py`) |
| `config.json` | yes | Special token IDs (eos / im_end, etc.) |
| `merges.bin` | no | BPE merge table; falls back to greedy longest-prefix when missing |

---

## 3. Model Preparation

> Per-architecture adaptation notes (Dense / MoE / Multimodal / Embedding / Laya / YOLO)
> are in [docs/模型适配.md](docs/模型适配.md).

### 3.1 Obtain the original model

Use a HuggingFace model directory (`config.json` + `*.safetensors` + tokenizer files).
Examples:

- `Qwen2.5-0.5B` (dense, small)
- `Qwen3-30B-A3B` (MoE, large)
- `Qwen3-VL-2B-Instruct` (multimodal)

### 3.2 Convert safetensors -> VQF

```bash
# Windows (MinGW-w64 gcc)
cd vqf_convert && build.bat
# Linux
cd vqf_convert && ./build.sh
# Artifact: vqf_conv(.exe)
```

Conversion command:

```bash
vqf_conv --model <safetensors dir> --convert-vqf <out .vqf> \
         [--stream] [--wmode dual|q4|q8|q4i|g256]
```

- `--wmode`: `dual` (default, Q4+Q8 both) / `q4` / `q8` / `q4i` (pre-unpacked int8) / `g256`
- `--stream`: low peak-memory conversion (quantization buffers never resident; suits 30B-class models)
- Environment variables:
  - `VLLM_VQF_KEY=<pass>` emit a SM4-CTR + HMAC-SM3 encrypted file
  - `VLLM_VQF_SIGN_PRIV=<64hex>` embed an SM2 supply-chain signature
  - `VLLM_WMODE=q4|q8|q4i|g256` equivalent to `--wmode`

After conversion, place `model.vqf` together with `vocab.bin`, `config.json`
(and optionally `merges.bin`) in the same directory.

---

## 4. Building

### 4.1 x86-64 host (Windows + MinGW-w64) — currently verifiable path

Requires MinGW-w64 x86_64 GCC (with winpthreads and winsock2).

```powershell
# Build only
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1
# Specify gcc and output dir
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 `
    -Gcc D:\tools\mingw64\bin\gcc.exe -OutDir .\build-x64
# Artifact: build-x64\vllm_kestrel_x64.exe
```

**Regression gate** (build + L3/Sparse self-test counts):

```powershell
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
```

Expected: exit code 0, FAIL = 0, and PASS+SKIP equal to the case total (`--test-l3` 17, `--test-sparse` 10). One case per suite is `[SKIP]` depending on ISA: the NEON comparison (no NEON build) and the RVV Q8 PV comparison (no RVV build). Measured: x86-64 L3 16 PASS + 1 SKIP, Sparse 9 PASS + 1 SKIP; rv64gcv L3 16 PASS + 1 SKIP, Sparse 10 PASS + 0 SKIP.

### 4.2 RISC-V 64 (Linux)

```bash
# Native build on board (auto-detects gcc)
./tools/build/build_riscv.sh

# Cross-compile on an x86 host (requires riscv64-linux-gnu-gcc)
CC=riscv64-linux-gnu-gcc ./tools/build/build_riscv.sh
# Artifact: build-riscv/vllm_kestrel
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_RISCV_CC` | auto-detect | Compiler to use |
| `VLLM_RISCV_MARCH` | `rv64gcv` | `-march`; use `rv64gc` for boards without vector |
| `VLLM_RISCV_OUTDIR` | `<repo>/build-riscv` | Output directory |
| `VLLM_RISCV_EXTRA` | empty | Extra compile/link flags |

**Regression gate**:

```bash
./tools/build/check_riscv.sh          # native run on board (includes self-tests)
./tools/build/check_riscv.sh --build  # build only (cross host without qemu)
```

Determinism red line: `-ffp-contract=off` must be preserved — the compiler must not
contract `mul+add` into `fma`, otherwise the rounding chain diverges from the reference
implementation and bit-level comparison becomes invalid.

---

## 5. Running

### 5.1 Start the HTTP server (recommended)

```bash
# Linux / board
./build-riscv/vllm_kestrel --serve --port 8080 --model <model_dir> --threads 8

# Windows
.\build-x64\vllm_kestrel_x64.exe --serve --port 8080 --model <model_dir> --threads 8
```

- `--model` points at a **directory containing `model.vqf`**, or directly at a single `.vqf` file
- `--threads N`: worker threads (1–32)
- Running with no arguments = start the server (port 8080, model loaded manually via the admin page)
- `--serve` (without `--port`) = interactively prompt for the port

On startup the server prints its entry points:

```
[SERVE] OpenAI-compatible API listening on http://0.0.0.0:8080
        admin UI: /admin/   chat UI: /chat   review UI: /review
        GET  /v1/models      GET /health
        POST /v1/chat/completions  (stream=true SSE supported)
        POST /v1/completions       POST /v1/embeddings
```

### 5.2 Common server flags

| Flag | Description |
|---|---|
| `--model <dir\|.vqf>` | Model directory or single VQF file |
| `--model-id <id>` | Advertised model id (default: the model directory basename, e.g. `Qwen3-0.6B-q4`) |
| `--port <N>` | Explicit port |
| `--threads <N>` | Worker threads |
| `--wmode <m>` | `dual`/`q4`/`q4i`/`q8`/`q2mix`/`g256` |
| `--auto-load` | Load the model at startup |
| `--no-auto-load` | Do not load at startup (default); load manually via admin page |
| `--load-on-use` | Load automatically when the first request arrives |
| `--auto-unload <s>` | Unload after N idle seconds |
| `--batch-max <N>` | Continuous batching limit |
| `--max-queued <N>` | Queue limit |
| `--min-free-mb <N>` | Free-memory floor to preserve |
| `--spec` / `--spec-k <N>` | n-gram speculative decoding |

### 5.3 Offline / self-test modes (no model needed)

```bash
vllm_kestrel --test-l3        # L3 Q4 kernel + disk eviction self-test (16 PASS + 1 SKIP)
vllm_kestrel --test-sparse    # Sparse attention self-test (9 PASS + 1 SKIP on x86, 10 PASS on RVV)
```

Running with no arguments executes the axiom self-test suite and then enters the server
(see `main.c`).

### 5.4 Other modes (on demand)

```bash
--embed "text" / --embed-file <f> / --embed-stdin    # sentence embeddings
--yolo-model <vqf> --yolo-image <img>                # single-image detection
--laya-pack <dir> --laya-cases <f>                   # Laya decision model
--longctx-quality --model <vqf>                      # long-context needle-in-a-haystack
```

---

## 6. Usage (client)

`tools/client/vllm_client.py` is a zero-dependency OpenAI-compatible client
(works on Python 3.6+).

```bash
python tools/client/vllm_client.py --url http://127.0.0.1:8080 health
python tools/client/vllm_client.py --url http://127.0.0.1:8080 models
python tools/client/vllm_client.py chat "hello" --stream --max-tokens 128
python tools/client/vllm_client.py chat --system "You are an assistant" "introduce yourself"
python tools/client/vllm_client.py complete "The capital of France is"
python tools/client/vllm_client.py chat --image photo.jpg "what is in this image?" --stream
echo "hello" | python tools/client/vllm_client.py chat
```

Any OpenAI-compatible client (e.g. `curl`) works too:

```bash
curl http://127.0.0.1:8080/v1/models
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

---

## 7. Honest Notes (current boundaries)

1. **x86-64 builds only prove "compiles + self-tests match ARM"**; they are not a
   performance benchmark. Open items: occasional whole-request hangs on x64; the L3 Q4
   dot product has no AVX2 kernel on x86 (scalar fallback — numerically correct,
   performance meaningless).
2. **RISC-V has been tested on SpacemiT K1.** Cross-compilation alone proves only
   build compatibility; numerical checks and performance measurements require execution.
   The MiniCPM5 results above describe one short-context workload and retain their
   measurement limitations. They are not a general performance guarantee.
3. **The regression gate only proves "builds + self-tests pass + identical counts across
   architectures"**; it is not a performance conclusion.
4. Model conversion, encryption/signing, and bit-level comparison all have determinism
   red lines (`-ffp-contract=off`, identical FNV/HMAC conventions, etc.). Read the header
   comments of the relevant scripts before changing build parameters.
5. **`VLLM_SILU_RVV=1` is an approximation, not bit-level identical.** On RISC-V
   (VLEN=256) it routes the dense-FFN SiLU through an RVV polynomial-exp branch
   (`|err|≈2.5e-7`, ~+6–8% prefill, on by default). Attestation mode (`VLLM_ATTEST=1`)
   forces SiLU back to the scalar `fast_silu` (`expf`) path and ignores `VLLM_SILU_RVV`:
   the approximation can flip a token under greedy argmax, changing the generated text
   and therefore the SM3 digest preimage, so an external recomputation against the
   reference text would no longer match. Attestation thus trades the prefill gain for
   bit-level determinism. (`VLLM_SILU_RVV=0` is the explicit scalar escape hatch.)

For more performance / adaptation details, see the `arm记录/` and `mcu参考/` document
series at the repository root, and `tools/preproc/README.md`.
