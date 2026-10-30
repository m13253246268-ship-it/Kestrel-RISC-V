[English](README.en.md) | **简体中文**

# vLLM-Kestrel — 边云协同 · 精确注意力 LLM 推理引擎

> Axiom-Guided Exact-Attention edge LLM runtime

纯 C 实现的自包含 LLM 推理引擎，面向板端（aarch64-Linux / RISC-V）做一级目标，提供
OpenAI 兼容的 HTTP 服务、自研 VQF 量化模型格式、连续批调度、KV 分层驻留、MoE 专家并行、
多模态（Qwen3-VL / YOLO）与嵌入检索等能力。

本仓库当前处于**整理开源资料**阶段：板端（RISC-V）暂时不可用，但
**x86-64 主机仍可编译 + 数值自检**，RISC-V 也可交叉编译验证「可编译 + 自检通过」。
x86 构建只用于「可编译 + 与 ARM 同口径的回归」，**不作为性能基准**（详见文末「诚实说明」）。

---

## 1. 目录结构

```
kestrel_vllm_riscv/
├── include/                  # 头文件（common / core / serve / model / media）
├── src/                      # 引擎源码（main.c 为入口）
│   ├── core/                 # 注意力 / 调度 / transformer / matmul / TP / EP / L3 / KV
│   ├── model/                # safetensors 解析 / VQF / 分词器 / vision / laya / yolo
│   ├── serve/                # HTTP 服务 / 连续批 / 管理 / 会话 / 审计 / 规则包 / 内嵌 Web
│   ├── media/                # RTSP / H264 / MP4 视频源
│   └── common/               # 设备探测 / 国密（SM2/SM3/SM4）
├── tools/
│   ├── build/                # 构建 + 回归闸门脚本（x64 / riscv）
│   ├── client/               # vllm_client.py（OpenAI 兼容客户端，零依赖）
│   ├── preproc/              # 数据预处理脚本（自研层链驱动用，见其 README）
│   ├── bench/                # 内核级探针 / 基准脚本
│   ├── yolo/                 # YOLO 量化 / 基准工具
│   └── ops/                  # vllm_mgr.py 运维脚本
├── vqf_convert/              # 独立模型转换工具：safetensors -> VQF
│   ├── build.bat / build.sh
│   └── vqf_conv(.exe)        # 转换产物
└── build-riscv/              # RISC-V 构建产物（默认输出目录）
```

模型权重**不随源码分发**，统一放在仓库外（见「模型准备」）。

---

## 2. 模型格式：VQF

引擎**只加载自研 VQF 格式**，原始 safetensors / GGUF 已整体移出引擎，由独立工具
`vqf_convert/`（`vqf_conv`）负责转换。

一个**可加载的模型目录**需包含：

| 文件 | 必需 | 说明 |
|---|---|---|
| `model.vqf` | 是 | 量化权重（或 `--model` 直接指向 `.vqf` 单文件） |
| `vocab.bin` | 是 | 二进制词表（由 `tools/build/build_vocab_bin.py` 生成） |
| `config.json` | 是 | 特殊 token ID（eos / im_end 等） |
| `merges.bin` | 否 | BPE 合并表；缺失回退贪心最长前缀 |

---

## 3. 模型准备

> 各类模型（Dense / MoE / 多模态 / Embedding / Laya / YOLO）的**适配方式**详见
> [docs/模型适配.md](docs/模型适配.md)。

### 3.1 获取原始模型

以 HF 模型目录为例（需 `config.json` + `*.safetensors` + 分词器文件）。例如：

- `Qwen2.5-0.5B`（dense，小）
- `Qwen3-30B-A3B`（MoE，大）
- `Qwen3-VL-2B-Instruct`（多模态）

### 3.2 转换 safetensors → VQF

```bash
# Windows（MinGW-w64 gcc）
cd vqf_convert && build.bat
# Linux
cd vqf_convert && ./build.sh
# 产物：vqf_conv(.exe)
```

转换命令：

```bash
vqf_conv --model <safetensors 目录> --convert-vqf <输出 .vqf> \
         [--stream] [--wmode dual|q4|q8|q4i|g256]
```

- `--wmode`：`dual`(默认，Q4+Q8 双份) / `q4` / `q8` / `q4i`(预解包 int8) / `g256`
- `--stream`：低峰值内存转换（量化缓冲不驻留 RAM，适合 30B 级大模型）
- 环境变量：
  - `VLLM_VQF_KEY=<口令>` 输出 SM4-CTR + HMAC-SM3 加密文件
  - `VLLM_VQF_SIGN_PRIV=<64hex>` 内嵌 SM2 供应链签名
  - `VLLM_WMODE=q4|q8|q4i|g256` 等价于 `--wmode`

转换完成后，把 `model.vqf` 与 `vocab.bin`、`config.json`（可选 `merges.bin`）放在同一目录。

---

## 4. 编译

### 4.1 x86-64 主机（Windows + MinGW-w64）—— 当前可验证路径

依赖：MinGW-w64 x86_64 GCC（含 winpthreads、winsock2）。

```powershell
# 仅构建
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1
# 指定 gcc 与输出目录
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 `
    -Gcc D:\tools\mingw64\bin\gcc.exe -OutDir .\build-x64
# 产物：build-x64\vllm_kestrel_x64.exe
```

**回归闸门**（构建 + L3/Sparse 自检计数）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
```

期望输出：`--test-l3 16 PASS / 0 FAIL`、`--test-sparse 9/9`，退出码 0。

### 4.2 RISC-V 64（Linux）

```bash
# 板上原生构建（自动探测 gcc）
./tools/build/build_riscv.sh

# x86 主机交叉编译（需 riscv64-linux-gnu-gcc）
CC=riscv64-linux-gnu-gcc ./tools/build/build_riscv.sh
# 产物：build-riscv/vllm_kestrel
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_RISCV_CC` | 自动探测 | 指定编译器 |
| `VLLM_RISCV_MARCH` | `rv64gcv` | `-march`；无向量板改 `rv64gc` |
| `VLLM_RISCV_OUTDIR` | `<repo>/build-riscv` | 产物目录 |
| `VLLM_RISCV_EXTRA` | 空 | 追加编译/链接参数 |

**回归闸门**：

```bash
./tools/build/check_riscv.sh          # 板上原生跑（含自检）
./tools/build/check_riscv.sh --build  # 只构建（交叉编译主机无 qemu 时）
```

确定性红线：`-ffp-contract=off` 必须保留，禁止编译器把 `mul+add` 收缩成 `fma`，
否则与参考实现的舍入链不同序，位级对拍失效。

---

## 5. 启动

### 5.1 启动 HTTP 服务（推荐）

```bash
# Linux / 板端
./build-riscv/vllm_kestrel --serve --port 8080 --model <模型目录> --threads 8

# Windows
.\build-x64\vllm_kestrel_x64.exe --serve --port 8080 --model <模型目录> --threads 8
```

- `--model` 指向**含 `model.vqf` 的目录**，或直接指向 `.vqf` 单文件
- `--threads N`：worker 线程数（1–32）
- 不带任何参数直接运行 = 启动服务（端口 8080，通过管理页手动加载模型）
- `--serve`（不带 `--port`）= 交互式输入端口号

服务启动后打印可用入口：

```
[SERVE] OpenAI-compatible API listening on http://0.0.0.0:8080
        管理入口 /admin/   对话入口 /chat   审查入口 /review
        GET  /v1/models      GET /health
        POST /v1/chat/completions  (stream=true SSE supported)
        POST /v1/completions       POST /v1/embeddings
```

### 5.2 常用服务参数

| 参数 | 说明 |
|---|---|
| `--model <dir\|.vqf>` | 模型目录或 VQF 单文件 |
| `--model-id <id>` | 模型标识（默认 `qwen3-vl-8b`） |
| `--port <N>` | 显式端口 |
| `--threads <N>` | worker 线程 |
| `--wmode <m>` | `dual`/`q4`/`q4i`/`q8`/`q2mix`/`g256` |
| `--auto-load` | 启动即加载模型 |
| `--no-auto-load` | 启动不加载（默认），经管理页手动加载 |
| `--load-on-use` | 首个请求到达时自动加载 |
| `--auto-unload <s>` | 空闲 N 秒自动卸载 |
| `--batch-max <N>` | 连续批调度上限 |
| `--max-queued <N>` | 排队上限 |
| `--min-free-mb <N>` | 保留空闲内存阈值 |
| `--spec` / `--spec-k <N>` | n-gram 投机解码 |

### 5.3 离线 / 自检模式（无需模型）

```bash
vllm_kestrel --test-l3        # L3 Q4 内核 + 磁盘驱逐自检（期望 16 PASS）
vllm_kestrel --test-sparse    # 稀疏注意力自检（期望 9 PASS）
```

不带任何参数运行时，会执行公理自测套件后进入服务（见 `main.c`）。

### 5.4 其它模式（按需）

```bash
--embed "文本" / --embed-file <f> / --embed-stdin   # 句子向量
--yolo-model <vqf> --yolo-image <图>                 # 单图检测
--laya-pack <目录> --laya-cases <f>                  # Laya 决策模型
--longctx-quality --model <vqf>                      # 长上下文捞针
```

---

## 6. 使用（客户端）

`tools/client/vllm_client.py` 是零第三方依赖的 OpenAI 兼容客户端（Python 3.6+ 可跑）。

```bash
python tools/client/vllm_client.py --url http://127.0.0.1:8080 health
python tools/client/vllm_client.py --url http://127.0.0.1:8080 models
python tools/client/vllm_client.py chat "你好" --stream --max-tokens 128
python tools/client/vllm_client.py chat --system "你是一个助手" "介绍一下自己"
python tools/client/vllm_client.py complete "The capital of France is"
python tools/client/vllm_client.py chat --image photo.jpg "这张图里有什么？" --stream
echo "你好" | python tools/client/vllm_client.py chat
```

也可用任意 OpenAI 兼容客户端（如 `curl`）直接调用：

```bash
curl http://127.0.0.1:8080/v1/models
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

---

## 7. 诚实说明（当前边界）

1. **x86-64 构建只用于「可编译 + 自检与 ARM 同口径」**，不作为性能基准。开放项：
   x64 偶发整请求卡死；L3 Q4 点积在 x86 无 AVX2 内核（走标量回退，数值对、性能无意义）。
2. **板端（RISC-V）当前不可用**。RISC-V 侧可交叉编译验证「可编译 + 自检通过」，
   热内核为 RVV 档位 3，`VLLM_RVV=0` 可随时退回标量参考路径（逃生舱）；收益只在板端实测界定。
3. **回归闸门只证明「能编译 + 自检通过 + 跨架构同计数」**，不构成性能结论。
4. 模型转换、加密签名、位级对拍均有确定性红线（`-ffp-contract=off`、FNV/HMAC 同口径等），
   改动构建参数前请先读对应脚本头部的注释。

更多性能 / 适配细节见仓库根目录的 `arm记录/`、`mcu参考/` 系列文档与
`tools/preproc/README.md`。
