[English](README.en.md) | **简体中文**

# vLLM-Kestrel — 边云协同 · 精确注意力 LLM 推理引擎

> Axiom-Guided Exact-Attention edge LLM runtime

纯 C 实现的自包含 LLM 推理引擎，面向板端（aarch64-Linux / RISC-V）做一级目标，提供
OpenAI 兼容的 HTTP 服务、自研 VQF 量化模型格式、连续批调度、KV 分层驻留、MoE 专家并行、
多模态（Qwen3-VL / YOLO）与嵌入检索等能力。

本仓库当前处于**整理开源资料**阶段：**x86-64 主机可编译 + 数值自检**，RISC-V 可交叉编译
验证「可编译 + 自检通过」；板端实测数据来自云端 K1（8 核 rv64gcv，4 线程绑核 hart0-3），
只对同型号板子成立。x86 构建只用于「可编译 + 与 ARM 同口径的回归」，**不作为性能基准**
（详见文末「诚实说明」）。

本文第 6 节把当前千问（Qwen3 0.6B / 1.7B / 4B）的全部优化开关按**质量档**（会改变数值）
和**快档**（位级一致、只换速度）分开列出，并给出复现脚本。

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
│   ├── yolo/                 # YOLO 量化工具 + 板端实测脚本（见 6.5）
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

## 6. 优化模式：质量档与快档

千问（Qwen3）这条线上有两层开关，**不要混用**：

- **质量档**：改变数值结果，按精度从高到低取舍，包括权重量化档位（转换时定死）、KV 缓存
  精度、稀疏注意力、投机解码的位级约束。
- **快档**：**位级一致**（输出 token 与标量参考路径逐位相同），只换指令与访存方式，可以
  任意叠开。

判定口径：快档一律要求 `TOKIDS` 的**完整 id 序列**与基准一致（脚本需哈希整段，只取首行会
退化成所有模型同值，见 6.5），质量档则必须过 NLL 质量门（见 6.5）。开关的推导与实测注释分别
写在 `src/model/vllm_safetensors.c`（内核）和 `src/model/vqf.c`（权重 / 稀疏驻留）对应位置。

### 6.1 质量档 A：权重量化（转换时决定）

`--wmode` 在 `vqf_conv` 转换时就定死，运行时无法切换。精度从高到低：

| 档 | 转换参数 | 权重精度 | 说明 |
|---|---|---|---|
| `q8` | `--wmode q8` | 8-bit | 全 Q8_0，精度最高、体积最大，是质量基线 |
| `q4i` | `--wmode q4i` | 8-bit（同 q8 量化） | Q8 权重预解包成 int8 行布局，省掉运行期解包 |
| `g256` | `--wmode g256` | 8-bit（256 元素分组） | 分组粒度与 q8 不同，误差分布不同，需实测 NLL |
| `dual` | `--wmode dual`（默认） | 混合 | Q4_0 + Q8_0 **双份**：prefill 走 Q8；decode 在 aarch64 优先 Q4，RISC-V 上 dense 单 token 无 Q4 SIMD，实际也走 Q8 |
| `q4` | `--wmode q4` | 4-bit | 只有 Q4_0，体积最小，dense 模型最低可用档 |
| `q2mix` | 引擎 `--wmode q2mix` | 注意力 Q4 + FFN Q2_1 | 最省内存的档，但 **v1 转换器拒收**（`vqf_conv` 只认前面五种），当前没有可用的产出路径；代码注释记录 dense 上 2-bit FFN 误差不可恢复（`g_q2mix_tail`） |

再强调一次：`dual` 里同时带 Q4/Q8 两份权重，Q4 那份只在**有 Q4 内核的架构**上换来速度，
并不代表精度更高。选档看 NLL，不看体积。

### 6.2 质量档 B：运行时近似开关（默认全关）

| 开关 | 做法 | 代价与适用面 |
|---|---|---|
| `VLLM_KV_F32=1` | KV 缓存不做 int8 | **往上**提精度（默认是 int8 KV），内存翻倍，只用于质量基线/A-B |
| `--kv-q4` | KV 走 Q4 载荷缓存 | 往下省内存，精度低于默认 int8 KV |
| `--sparse-attn [--sparse-k N] [--sparse-ratio R]` | decode 只对 top-k KV 块做注意力 | 长上下文省算力，**是近似**；prefill 历史段也会受影响，必须过 NLL 门 |
| `--spec --spec-k N` + `VLLM_SPEC_FAST=1` | 投机解码放开「位级一致」硬约束 | 输出「几乎一致」但非逐 token 相同；不设 `VLLM_SPEC_FAST` 时 `--spec` 走回滚+重放，位级一致（属快档） |

`VLLM_PREFILL_Q8CACHE=1` 不是运行档位，而是**质量门的测量开关**：把 prefill 历史段的 f32
正典行改写成 q8 反量化值，用来量化「删掉 f32 KV」的 NLL 代价。

### 6.3 快档：位级一致，随便叠

| 快档 | 开启方式 | 默认 | 作用面 |
|---|---|---|---|
| RVV 档位 | `VLLM_RVV=1\|2\|3` | `3` | Q4/Q8 取数与 prefill 跨 token 权重复用；`0` = 标量参考路径（逃生舱） |
| token 分块内核 | `VLLM_Q4_INTB=0` 关闭 | 开 | prefill GEMM 一个工作项 = 32 行 tile × 8 token |
| IME 混合 | `VLLM_Q4_IME=1` | 关 | slot0-3 绑 hart0-3 走 vmadot，其余 slot 走 colbt |
| IME 纯路径 | `VLLM_Q4_IME=2` | 关 | **4 线程生产推荐**；所有单元都走 IME（落到 hart4-7 的线程按 CPU 号自动退回 colbt） |
| IME K 分块 | `VLLM_Q4_IME_KB=n` | `0` | 单元内按 n 块切 K，只改访存局部性 |
| decode vmadot | `VLLM_Q4_VMADOT=1` | 关 | M=1 单 token matvec 走 vmadot |
| decode 新内核 | `VLLM_Q4_DEC=1` | 关 | `rv_q4_dec_rows` 入口（`VMADOT=1` 会隐含打开） |
| Q4 bt4 批内核 | `VLLM_Q4_BT=0` 关闭 | 开 | 每 4 个 token 走 bt4 行内核，余数回退单 token |
| 线程与绑核 | `--threads N` + `taskset -c 0-(N-1)` | 依平台 | 线程池大小与亲和性 |
| 投机解码 | `--spec --spec-k N` | 关 | n-gram 草稿 + 验证，默认回滚重放，位级一致 |
| prefill 走 Q8 | `--prefill-q8` | `dual` 下已为 1 | 仅影响同时带 Q4/Q8 的模型 |

`--warmup` 不加速，只是把首轮页入踢出计时窗口（`--stream-test` 会跑一轮完整 prefill +
8 步 decode 再丢弃），所有性能数字都要求在 `--warmup` 下取。

### 6.4 推荐组合

```bash
# 板端 4 线程（当前生产推荐；8 线程尚未实测，保持默认 VLLM_Q4_IME=0 即可）
taskset -c 0-3 env VLLM_Q4_VMADOT=1 VLLM_Q4_IME=2 \
  VLLM_THREADS=4 OMP_NUM_THREADS=4 \
  ./build-riscv/vllm_kestrel --serve --model <模型目录> --threads 4

# 质量优先：q8 权重 + f32 KV + 不开稀疏注意力
vqf_conv --model <safetensors> --convert-vqf model.vqf --wmode q8
VLLM_KV_F32=1 ./build-riscv/vllm_kestrel --serve --model <模型目录> --threads 4

# 内存优先：q4 权重 + Q4 KV；长上下文再叠 --sparse-attn（需过 NLL 门）
vqf_conv --model <safetensors> --convert-vqf model.vqf --wmode q4
./build-riscv/vllm_kestrel --serve --model <模型目录> --threads 4 --kv-q4 --sparse-attn
```

### 6.5 复现脚本

**构建与自检**（x86 或板端）：

```bash
./tools/build/build_riscv.sh          # 板端原生构建，成功日志出现 [riscv] OK
./tools/build/check_riscv.sh          # 构建 + 自检：--test-l3 16 PASS、--test-sparse 9 PASS
```

**性能采集**（板端，`tools/yolo/` 下是当时实测用的脚本，路径写死为 `/root/kv/...`）：

| 脚本 | 做的事 |
|---|---|
| `tools/yolo/_k1_final4.sh` | 官方 `llama-bench`（Q4_0, `-p 32,128 -n 32 -t 4 -r 3`）vs 本引擎（`VLLM_Q4_IME=2`），3 轮交替、同会话同负载 |
| `tools/yolo/_k1_ime4.sh` | `VLLM_Q4_IME=2` vs `=0` 的 A/B，含 `PREFILL-TIMING` 与 `PREFILL-KERNELS` 分解 |
| `tools/yolo/_k1_p4.sh` | `VLLM_Q4_IME_KB` 的位级 + A/B，附 `VmHWM/RssAnon/RssFile` 峰值 |
| `tools/yolo/_k1_multimodel.sh` | 0.6B / 1.7B / 4B 多模型跑同一口径 |

单臂手工复现（4 线程、ctx 128）：

```bash
taskset -c 0-3 env VLLM_Q4_VMADOT=1 VLLM_Q4_IME=2 VLLM_THREADS=4 OMP_NUM_THREADS=4 \
  ./build-riscv/vllm_kestrel --stream-test --model /root/kv/models/Qwen3-0.6B-col \
  --stream-ctx 128 --stream-n 32 --threads 4 --warmup
```

`--stream-test` 会打印 prefill 用时、逐 token 的 `t=`/`ms=`、稳态 decode 均值，最后一行
`TOKIDS` 用来做位级对拍。

**位级 / 质量门**：

```bash
# 位级：TOKIDS 的完整 id 序列 md5 必须与基线一致（基线见下表）
taskset -c 0-3 env VLLM_Q4_VMADOT=1 VLLM_Q4_IME=2 VLLM_THREADS=4 OMP_NUM_THREADS=4 \
  ./build-riscv/vllm_kestrel --stream-test --model /root/kv/models/Qwen3-0.6B-col \
  --stream-ctx 128 --stream-n 32 --threads 4 --warmup | grep TOKIDS

# 质量：CORPUS_NLL_MEAN 与基准一致（0.6B 3.846915 / 1.7B 4.709014 / 4B 4.799563）
taskset -c 0-3 env VLLM_STREAM_NLL=1 VLLM_Q4_VMADOT=1 VLLM_Q4_IME=2 \
  VLLM_THREADS=4 OMP_NUM_THREADS=4 \
  ./build-riscv/vllm_kestrel --stream-test --model /root/kv/models/Qwen3-0.6B-col \
  --stream-ctx 32 --stream-n 32 --threads 4 --warmup
```

`TOKIDS` 基线（`--stream-n 32`，md5 为 32 个 id 拼成 `a,b,c,...,` 后取前 8 位）：

| 模型 | `--stream-ctx 32` | `--stream-ctx 128` |
|---|---|---|
| Qwen3-0.6B | `d29857f2` | `fb9ac449` |
| Qwen3-1.7B | `2e9bdbdd` | `2e9bdbdd` |
| Qwen3-4B | `8b081e31` | `47dfdb01` |

注意：`VLLM_STREAM_NLL=1` 时不输出 `TOKIDS`（md5 退化为空串的 `d41d8cd9`），位级与质量
两项必须分两次跑。NLL 数值以本机实测为准，上面是 K1 上的参考值。

### 6.6 实测量（云端 K1，2026-11-03）

口径：`taskset -c 0-3` + `--threads 4` + `--warmup`，官方 `llama-bench`（Q4_0, `-p 32,128
-n 32 -t 4 -r 3`）与本引擎（`VLLM_Q4_IME=2`）**交替跑 3 轮**，机器负载 3–7（桌面环境常驻）。
官方取 3 轮中位数，本引擎的 pp 由 `prefill ... in X.XXXs` 反算（引擎只打印整数 tok/s）。

| 模型 | ctx | 官方 tok/s | 本引擎 tok/s | 比值 | 官方 decode ms/tok | 本引擎 decode ms/tok |
|---|---|---|---|---|---|---|
| 0.6B | 32 | 57.99 | 54.5 | 94% | 97.6 | 84.1 |
| 0.6B | 128 | 43.81 | 50.3 | 115% | 97.6 | 90.0 |
| 1.7B | 32 | 26.49 | 22.1 | 84% | 219.3 | 207.0 |
| 1.7B | 128 | 16.12 | 21.7 | 135% | 219.3 | 211.2 |
| 4B | 32 | 11.07 | 9.57 | 86% | 460.8 | 465.5 |
| 4B | 128 | 8.16 | 8.44 | 103% | 460.8 | 475.9 |

结论：**短 prompt（32 token）仍落后 6%~16%**（prefill 里 GEMM 占 77%~90%，是纯算力差距）；
**128 token 已超过官方 3%~35%**；decode 在 0.6B/1.7B 上快 6%~14%，4B 慢 1%~3%（持平）。
注意本引擎 `--stream-ctx 128` 实际是 118 token，官方 pp128 是 128 token，批次差 8%，
长 ctx 的比值按此打折看。

IME 开关本身的收益（`VLLM_Q4_IME=2` vs `=0`，3 轮均值，prefill 总时长）：

| 模型 | ctx 32 | ctx 128 |
|---|---|---|
| 0.6B | 819.7 → 547.9 ms（1.50x） | 3080.8 → 2281.7 ms（1.35x） |
| 1.7B | 2262.6 → 1356.5 ms（1.67x） | 8809.0 → 5364.1 ms（1.64x） |
| 4B | 5606.9 → 3234.5 ms（1.73x） | 21693.6 → 13842.1 ms（1.57x） |

两臂的 `TOKIDS` 完全一致，确认 IME 是位级中立的快档。

---

## 7. 使用（客户端）

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

## 8. 诚实说明（当前边界）

1. **x86-64 构建只用于「可编译 + 自检与 ARM 同口径」**，不作为性能基准。开放项：
   x64 偶发整请求卡死；L3 Q4 点积在 x86 无 AVX2 内核（走标量回退，数值对、性能无意义）。
2. **板端数据只对云端 K1 成立**（8 核 rv64gcv，`vmadot` 仅 hart0-3 可执行，hart4-7 会 SIGILL）。
   RISC-V 侧可交叉编译验证「可编译 + 自检通过」；热内核为 RVV 档位 3，`VLLM_RVV=0` 可随时
   退回标量参考路径（逃生舱）；换板子必须重新实测，不能直接套用第 6 节的数字。
3. **回归闸门只证明「能编译 + 自检通过 + 跨架构同计数」**，不构成性能结论。
4. 模型转换、加密签名、位级对拍均有确定性红线（`-ffp-contract=off`、FNV/HMAC 同口径等），
   改动构建参数前请先读对应脚本头部的注释。

更多性能 / 适配细节见仓库根目录的 `arm记录/`、`mcu参考/` 系列文档与
`tools/preproc/README.md`。
