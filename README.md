[English](README.en.md) | **简体中文**

# vLLM-Kestrel

**面向 RISC-V 板端的纯 C 推理引擎。**

以 SpacemiT K1 为主要验证平台，通过自研 VQF 权重布局、RVV 内核和分层驻留运行 Qwen 稠密模型与 MoE；提供 OpenAI 兼容 HTTP 接口，并包含视觉、嵌入和 Laya 决策模型的独立执行路径。

项目重点是板端推理：让量化布局、计算内核和内存访问协同工作，同时保留可检查的数值与性能验证入口。模型权重不随源码分发。

[快速开始](#快速开始) · [模型准备](#模型准备) · [执行路径](#执行路径与配置) · [MoE 部署](#moe-部署) · [性能记录](#历史性能记录) · [验证方法](#正确性与性能验证) · [源码导航](#源码导航)

## 能力与平台

| 能力 | 当前入口与范围 |
|---|---|
| 稠密 LLM | Qwen2.5 / Qwen3；VQF Q4、Q8 等权重格式 |
| MoE | Qwen3-30B-A3B；专家连续布局、lane sidecar、按层及专家驻留 |
| HTTP 服务 | 对话、文本补全、嵌入接口；SSE 流式输出、连续批调度、KV 前缀复用 |
| 多模态与检测 | Qwen3-VL、YOLO；需分别准备相应模型 |
| 嵌入与决策 | Qwen3-Embedding、Laya / ModernBERT |
| 验证工具 | L3 / Sparse 自检、算子对拍、生成 token 序列与中间张量比较 |

主要性能记录来自 **SpacemiT K1：8 核、rv64gcv、VLEN=256**。本地约 8 GB 与云端约 16 GB 的机器分别记录，尤其不能混用 MoE 成绩。K1 的 IME / `vmadot` 是硬件专用能力，历史测试使用 hart 0–3。

Windows x86-64 构建用于开发与数值自检，本文不提供 x86 性能基准。源码也包含 aarch64 路径；其他硬件的功能和性能需要单独验证。

## 快速开始

以下 Linux 命令均从本项目目录执行。准备好包含 `model.vqf`、词表和配置的模型目录后，可直接构建并启动。

### 1. 构建 RISC-V 程序

```bash
# K1 板上原生构建
./tools/build/build_riscv.sh

# 或在 Linux 主机上交叉编译
CC=riscv64-linux-gnu-gcc ./tools/build/build_riscv.sh
```

产物为 `build-riscv/vllm_kestrel`。构建脚本支持：

| 环境变量 | 默认 / 用途 |
|---|---|
| `VLLM_RISCV_CC` | 自动探测编译器，也可显式指定 |
| `VLLM_RISCV_MARCH` | `rv64gcv` |
| `VLLM_RISCV_OUTDIR` | 项目下的 `build-riscv` |
| `VLLM_RISCV_EXTRA` | 追加编译与链接参数 |

保留脚本中的 `-ffp-contract=off`：乘加融合会改变浮点舍入顺序，影响参考实现对拍。

### 2. 启动服务并加载模型

```bash
./build-riscv/vllm_kestrel \
  --serve --port 8080 \
  --model /path/to/model \
  --threads 8 --auto-load
```

显式指定 `--port` 可避免交互式端口询问。服务默认不自动加载模型；这里使用 `--auto-load`，也可改用 `--load-on-use` 在首个请求时加载。

服务入口包括 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`，以及管理页 `/admin/` 和对话页 `/chat`。

### 3. 发起请求

```bash
curl http://127.0.0.1:8080/health

curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"stream":true}'
```

也可使用项目提供的零第三方依赖 Python 客户端：

```bash
python tools/client/vllm_client.py --url http://127.0.0.1:8080 models
python tools/client/vllm_client.py --url http://127.0.0.1:8080 chat "你好" --stream --max-tokens 128
```

<details>
<summary>Windows x86-64 开发构建</summary>

依赖 MinGW-w64 GCC（含 winpthreads、winsock2）。

```powershell
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1

.\build-x64\vllm_kestrel_x64.exe --serve --port 8080 `
  --model D:\models\qwen --threads 8 --auto-load
```

也可向构建脚本传入 `-Gcc` 和 `-OutDir` 指定编译器与输出目录。

</details>

## 模型准备

### 模型目录

通用 Qwen LLM 运行路径加载 VQF；原始 safetensors 由独立转换器处理，GGUF 不能直接用于该路径。Laya、YOLO 使用各自的打包工具，详见[模型适配说明](docs/模型适配.md)。

```text
model/
├── model.vqf       # 量化权重
├── config.json     # 模型配置与特殊 token 信息
├── vocab.bin       # 引擎词表，必需
└── merges.bin      # BPE 合并表，建议提供
```

服务的 `--model` 可指向模型目录或 `.vqf` 文件；下文的 `--stream-test` 示例统一使用目录。词表必须与模型匹配，缺失 `merges.bin` 时的分词回退可能改变输入 token 序列。

### 转换权重

在 Linux 主机上构建转换器：

```bash
(cd vqf_convert && ./build.sh)

./vqf_convert/vqf_conv \
  --model /path/to/hf-model \
  --convert-vqf /path/to/output/model.vqf \
  --wmode q4 --stream
```

输出目录需预先存在。Windows 可在 `vqf_convert` 中运行 `build.bat`，生成 `vqf_conv.exe`。

| 转换选项 | 用途 |
|---|---|
| `--wmode q4` | Q4 权重，较小存储占用 |
| `--wmode q8` | Q8 权重，用于更高权重精度的配置 |
| `--wmode dual` | 同时保存 Q4 / Q8，转换器默认值 |
| `--wmode q4i` / `g256` | 特定预打包或分组格式，按模型适配说明使用 |
| `--stream` | 降低转换过程峰值内存，适用于大模型 |

量化精度与布局是两个维度。`--wmode q4` 不等于自动获得性能表中的 col-major 布局；复现时必须核对 VQF 布局和生成工具。转换阶段未保存的精度不能靠运行参数恢复。

### 准备词表

```bash
python tools/build/build_vocab_bin.py \
  /path/to/tokenizer.json /path/to/existing_vocab.bin /path/to/output/vocab.bin

python tools/gen_merges_bin.py \
  /path/to/tokenizer.json /path/to/output/merges.bin
```

当前 `build_vocab_bin.py` 需要已有的兼容 `vocab.bin` 来复制特殊 token，并非仅凭 `tokenizer.json` 就能从零生成任意模型词表。准备时还需将匹配的 `config.json` 放入模型目录；具体约束见[词表准备](docs/模型适配.md#13-词表准备)。

## 执行路径与配置

### 权重布局与内核

| 路径 | 作用 | 当前行为 |
|---|---|---|
| VQF row / 4×4 | 基础 Q4 布局 | 根据张量布局选择对应实现 |
| VQF lane-major | 连续 lane 访问，减少离散取数 | RVV 路径默认 level 3，仍受向量长度与形状条件约束 |
| VQF col-major | 面向批量 prefill 和单 token decode | 专用 decode 内核默认开启，复用激活量化与缓冲 |
| MoE sidecar | gate/up、down 按专家及 lane 连续存储 | RVV 构建在提供路径后懒加载并校验 |
| Laya LQ8V | 缓存 INT8 / IME 预打包数据 | 减少重复量化与打包；持久化 Arena 复用前向缓冲 |

普通稠密模型先使用默认配置，无需复制历史实验中的整组环境变量。以下配置按当前实现核对：

| 配置 | 当前语义 |
|---|---|
| `VLLM_Q4_RVV` | 默认 `3`；`0` 可用于参考路径对照，具体分支受布局约束 |
| `VLLM_Q4_DEC` | 默认开启；`0` 关闭专用 decode 内核 |
| `VLLM_Q4_IME` | 当前 `rv_q4_ime_mode()` 固定返回 `0`；旧版 `=1/2` 实验不能作为当前推荐命令 |
| `VLLM_Q4_VMADOT=1` | K1 专用 decode 路径，并隐含开启 DEC；需满足硬件与核亲和性要求 |
| `VLLM_GU_LANE_PATH` / `VLLM_DN_LANE_PATH` | 指向 MoE sidecar；不再需要旧的 `GU_LANE=1` / `DN_LANE=1` 启用开关 |

这些默认行为的源码入口为 [vllm_safetensors.c](src/model/vllm_safetensors.c) 和 [vqf.c](src/model/vqf.c)。启用某个内核不意味着其在任意模型、形状和硬件上都会执行。

### 数值精度与服务配置

权重量化、KV 量化、稀疏注意力及近似算子都需要独立验证。**生成 token 相同与中间浮点结果逐位相同是不同的验证层次。** 尤其 MoE 的历史 RVV 路径相对标量曾出现约 1 ulp 差异和 token 翻转，不应将所有优化统称为“位级一致”。

| 参数 | 用途 |
|---|---|
| `--threads N` | 工作线程数；基准比较时固定 |
| `--auto-load` / `--load-on-use` | 启动加载 / 首次使用加载 |
| `--auto-unload N` | 空闲 N 秒后卸载 |
| `--batch-max N` / `--max-queued N` | 连续批与排队上限 |
| `--min-free-mb N` | 保留空闲内存阈值 |
| `VLLM_KV_F32=1` | F32 KV 对照配置，增加内存需求 |
| `--kv-q4` | 更低精度 KV 配置，需质量评估 |
| `--sparse-attn` | 显式开启近似稀疏注意力；默认不开启 |
| `--spec --spec-k N` | 投机解码配置，需验证所选模式的输出约束 |
| `--no-prefix-kv` | 关闭服务端 KV 前缀复用 |

多轮请求可将响应中的 `history_tokens` 作为下一轮的 `context_tokens` 回传，供服务端匹配精确 token 前缀。复用是否命中取决于缓存和输入条件。

## MoE 部署

Qwen3-30B-A3B 使用 48 层、128 个专家、top-8 路由。历史 Q4 权重约 16.5 GB，板端运行依赖分层驻留；decode 对存储、缺页与系统页缓存十分敏感。

### 数据准备顺序

1. 转换 Q4 VQF，准备词表和配置。
2. 将 down 权重转成专家连续布局（EC）。该步骤会原地修改文件，应先保留原件。
3. 从对应源文件生成 GU / DN lane sidecar，并完成全量校验。

工具源码：[vqf_down_relayout.c](tools/bench/vqf_down_relayout.c)、[gu_lane_repack.c](tools/bench/gu_lane_repack.c)、[dn_lane_repack.c](tools/bench/dn_lane_repack.c)。编译工具后的调用顺序为：

```bash
# 0：检查；1：原地转换 EC
./vqf_down_relayout /path/to/model/model.vqf 0
./vqf_down_relayout /path/to/model/model.vqf 1

# 0：预检查；1：生成；2：全量校验
./gu_lane_repack /path/to/model/model.vqf /path/to/model/gu_lane.bin 0
./gu_lane_repack /path/to/model/model.vqf /path/to/model/gu_lane.bin 1
./gu_lane_repack /path/to/model/model.vqf /path/to/model/gu_lane.bin 2

./dn_lane_repack /path/to/model/model.vqf /path/to/model/dn_lane.bin 0
./dn_lane_repack /path/to/model/model.vqf /path/to/model/dn_lane.bin 1
./dn_lane_repack /path/to/model/model.vqf /path/to/model/dn_lane.bin 2
```

每一步成功后再执行下一步。**`VLLM_DN_EC=1` 必须对应已转换的 EC 权重，DN sidecar 也必须从该布局生成**，否则可能错读专家数据。历史 30B 配置的两个 sidecar 合计约 15.1 GB，加上原 VQF 约需 31.6 GB 存储，另需预留备份与转换空间。

### 运行示例

```bash
MODEL=/path/to/model

env OMP_NUM_THREADS=8 VLLM_THREADS=8 \
  VLLM_VQF_STREAM=12 VLLM_DN_EC=1 VLLM_SA=1 \
  VLLM_GU_LANE_PATH="$MODEL/gu_lane.bin" \
  VLLM_DN_LANE_PATH="$MODEL/dn_lane.bin" VLLM_DN_SAPF=1 \
  ./build-riscv/vllm_kestrel \
  --stream-test --model "$MODEL" --stream-n 256 --threads 8
```

`VLLM_VQF_STREAM` 控制保留层数，`VLLM_SA` 控制专家驻留，`VLLM_DN_SAPF` 配合已注册的 DN sidecar 做专家级预取与驱逐。需要分项诊断时增加 `VLLM_MOE_PROF=1 VLLM_SA_STAT=1`。

检查日志中 sidecar 启用、STREAM 和 SA 状态；出现拒绝或回退时，不应把该次运行计为优化配置。加密 VQF 与可驱逐映射存在兼容限制。sidecar 加载校验头部及源文件绑定信息，不替代生成阶段的数据区全量校验。

[final_bench_cloud.sh](tools/bench/final_bench_cloud.sh) 提供首尾基线及 n=64/128/256 的测试矩阵，使用前需修改脚本中的设备路径。

## 历史性能记录

**本节是历史记录整理，不是当前提交的复测结果，也不构成“收口后无退化”的证明。** 稠密与 Laya 数据来自工作区[最近性能记录](../最近性能记录.txt)，MoE 与四线程对照表沿用此前 README 的记录。历史汇报未完整保存提交号、原始日志及所有统计条件，引用时应同时保留下述口径。

### 稠密 Qwen：云端 K1，8 核 RVV

历史命令参数为 `--stream-ctx 32 --stream-n 16`；下表为 col-major + DEC 路径。Prefill 吞吐、decode 延迟与吞吐按记录原值列出，存在取整和统计分母差异，不能将 ctx 参数直接视为实际 prefill token 数。记录未完整注明 warmup 与重复轮数。

| 模型 | Prefill 耗时 | 报告 prefill 吞吐 | Decode 延迟 | 报告 decode 吞吐 | 峰值 RSS |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 302.9 ms | 97 tok/s | 48 ms/tok | 20.83 tok/s | 293 MB |
| Qwen3-0.6B | 414.0 ms | 72 tok/s | 58 ms/tok | 17.13 tok/s | 356 MB |
| Qwen3-1.7B | 1178.7 ms | 25 tok/s | 148 ms/tok | 6.74 tok/s | 971 MB |
| Qwen3-4B | 3000.8 ms | 10 tok/s | 341 ms/tok | 2.94 tok/s | 2240 MB |

同一历史批次中，col-major 的 DEC 关闭 → 开启记录分别为：0.6B **75 → 58**、1.7B **195 → 148**、4B **428 → 341 ms/tok**。这是 decode 内核开关对比，应与 lane → col 的布局变化分开理解。

### SpacemiT 官方 llama-bench 对照

最近记录中的官方程序使用 **4 线程、Q4_0 GGUF**，Kestrel 使用 **8 核、col-major VQF**。该官方版本在 8 线程配置下报告首选核心数量错误；这仅描述当时安装的版本，不代表所有 llama.cpp 版本的限制。

| 模型 / 指标 | 官方历史值 | Kestrel 历史值 |
|---|---:|---:|
| Qwen3-0.6B prefill | 58.94 tok/s（pp32） | 72.00 tok/s |
| Qwen3-0.6B decode | 10.68 tok/s（tg16） | 17.13 tok/s |
| Qwen3-1.7B decode | 4.71 tok/s（tg16） | 6.74 tok/s |
| Qwen3-4B decode | 2.28 tok/s（tg16） | 2.94 tok/s |

这组结果反映不同线程配置下的部署表现，不能单独归因于内核效率；权重格式、输入与计时边界也需对齐后才能做严格性能比较。

<details>
<summary>早期四线程对照：2026-11-03，保留供追溯</summary>

历史条件：`taskset -c 0-3`、4 线程、warmup、旧版 `VLLM_Q4_IME=2`，交替三轮。当前源码已固定关闭该 IME prefill 模式，不能用当前二进制直接复现旧开关行为。

| 模型 | ctx 参数 | 官方 prefill tok/s | Kestrel prefill tok/s | 官方 decode ms/tok | Kestrel decode ms/tok |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 32 | 57.99 | 54.5 | 97.6 | 84.1 |
| Qwen3-0.6B | 128 | 43.81 | 50.3 | 97.6 | 90.0 |
| Qwen3-1.7B | 32 | 26.49 | 22.1 | 219.3 | 207.0 |
| Qwen3-1.7B | 128 | 16.12 | 21.7 | 219.3 | 211.2 |
| Qwen3-4B | 32 | 11.07 | 9.57 | 460.8 | 465.5 |
| Qwen3-4B | 128 | 8.16 | 8.44 | 460.8 | 475.9 |

历史记录注明 Kestrel 的 ctx=128 实际输入为 118 token，官方 pp128 为 128 token，因此不是等工作量比较。相关脚本为 [_k1_final4.sh](tools/yolo/_k1_final4.sh)。

</details>

### MoE：Qwen3-30B-A3B

8 线程、STREAM=12、EC、RVV、SA；优化配置增加 GU / DN lane 与 DN_SAPF。以下为历史 decode 平均延迟，**数值越低越好**。

| 配置 | 本地 K1，约 7.8 GB RAM | 云端 K1，约 15.5 GB RAM |
|---|---:|---:|
| 无 lane sidecar 基线，n=256 | 2048 ms/tok | 1983 ms/tok |
| lane + DN_SAPF，n=64 | 1778 ms/tok | 1629 ms/tok |
| lane + DN_SAPF，n=128 | 1651 ms/tok | 1390 ms/tok |
| lane + DN_SAPF，n=256 | 1581 ms/tok | 1329 ms/tok |

n=256 同口径下的延迟降幅分别为 22.8% 和 33.0%。生成长度改变会影响冷启动与缺页开销在均值中的占比，不能用 n=16 与 n=256 判断回归。云端更充裕的页缓存是历史分析中的重要差异，换机器、内存或存储后都需重测。

历史优化配置的 token 对照只覆盖当时测试组合；RVV 与标量并非普遍逐位等价。进程 RSS 也不包含系统页缓存等全部内存消耗，不能用较小 RSS 推导不会 OOM。

### Laya：预打包加载与 Arena 复用

历史云端 K1 数据，模型为 ModernBERT-large 28 层编码器及决策头，约 4.21 亿参数、206 个张量。

| 执行路径 | 原加载耗时 | 首次构建缓存 | 已有缓存加载 | 缓存加载加速 |
|---|---:|---:|---:|---:|
| RVV，8 核 | 21795.6 ms | 6490.8 ms | 809.9 ms | 26.91× |
| K1 IME，4 核 | 21822.5 ms | 10033.7 ms | 1582.2 ms | 13.79× |

n=59 的峰值 RSS：RVV **1210.6 → 410.8 MB**（下降 66.1%），IME **1210.6 → 763.4 MB**（下降 36.9%）。IME 首请求耗时分别为 n=59 的 **5134.2 → 3037.7 ms**、n=256 的 **11018.5 → 7867.8 ms**。

加载加速、首请求与稳态推理是不同指标。历史门禁报告 42 个中间层及输出文件 `SAME=42 DIFF=0`，结论限定于该组输入和配置。

Laya 数据准备入口为 [laya_pack.py](tools/laya_pack.py)，模型接入流程见[模型适配说明](docs/模型适配.md)。

## 正确性与性能验证

### 无模型自检

```bash
# 板端构建并执行回归门禁
./tools/build/check_riscv.sh

# 交叉编译环境仅构建
./tools/build/check_riscv.sh --build

# 单独执行，保留完整输出和退出码
./build-riscv/vllm_kestrel --test-l3
./build-riscv/vllm_kestrel --test-sparse
```

当前 `check_riscv.sh` 的硬编码通过条件为 L3 **16 PASS / 0 FAIL**、Sparse **9 PASS / 0 FAIL**，SKIP 单独统计。此前测试汇报出现过 Sparse 10 PASS；脚本期望与该记录尚未统一，应核对当前二进制的实际输出，不能直接宣称门禁已通过。本次 README 整理未运行构建或自检。

### 稠密模型基准

```bash
env OMP_NUM_THREADS=8 VLLM_THREADS=8 \
  ./build-riscv/vllm_kestrel \
  --stream-test --model /path/to/model \
  --stream-ctx 32 --stream-n 64 --threads 8 --warmup
```

此命令用于建立新的固定口径，不等同于上述 n=16 历史测试。应读取日志中的实际 prefill token 数；`--warmup` 会执行预热，必须在比较双方保持一致。

### 记录与判定

- **固定条件**：同一设备、线程与亲和性、模型文件及布局、输入 token、生成长度、量化配置和 warmup 策略。
- **保留证据**：源码提交与二进制标识、编译选项、环境变量、模型校验值、完整 stdout/stderr、退出码、内存与存储状态。
- **交替测试**：使用 A/B/A/B 或首尾基线控制漂移，报告重复轮数和统计方式；不将单次波动当作收益。
- **分层对拍**：算子及中间张量检查数值差异；完整 `TOKIDS` 检查生成结果。不要只比较首行 token，也不要把 token 相同当成浮点逐位相同。
- **单独评估近似**：涉及量化或稀疏注意力时，使用固定语料的 NLL、召回或任务质量指标。质量测试与速度测试分开采集。

`VLLM_Q4CHK=1` 可用于相关 Q4 算子对拍；`VLLM_STREAM_NLL=1` 为 NLL 测量配置。历史记录注明 NLL 模式不输出 `TOKIDS`，不能对空输出取哈希后宣称一致。

## 源码导航

| 路径 | 内容 |
|---|---|
| [src/main.c](src/main.c) | 命令行、服务与测试入口 |
| [src/core](src/core/) | 注意力、调度、线程池、KV 与内核 |
| [src/model](src/model/) | VQF、分词器、LLM、视觉、Laya 与 YOLO |
| [src/serve](src/serve/) | HTTP、连续批、会话与内嵌页面 |
| [src/media](src/media/) | 视频源与媒体处理 |
| [include](include/) | 公共接口 |
| [vqf_convert](vqf_convert/) | 独立权重转换器 |
| [tools/build](tools/build/) | 构建和自检脚本 |
| [tools/bench](tools/bench/) | 基准、布局重排与诊断工具 |
| [tools/client](tools/client/) | Python 客户端 |
| [模型适配说明](docs/模型适配.md) | 各类模型的准备与接入约束 |

本文区分当前源码行为和历史测试记录。构建成功、自检通过、输出一致与性能无退化分别需要对应证据；旧实验脚本中的开关及设备绝对路径，使用前应与当前实现核对。
