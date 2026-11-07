# Kestrel - RISC-V模型协同推理平台

[English](README.en.md) | **简体中文**

**纯 C11、零第三方运行时依赖的 RISC-V（rv64gcv）模型协同推理平台（SpacemiT K1 开发基准）。**

> **Kestrel（红隼）** 为本平台的品牌 / 对外名（取义：最小猛禽、俯冲精确——边缘小模型 + 可验证推理）。
> 工程名与可执行文件为 `vllm_kestrel`：下文命令、日志与代码中的 `vllm_kestrel` / `vllm` 均指本平台。
> 本仓库为 **RISC-V（`rv64gcv`）分支**，工程位于 `https://gitee.com/pei-xiaoguang/kestrel-risc-v`。

vllm_kestrel 是一个从零自研、面向 RISC-V 边缘设备的纯 C 模型**协同推理**平台：
**同一个二进制、同一进程**承载文本 LLM / MoE / 向量嵌入 / 多模态（Qwen3-VL）/ 决策（Laya）/
检测（YOLO）六类模型，各占独立运行槽位、按端点切换，并通过 `/v1/ask` 统一编排入口做多模型协同。
单文件 `vllm_kestrel` 即可服务 OpenAI 兼容 HTTP API。核心亮点：

- **同进程多模型协同**：`--model` / `--embed-model` / `--laya-pack` / `--yolo-model` 四类槽位可并存，
  `/v1/ask` 按显式字段路由到 chat / rag / decide / scene 四条链（见「协同推理」）。
- **纯 CPU 推理**：手写 RVV 量化 GEMM/GEMV（4×4、lane-major、col-major 多布局）、自研线程池
  `vllm_tp`，不依赖 GPU/NPU/MAC 即可运行。
- **零第三方依赖**：自研 SM3/SM4/SM2 国密原语、自研线程池（替代 OpenMP），单文件 mmap 权重格式 VQF；
  标准构建仅需 gcc + libm。
- **自研 VQF 单文件格式**：mmap 直挂、按需建页、多量化布局（Q4/Q8、row/lane/col-major）随张量固化，
  支持 MoE 专家连续布局（EC）与 lane sidecar。
- **长上下文工程优化**：稀疏注意力、prefix-KV 前缀复用、L3 分层驻留（逐层推理）、磁盘 KV 持久化、
  投机解码、连续批处理（详见 [docs/模型适配.md](docs/模型适配.md)）。

---

## 快速目录

- [核心优势](#核心优势)
- [协同推理](#协同推理)
- [性能](#性能)
- [六个模型的详细配置与复现](#模型配置与复现)
  - [① 稠密 Qwen2.5/Qwen3](#集合-①稠密-qwen25qwen3chat)
  - [② MoE Qwen3-30B-A3B](#集合-②moe-qwen3-30b-a3b)
  - [③ 多模态 Qwen3-VL](#集合-③多模态-qwen3-vl)
  - [④ 向量嵌入 Qwen3-Embedding](#集合-④向量嵌入-qwen3-embedding)
  - [⑤ 决策 Laya](#集合-⑤决策-laya)
  - [⑥ 检测 YOLO](#集合-⑥检测-yolo)
- [启动流程](#启动流程)
- [正确性与性能验证](#正确性与性能验证)
- [模型转换工具](#模型转换工具vqf_convert)
- [权重保护与推理验证](#权重保护与推理验证attestation)
- [零第三方依赖自证清单](#体量与依赖小而全)
- [目录结构](#目录结构)
- [变更摘要](#变更摘要相对上一公开形态)
- [许可与联系](#许可与合规)

---

## 核心优势

一块 RISC-V 开发板（开发基准：SpacemiT K1，8 核，`rv64gcv`，VLEN=256）+ 一个约 **1.3 MB** 的单文件
可执行程序 = 一个**模型协同推理平台**——自研 VQF 权重单文件 mmap 直挂、按需建页，同一进程承载六类模型。

| 核心优势 | 一句话指标 |
|---|---|
| **六类模型同进程协同** | 文本 LLM / MoE / 向量 / 多模态 / 决策 / 检测，一个二进制 + 四类槽位并存，`/v1/ask` 统一编排 |
| **自研 VQF 权重布局** | 单文件 mmap 直挂、按需建页；量化与布局解耦，随张量固化（row 4×4 / lane-major / col-major / Q4 / Q8） |
| **col-major + 专用 DEC 解码** | 稠密 LLM prefill 提速 **2.1~2.4×**（相对 lane-major），decode 默认开启融合内核 **+25%~32%** |
| **MoE 分层驻留** | `VLLM_VQF_STREAM` + EC 连续布局 + GU / DN lane sidecar：17 GB 级 MoE 在 8–16 GB 内存板上可服务 |
| **视觉塔两段式优化** | LLM 主干 Q4 + DEC 对多模态 decode **1.29×**；`VLLM_VIS_RVV=1` 图片编码 **13~14.5×**（单层 MLP 34×），24 层位级对拍 0 失配 |
| **纯自研格式与内核** | 只认自研 VQF 单文件；单产物 **≈1.3 MB**，零第三方运行时；手写 RVV 量化 GEMM/GEMV、自研线程池、自研国密 |
| **可验证推理** | SM2 供应链签名护权重 + 逐请求 attestation 凭证，浏览器内 / 离线零依赖验签（`tools/security/verify_attest.py`） |

---

## 协同推理

平台的核心价值是**多模型在同一进程内、同一二进制上协同**。四类运行槽位 + 统一编排入口：

| 槽位参数 | 模型类型 | 运行端点 |
|---|---|---|
| `--model` | 文本 LLM / MoE / 多模态 | `/v1/chat/completions`、`/v1/completions` |
| `--embed-model` | 向量嵌入 | `/v1/embeddings` |
| `--laya-pack` | 决策模型 | `/v1/systemone` |
| `--yolo-model` | 检测模型 | `/v1/detect` |

**统一编排入口 `/v1/ask`**：一个端点编排四条链，**只认显式字段、不做语义猜测**：

- 带 `image` → **scene**（YOLO 检测 → 场景叙事 → LLM）
- 带 `kb_dir` → **rag**（embedding 检索 → top-k → LLM）
- 带 `questions` → **decide**（Laya 决策）
- 否则 → **chat**（LLM 对话）

响应回带 `route` 字段（`chat` / `rag` / `scene` / `decide` 之一），标明本次由哪条链作答。多模型
协同使「感知 → 决策 → 生成」整条链在单板单进程内闭环，无需为每类模型各带一套运行时。

---

## 性能

### 性能数据分区说明

本项目性能/体积数字按口径分区：

1. **受控 A/B（可复现）**——`--stream-test` 口径，同一块板、同一模型文件、同一会话内交错两种配置。
2. **历史记录整理（不是当前提交的复测，也不构成“收口后无退化”证明）**——文中标注「历史」的数据整理自
   工作区[最近性能记录](../最近性能记录.txt)；未完整保存提交号、原始日志与统计条件，引用时应保留下述口径。
   换板复现请以 `./tools/build/check_riscv.sh` 的当前自检与产物为准。
3. **跨引擎对照**——与本仓库同源的主机（x86-64，MinGW）仅做**数值一致性自检，不作性能基准**。

**测量环境**：SpacemiT K1（8 核 RISC-V，`rv64gcv`，VLEN=256，Cluster 0 含 4 个 IME `vmadot` 核）。
本地约 8 GB 与云端约 15~16 GB 的机器分别记录，**尤其不能混用 MoE 成绩**。

**正确性判据**：量化 / KV 量化 / 稀疏注意力 / 近似算子各自独立验证；「生成 token 相同」与「中间浮点逐位
相同」是不同的验证层次。

### 结论速览

| 类别 | 进展 / 当前状态 |
|---|---|
| **稠密 LLM** | col-major Q4 相对 lane-major prefill **2.1~2.4×**；decode 默认 DEC **+25%~32%** |
| **MoE Qwen3-30B-A3B** | lane + DN_SAPF 分层驻留 n=256 decode 延迟 **−22.8%（本地）/ −33.0%（云端）** |
| **多模态 Qwen3-VL-2B** | LLM 主干 DEC **1.29×**；视觉 `VLLM_VIS_RVV=1` 图片编码 **13~14.5×**、端到端 **5.7~7×**；位级对拍 0 失配 |
| **Laya 决策** | 预打包零拷贝加载 **13.79~26.91×**，首请求 IME **1.40~1.69×**，峰值内存 **−36.9%~66.1%**（42/42 逐字节一致） |

---

## 模型配置与复现

> 本文每个模型给出**从下载 → 转换 → 运行 → 复现基准**的完整步骤。模型权重不随仓库分发，均需自行下载后
> 转换。Qwen 系权重遵循其原始开源许可（Qwen 社区许可），可自 ModelScope / HuggingFace 下载。

---

### 集合 ①：稠密 Qwen2.5 / Qwen3（chat）

形态：`Qwen2ForCausalLM` / `Qwen3ForCausalLM`，RMSNorm + GQA + SwiGLU + 因果注意力。
Qwen2 vs Qwen3 唯一加载差异（q/k/v bias、q_norm/k_norm）由 `config.json` 自动识别，无需改代码。

**1. 下载**（ModelScope，示例 Qwen3-0.6B，约 1.45 GB）：

```bash
# 无 git-lfs 的轻量下载：用 ModelScope 文件 API 逐文件取，或用 hf 客户端
# hf download <repo> --local-dir ./Qwen3-0.6B   （若板端有 hf 客户端）
```

**2. 转换权重 + 词表**：

```bash
python tools/build/build_vocab_bin.py ./Qwen3-0.6B/tokenizer.json \
  <旧vocab.bin> ./out/vocab.bin
python tools/gen_merges_bin.py ./Qwen3-0.6B/tokenizer.json ./out/merges.bin
cp -f ./Qwen3-0.6B/config.json ./out/config.json

(cd vqf_convert && ./build.sh)
./vqf_convert/vqf_conv --model ./Qwen3-0.6B --convert-vqf ./out/model.vqf \
  --wmode q4 --stream
```

**3. 启动服务**：

```bash
./build-riscv/vllm_kestrel --serve --port 8080 --model ./out --threads 8 --auto-load
```

**4. 复现基准（受控 A/B，Greedy）**：

```bash
./build-riscv/vllm_kestrel --stream-test --model ./out \
  --stream-ctx 32 --stream-n 16 --threads 8
```

期望（col-major + DEC，云端 K1 8 核历史值）：

| 模型 | VQF 布局/内核 | Prefill 耗时 | Prefill 吞吐 | Decode 延迟 | Decode 吞吐 | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | col-major + DEC | 302.9 ms | 97 tok/s | 48 ms/tok | 20.83 tok/s | 293 MB |
| Qwen3-0.6B | lane-major（默认 RVV） | 879.8 ms | 35 tok/s | 70 ms/tok | 14.30 tok/s | 356 MB |
| Qwen3-0.6B | col-major + DEC | 414.0 ms | 72 tok/s | 58 ms/tok | 17.13 tok/s | 356 MB |
| Qwen3-1.7B | col-major + DEC | 1,178.7 ms | 25 tok/s | 148 ms/tok | 6.74 tok/s | 971 MB |
| Qwen3-4B | col-major + DEC | 3,000.8 ms | 10 tok/s | 341 ms/tok | 2.94 tok/s | 2,240 MB |

> 同批 DEC 关→开：0.6B **75→58**、1.7B **195→148**、4B **428→341 ms/tok**。

**布局默认化修复**：`layout=2`（lane-major）张量默认启用 RVV（修复原标量退化，曾致 Qwen3-0.6B prefill
98,441 ms / 0.3 tok/s）；修复后默认 879.8 ms（prefill **111.9×**、decode **59.8×**）。

**对照官方 llama-bench（历史）**：官方为 SpacemiT 预装、**4 线程、Q4_0 GGUF**（`-t 8` 报错），Kestrel 用
**8 核、col-major VQF**；两者线程/权重不同，不能单独归因内核效率。

| 模型 / 指标 | 官方历史值 | Kestrel 历史值 | 比 |
|---|---|---:|---:|---:|
| Qwen3-0.6B prefill（pp32） | 58.94 tok/s | 72.00 tok/s | 1.22× |
| Qwen3-0.6B decode（tg16） | 10.68 tok/s | 17.13 tok/s | 1.60× |
| Qwen3-1.7B decode（tg16） | 4.71 tok/s | 6.74 tok/s | 1.43× |
| Qwen3-4B decode（tg16） | 2.28 tok/s | 2.94 tok/s | 1.29× |

---

### 集合 ②：MoE Qwen3-30B-A3B

形态：`Qwen3MoeForCausalLM`，稀疏专家 FFN（48 层 / 128 专家 / top-8）。`config.json` 中
`num_experts>0 && moe_intermediate_size>0 && num_experts_per_tok>0` 同时成立即判 MoE。
30B 为多分片 `model-N-of-N.safetensors`，转换器自动按序加载。

**1. 下载 + 转换（大模型用 `--stream` 降峰值内存）**：

```bash
mkdir -p out && cd vqf_convert && ./build.sh && cd ..
./vqf_convert/vqf_conv --model ./Qwen3-30B-A3B --convert-vqf ./out/model.vqf \
  --wmode q4 --stream
# + 生成 vocab.bin / merges.bin / config.json（同集合①）
```

**2. MoE 编译数据（专家连续布局 + lane sidecar）**：先**备份**原 VQF，再按序执行——

```bash
# EC：down 权重转专家连续布局（原地修改）
./vqf_down_relayout ./out/model.vqf 0      # 检查
./vqf_down_relayout ./out/model.vqf 1      # 原地转 EC

# GU / DN lane sidecar（0 预检查 / 1 生成 / 2 全量校验）
./gu_lane_repack ./out/model.vqf ./out/gu_lane.bin 0
./gu_lane_repack ./out/model.vqf ./out/gu_lane.bin 1
./gu_lane_repack ./out/model.vqf ./out/gu_lane.bin 2
./dn_lane_repack ./out/model.vqf ./out/dn_lane.bin 0
./dn_lane_repack ./out/model.vqf ./out/dn_lane.bin 1
./dn_lane_repack ./out/model.vqf ./out/dn_lane.bin 2
```

> `VLLM_DN_EC=1` 必须对应已转 EC 的权重，DN sidecar 也必须从该布局生成，否则错读专家数据。
> 两个 sidecar 历史合计约 15.1 GB，加原 VQF 需预留备份与转换空间。

**3. 运行（分层驻留 + sidecar）**：

```bash
MODEL=./out
env VLLM_VQF_STREAM=12 VLLM_DN_EC=1 VLLM_SA=1 \
  VLLM_GU_LANE_PATH="$MODEL/gu_lane.bin" \
  VLLM_DN_LANE_PATH="$MODEL/dn_lane.bin" VLLM_DN_SAPF=1 \
  ./build-riscv/vllm_kestrel --serve --port 8080 --model "$MODEL" --threads 8 --auto-load
```

**4. 复现基准（历史 decode 平均延迟，数值越低越好）**：

| 配置 | 本地 K1（约 7.8 GB） | 云端 K1（约 15.5 GB） |
|---|---:|---:|
| 无 lane sidecar 基线，n=256 | 2048 ms/tok | 1983 ms/tok |
| lane + DN_SAPF，n=64 | 1778 ms/tok | 1629 ms/tok |
| lane + DN_SAPF，n=128 | 1651 ms/tok | 1390 ms/tok |
| lane + DN_SAPF，n=256 | 1581 ms/tok | 1329 ms/tok |

n=256 同口径降幅 **22.8%（本地）/ 33.0%（云端）**。`VLLM_VQF_STREAM` 控制保留层数，`VLLM_SA` 控制专家驻留；
分项诊断加 `VLLM_MOE_PROF=1 VLLM_SA_STAT=1`。sidecar 加载会校验头部与源文件绑定；加密 VQF 与逐层
驻留存在兼容限制。

---

### 集合 ③：多模态 Qwen3-VL

形态：`Qwen3VLForConditionalGeneration` = Qwen3 语言主干 + ViT 视觉编码器 + DeepStack 桥接。
转换器 `config.json` 存在 `vision_config` 子对象即自动解析视觉段，写出 `VQF_FLAG_VISION` 与整套
`v_q8_*` / `v_*` 张量（视觉失败则降级为纯文本 VQF，仅告警）。

**1. 下载 + 转换（本轮实测 Qwen3-VL-2B-Instruct，4.06 GB）**：

```bash
./vqf_convert/vqf_conv --model ./Qwen3-VL-2B-Instruct --convert-vqf ./out/model.vqf \
  --wmode q4 --stream
# + 生成 vocab.bin / merges.bin / config.json（含视觉特殊 token）
```

> 转换日志应出现 `[ST] Vision` 且产出含视觉段；`model.vqf` 体积约 **2223 MB**
> （`dim=2048, layers=28, heads=16, kv_heads=8, ffn=6144, vocab=151936, has_q_norm=1, has_mrope=1`，
> ViT `depth=24, hidden=1024, patch=16, merge=2, DeepStack=[5,11,17]`）。

**2. 运行（务必开视觉 RVV 优化权重）**：

```bash
VLLM_VIS_RVV=1 ./build-riscv/vllm_kestrel --serve --port 8080 \
  --model ./out --threads 8 --auto-load

# 客户端传图
python tools/client/vllm_client.py --url http://127.0.0.1:8080 chat \
  "描述图片内容" --image photo.jpg --max-tokens 128
```

**3. 复现基准（云端 K1，8 线程，合成几何图）**：

| 测试模式 | 输入 | 图片编码(ViT) | 单层 ViT(L0) | LLM Decode(TPOT) | 端到端 Wall（冷图） |
|---|---|---|---|---|---|
| 默认（视觉 Q8 线性 + LLM col-Q4/DEC） | 128×128 | 93,433 ms | 3.324 s | — | 104.6 s |
| **`VLLM_VIS_RVV=1`** | 128×128 | **6.43~7.15 s（13~14.5×）** | **0.156 s（21×）** | 261.9 ms（3.82 tok/s） | **14.8~18.2 s（5.7~7×）** |
| 仅开 LLM DEC | 128×128 | — | — | 336.9→**261.9 ms（1.29×）** | — |
| `VLLM_VIS_RVV=1` | 256×256 | **31.3 s（≈10.9×）** | 1.085 s | — | 48.3 s |

正确性：`VLLM_VIS_ATTN_CHK=1`（每层 head0 标量对拍）**24 层 0 失配**；多模式贪心输出逐字一致。
代价：视觉 8×8 重排权重启动额外占 **≈306 MB**（K1 上总 RSS 约 2.0 GB）。

---

### 集合 ④：向量嵌入 Qwen3-Embedding

形态：`Qwen3ForCausalLM`（与 Qwen3-0.6B 同构但**无 `lm_head`（tied embeddings）**、`vocab_size=151669`）。
取值语义：last-token pooling + L2 归一化，尾部恒追加 EOS=151643。

**1. 转换 + 词表**：

```bash
./vqf_convert/vqf_conv --model ./Qwen3-Embedding-0.6B \
  --convert-vqf ./out/model.vqf --wmode q4 --stream
python tools/build/build_vocab_bin.py ./Qwen3-Embedding-0.6B/tokenizer.json \
  <旧vocab.bin> ./out/vocab.bin
python tools/gen_merges_bin.py ./Qwen3-Embedding-0.6B/tokenizer.json ./out/merges.bin
cp ./Qwen3-Embedding-0.6B/config.json ./out/config.json
```

**2. 运行（CLI 或 HTTP）**：

```bash
# CLI：一次性取值
./build-riscv/vllm_kestrel --model ./out --embed "hello world"
./build-riscv/vllm_kestrel --model ./out --embed-file ./texts.txt --embed-out ./vecs.bin

# HTTP 服务
./build-riscv/vllm_kestrel --serve --port 8080 --embed-model ./out --model <chat-dir> --threads 8
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" -d '{"input":"hello world"}'
```

> 引擎通过 `st_embed_request()` 一次性取值钩子复用 prefill 内核，未登记时对 chat 路径零行为改变。

---

### 集合 ⑤：决策 Laya（ModernBERT）

形态：**双向编码器** + 决策头，非自回归（无 decode 循环、无 KV cache），一次前向对每个 `[MASK]` 打分，
输出类型化决策。非 Qwen 系，走独立打包路径。

**1. 打包 + 词表**：

```bash
python tools/gen_modernbert_bin.py <ModernBERT/tokenizer.json> ./out      # vocab.bin + merges.bin
python tools/laya_pack.py <model_dir> ./out                               # laya_f16.bin + manifest
```

**2. 运行（CLI 或 `/v1/systemone`）**：

```bash
./build-riscv/vllm_kestrel --laya-pack ./out --laya-cases <cases.txt> --laya-decide
# 或 serve 下
./build-riscv/vllm_kestrel --serve --port 8080 --laya-pack ./out --threads 8
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" -d '{"question":"..."}'
```

**3. 复现基准（历史云端 K1，ModernBERT-large 206 张量）**：

| 执行路径 | 原加载耗时 | 首次构建缓存 | 已有缓存加载 | 加载加速 |
|---|---:|---:|---:|---:|
| RVV，8 核 | 21795.6 ms | 6490.8 ms | 809.9 ms | 26.91× |
| K1 IME，4 核 | 21822.5 ms | 10033.7 ms | 1582.2 ms | 13.79× |

n=59 峰值 RSS：RVV **1210.6 → 410.8 MB（−66.1%）**，IME **1210.6 → 763.4 MB（−36.9%）**；K1 IME 首请求
n=59 **5134.2 → 3037.7 ms（1.69×）**。门禁 `laya_gate.sh` 报告 `SAME=42 DIFF=0`。

> 开启优化用环境变量 `VLLM_LAYA_PREPACK=1 VLLM_LAYA_ARENA=1`（默认开启 8×8 预打包零拷贝加载 +
> 生命周期不交集静态 Arena 复用，`LQ8V` 魔数缓存）。

---

### 集合 ⑥：检测 YOLO（v5 / v8 / v11）

形态：卷积网络（Conv+BN+SiLU / C2f / SPPF / Upsample / Concat / Detect head），与 LLM 完全不同。
红线：部署格式必须为引擎自有 **VQF v4**，不经过 ONNX / `.rknn`。

**1. 离线转换**（需 torch/ultralytics 环境）：

```bash
python tools/yolo/yolo_to_vqf.py --pt yolov8n.pt --out yolov8n.vqf            # F32
python tools/yolo/yolo_to_vqf.py --pt yolov8n.pt --out yolov8n_q8.vqf --q8   # per-channel int8
python tools/yolo/yolo_to_vqf.py --pt yolov8n.pt --out yolov8n.vqf --check   # 转换+读回自检
```

**2. 运行**：

```bash
./build-riscv/vllm_kestrel --yolo-model yolov8n.vqf --yolo-image photo.jpg
# 或 serve 下 POST /v1/detect
```

版本差异：v5 anchor-based、v8 anchor-free + DFL、v11 含 C2PSA 注意力，按版分别适配。

---

## 启动流程

从裸板到 HTTP 就绪共 4 步：**构建 → 权重就位 → 启动 → 验证**。

### 1) 构建（RISC-V / rv64gcv）

```bash
# K1 板上原生构建（需 gcc）
./tools/build/build_riscv.sh            # 产物 build-riscv/vllm_kestrel
./tools/build/check_riscv.sh            # 构建 + L3/Sparse 自检闸门

# 或在 Linux 主机上交叉编译
CC=riscv64-linux-gnu-gcc ./tools/build/build_riscv.sh
```

构建环境变量：

| 环境变量 | 默认 / 用途 |
|---|---|
| `VLLM_RISCV_CC` | 自动探测编译器，也可显式指定 |
| `VLLM_RISCV_MARCH` | `rv64gcv` |
| `VLLM_RISCV_OUTDIR` | 项目下的 `build-riscv` |
| `VLLM_RISCV_EXTRA` | 追加编译与链接参数 |

> 保留 `-ffp-contract=off`：乘加融合改变浮点舍入，影响参考实现对拍。K1 的 IME / `vmadot` 属硬件专用路径，
> 受核亲和性约束，默认不强制开启。

#### x86_64 原生构建（Windows / MinGW，仅自检，不作性能基准）

```powershell
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe
```

> 平台约束：引擎本质是 **RISC-V（rv64gcv）CPU 推理**，K1 是开发与基准平台。换板运行请先跑
> `--test-l3` / `--test-sparse` 自检并以结果为准。非 rv64gcv（或无 V 扩展）需调整 MARCH。

### 2) 权重就位

通用 Qwen LLM 路径加载 **VQF** 单文件（mmap 直挂）。模型目录：

```text
model/
├── model.vqf       # VQF 量化权重（单文件）
├── config.json     # 模型配置与特殊 token
├── vocab.bin       # 引擎词表，必需
└── merges.bin      # BPE 合并表，建议提供
```

```bash
python tools/build/build_vocab_bin.py <tokenizer.json> <existing_vocab.bin> <out/vocab.bin>
python tools/gen_merges_bin.py <tokenizer.json> <out/merges.bin>
```

词表必须与模型匹配；缺失 `merges.bin` 的分词回退可能改变 token 序列。

### 3) 启动服务

```bash
# 文本（稠密 / MoE / 多模态共用 --model 槽位）
./build-riscv/vllm_kestrel --serve --port 8080 --model <model-dir> --threads 8 --auto-load

# 长上下文优化档（全部可选；prefix-kv 默认开启）
./build-riscv/vllm_kestrel --serve --port 8080 --model <model-dir> --threads 8 --auto-load \
    --sparse-attn --sparse-k 32 --kv-q4 --spec --spec-k 4

# 多模型并存：chat + 嵌入 + 决策 + 检测 四槽位合并一次启动
./build-riscv/vllm_kestrel --serve --port 8080 \
    --model <chat-dir> --embed-model <emb-dir> --laya-pack <laya-dir> --yolo-model yolo.vqf \
    --threads 8
```

服务入口：`/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、
`/v1/systemone`、`/v1/detect`、`/v1/ask`，以及管理页 `/admin/`、对话页 `/chat/`。

### 4) 验证就绪

```bash
curl http://<board>:8080/health
curl http://<board>:8080/v1/models     # model id 缺省时按模型目录名推导，如 Qwen3-0.6B-q4

# 统一编排入口示例：kb 检索 → LLM
curl http://<board>:8080/v1/ask -H "Content-Type: application/json" \
  -d '{"query":"手册里散热片的告警阈值是多少？","kb_dir":"<kb-dir>","top_k":3}'
# 响应回带 "route":"rag"
```

### 5) 分布式专家并行（EP）——仅限可信网络

把 MoE 专家分片到多台机器（`--moe-ep-coord` / `--moe-ep-worker`，跨机填 `--ep-host` 对端 IP）。
EP 传输协议无鉴权/完整性校验：`--ep-host` 默认 `127.0.0.1`，跨机绑定板 IP 后端口对局域网开放，
请仅在可信网络使用，勿暴露公网。

---

## 正确性与性能验证

### 无模型自检

```bash
./tools/build/check_riscv.sh         # 板端构建 + 回归门禁
./tools/build/check_riscv.sh --build # 交叉环境仅构建
./build-riscv/vllm_kestrel --test-l3
./build-riscv/vllm_kestrel --test-sparse
```

判据：**进程退出码 0、FAIL 为 0、且 PASS+SKIP 等于用例总数（L3 17、Sparse 10）**。用总数而非固定 PASS 数，
因为每套各有 1 条随构建 ISA 打 `[SKIP]`：L3 的 (1b) NEON 对比在无 NEON 构建跳过，Sparse 的
「RVV Q8 PV candidate comparisons」在无 RVV 构建跳过。实测：rv64gcv 板端 L3 **16+1**、Sparse **10+0**；
x86-64 主机 L3 **16+1**、Sparse **9+1**。「Sparse 9 还是 10」取决于是否有 RVV，不是用例增减。

### 稠密模型基准

```bash
env OMP_NUM_THREADS=8 VLLM_THREADS=8 \
  ./build-riscv/vllm_kestrel --stream-test --model <model-dir> \
  --stream-ctx 32 --stream-n 64 --threads 8 --warmup
```

应读取日志中实际 prefill token 数；`--warmup` 预热，比较双方必须一致。

---

## 模型转换工具（vqf_convert/）

引擎只加载 VQF 单文件；原始 safetensors 由独立随版工具转换。

```bash
(cd vqf_convert && ./build.sh)
./vqf_convert/vqf_conv --model <safetensors-dir> --convert-vqf <out/model.vqf> --wmode q4 --stream
```

| 转换选项 | 用途 |
|---|---|
| `--wmode q4` | Q4 权重，较小存储占用 |
| `--wmode q8` | Q8 权重，用于更高权重量化精度 |
| `--wmode dual` | 同时保存 Q4 / Q8，转换器默认值 |
| `--wmode q4i` / `g256` | 特定预打包或分组格式，按 docs/模型适配.md |
| `--stream` | 降低转换峰值内存，适用于大模型（30B） |

> 量化精度与布局是两个维度：`--wmode q4` 不等于自动获得 col-major 布局，复现必须核对 VQF 布局与生成工具。
> 多模态转换时视觉塔 Q8 权重以 `VQF_FLAG_VISION` 标记存于同一 `model.vqf`，运行时由 `VLLM_VIS_RVV=1`
> 重排为 8×8 tiled。

---

## 权重保护与推理验证（attestation）

**产出侧**：置 `VLLM_VQF_KEY` 输出 VQF-Enc 加密（SM4-CTR + HMAC-SM3）；置 `VLLM_VQF_SIGN_PRIV` 内嵌
SM2 供应链签名，二者可叠加。
**加载侧**：置 `VLLM_VQF_KEY` 解密、`VLLM_VQF_SIGN_PUB` 验签；错口令 / 篡改字节均被拒绝。置
`VLLM_ATTEST=1` 开启逐响应出证（schema 含请求原文绑定），`GET /v1/attest` 下发设备公钥，
`tools/security/verify_attest.py` 离线验签（退出码 0=PASS / 1=FAIL）。

> 加密/签名文件需全层驻留；密钥目录须放在支持 POSIX 权限的文件系统（ext4/f2fs）。

---

## 体量与依赖（小而全）

| 项 | 数值 / 口径 |
|---|---|
| 可执行文件 | `vllm_kestrel` ≈ **1.3 MB**（rv64gcv Release，K1 板端实测 **1,302,384 B**），零第三方 .so |
| 源码 | 纯 C11，单工程单产物（include + src + serve + model + tools） |
| 运行时依赖 | 无第三方运行时——标准构建仅需 gcc + libm |
| 代码内第三方 | 仅 `stb_image.h`（MIT, Sean Barrett）与 llama.cpp 派生 4×4 asm 内核（MIT, The ggml authors） |
| 自研件 | RVV 量化 GEMM/GEMV、线程池 `vllm_tp`、国密 SM3/SM4/SM2、VQF 单文件 mmap 格式 |

### 零第三方依赖（自证清单）

| 常见依赖项 | 本项目实际情况 |
|---|---|
| 推理框架运行时（Python / PyTorch / vLLM 栈） | 无——单一 C11 可执行文件即完整 HTTP 服务 |
| GPU 计算栈（CUDA / ROCm） | 无——纯 CPU + RVV 量化内核 |
| 第三方推理/矩阵库（ggml、OpenBLAS、oneDNN…） | 无——GEMM/GEMV 手写（llama.cpp 派生 4×4 asm 为 MIT 提取件） |
| 密码学库（OpenSSL / GmSSL / MbedTLS…） | 无——SM3 / SM4-CTR / HMAC-SM3 / SM2 全自研并经国密标准 KAT 验证 |
| 图像/视频解码库（OpenCV / FFmpeg…） | 无——图片解码用单头 `stb_image.h`（MIT） |
| Web/HTTP 与 JSON 库 | 无——自研 HTTP + SSE、自研最小 JSON 解析 |

---

## 目录结构

```
├── README.md / README.en.md      # 中文 / 英文说明
├── LICENSE                       # Apache-2.0
├── include/                      # 公共头（common/core/media/model/serve）
├── src/
│   ├── main.c                    # 命令行 / 服务 / 测试入口
│   ├── core/                     # 线程池、调度、KV、注意力、L3、变换器、EP
│   ├── model/                    # vqf.c、vllm_safetensors.c、vllm_vision.c、vllm_laya.c、vllm_yolo.c
│   └── serve/                    # HTTP、连续批、会话、协同编排、管理页
├── tools/
│   ├── build/                    # build_riscv.sh / check_riscv.sh / build_x64.ps1 / check_x64.ps1
│   ├── bench/                    # 基准与布局重排（vqf_down_relayout / gu_lane_repack / dn_lane_repack）
│   ├── client/                   # vllm_client.py（OpenAI 兼容客户端，零依赖）
│   ├── security/                 # verify_attest.py + vllm_vqf_sign.c
│   └── yolo/                     # yolo_to_vqf.py 与板端分阶段脚本
├── vqf_convert/                  # 独立权重转换器（safetensors → VQF）
└── docs/                         # 模型适配说明（中文）
```

---

## 文档（docs/，中文）

| 文档 | 内容 |
|---|---|
| [docs/模型适配.md](docs/模型适配.md) | 六类模型的准备、接入、词表与转换约束（Dense / MoE / VL / Embedding / Laya / YOLO），含架构自动识别与 checklist |

---

## 变更摘要（相对上一公开形态）

**工程与形态**

- **平台化**：同一二进制 + 四类运行槽位（`--model` / `--embed-model` / `--laya-pack` / `--yolo-model`）
  承载六类模型，`/v1/ask` 统一编排四链（chat / rag / decide / scene）。
- **纯 VQF 运行时**：`model.vqf` 单文件 mmap 直挂、按需建页；量化布局随张量固化。
- **对外的 model id 修复**：`/v1/models` 等接口返回 `id`/`model` 不再硬编码，改为显式 `--model-id` →
  已加载模型目录 basename（如 `Qwen3-0.6B-q4`）→ 兜底 `kestrel`。

**性能与内存**

- **布局默认化**：`layout=2`（lane-major）默认启用 RVV（修复标量退化）；`layout=3`（col-major）默认开启
  DEC（`VLLM_Q4_DEC`），decode 吞吐 +25%~32%。
- **MoE 分层驻留**：`VLLM_VQF_STREAM` + EC + GU/DN lane + `DN_SAPF`，17 GB 级 MoE 在 8–16 GB 上可服务。
- **Laya 预打包 + Arena**（`VLLM_LAYA_PREPACK` / `VLLM_LAYA_ARENA`）：加载 13.79~26.91×，峰值内存 −36.9%~66.1%。
- **视觉塔 RVV**（`VLLM_VIS_RVV=1`）：图片编码 13~14.5×，24 层位级对拍 0 失配。

**正确性**

- **自检闸门修复**：`check_riscv.sh` / `check_x64.ps1` 判据改为「退出码 0 + FAIL=0 + PASS+SKIP==用例总数
  （L3 17 / Sparse 10）」，并校验子进程退出码。

---

## 许可与合规

- **许可：Apache License 2.0**（SPDX：`Apache-2.0`）——代码全部公开，可自由使用、修改、分发、商用，
  无需另行授权。要求保留版权与许可声明、标注修改过的文件（[LICENSE](LICENSE) 第 4 条）；第 3 条含明示
  专利授权（附专利报复条款）。
- **历史版本**：此前版本若采用不同许可，按当时文件头与版本为准、不可撤回。
- **第三方组件**按各自许可保留：`stb_image.h`（MIT, Sean Barrett）与 llama.cpp 派生 4×4 asm 内核
  （MIT, The ggml authors），均与 Apache-2.0 兼容。

## 联系

**QQ：398152090 · 邮箱：398152090@qq.com**
（Gitee Issues / 站内私信联系作者。For research collaborations / data requests / other enquiries, open an
issue at https://gitee.com/pei-xiaoguang/kestrel-risc-v/issues.）