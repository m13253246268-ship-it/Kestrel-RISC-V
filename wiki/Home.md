# vLLM-Kestrel Wiki

> Axiom-Guided Exact-Attention edge LLM runtime — 项目文档索引

vLLM-Kestrel 是一个纯 C 实现的边端 LLM 推理引擎：OpenAI 兼容 HTTP 服务、自研 VQF 量化格式、
连续批调度、KV 分层驻留、MoE 专家并行、多模态（Qwen3-VL / YOLO）与嵌入检索。

## 快速入口

- [README.md](../README.md) ｜ [README.en.md](../README.en.md)

## 核心文档

| 文档 | 内容 |
|---|---|
| [README.md](../README.md) | 项目总览：目录结构、模型格式、编译、启动、客户端 |
| [docs/模型适配.md](../docs/模型适配.md) | 各类模型（Dense / MoE / 多模态 / Embedding / Laya / YOLO）的适配方式 |
| [tools/preproc/README.md](../tools/preproc/README.md) | 数据预处理脚本说明 |

## 快速开始

```bash
# 1) 构建（x86-64 主机，MinGW-w64）
powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1

# 2) 构建（RISC-V 64）
./tools/build/check_riscv.sh

# 3) 转换模型（safetensors -> VQF）
vqf_conv --model <hf_dir> --convert-vqf <out>/model.vqf --wmode dual

# 4) 启动服务
./build-riscv/vllm_kestrel --serve --port 8080 --model <out> --threads 8
```

## 当前状态

- 板端（RISC-V）暂时不可用；x86-64 可编译并跑数值自检（非性能基准）。
- 回归闸门只证明「能编译 + 自检通过 + 跨架构同计数」，不构成性能结论。
