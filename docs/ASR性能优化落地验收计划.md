# ASR 性能优化落地验收计划

> 最新只读收尾状态（2026-11-18 08:32:58，Asia/Shanghai；本地解析 08:33:52）：原目录 `model_status.json=complete`，PID 57789 及其直属子进程已退出；127 条执行记录中正式样本 **90/90** 均有退出码且全部为 0，末项为 `profile_w8_D_t8_r5`。本轮仅定点收集已关闭的 90 份正式日志、90 份环境 JSON 及现成汇总，未扫描远端目录、重启、补验或改环境。36/36 正确性、18/18 对拍、78/78 层级检查继续通过；正式样本 requested/actual 均匹配档位、decode/generated/selected=11/11/12、指定 19 叶桶非负有限及闭合检查均通过，126 份完成日志无样本检查失败。18 组各五次齐全，但 **P0/P1 与性能仍未通过**：缺逐样本 governor、独立逐样本 trace、完整板端 Python/WAV 双门禁，且仅一段冻结输入、未覆盖 P7 2/4/16 线程。稳定性有 7/18 组超过极差/中位数 ≤3%：F32 B/t1 3.013%、A/t8 4.943%、default/t8 5.881%；W8 default/t8 5.192%、B/t8 10.782%、C/t8 7.621%、D/t8 6.409%。W8 D/t8 相对 C 的 conv1、projector 等叶桶还出现 >5% 回退，待解释。最终数值汇总见 `W/board_continuation_results.json`，`five_sample_matrix_passed=true` 只表明五次样本检查齐全，`environment_review_passed=null`、`performance_passed=false`；不得将数值比值阈值满足当作阶段验收通过。下列首样本和过程快照保留为历史记录。
>
> 以下为上一份首样本快照（2026-11-18 01:43:25，Asia/Shanghai）：原 PID 57789 未重启，36/36 正确性推理、18/18 对拍保持通过；本次自动层级负例复跑已 **78/78 通过且整体退出 0**（715.656974 秒）。正式 profile 已完成 **1/90**，`profile_f32_A_t1_r1` 退出 0，total=437.934805544 秒、wall=437.950913 秒，requested/actual=1/1，decode/generated/selected=11/11/12；具体 19 叶桶齐全、非负有限，排除 encoder aggregate 后闭合相对误差为 0，严格筛查及计划 ≤2% 检查均通过。启动环境为 loadavg 3.18/3.71/4.59、温区 40/41°C、8 核 scaling_cur_freq 均 1600000 kHz；逐样本 governor 仍缺失，不能据此通过完整环境审阅。
>
> 01:43:24 进程证据：第二个正式样本 `profile_f32_default_t1_r1` 已由原编排自动启动，子 PID 133694 已运行 1 分 03 秒、CPU 100%；原 PID 57789 运行 2 小时 12 分 42 秒、状态 running。本地采集器取得首样本后按条件退出，板端继续自动运行剩余 89 次（含当前样本）。本轮未重启、叠加板端计算、修改板端编排或生产代码，未放宽标准。首组 A 的五次需按 A/default/B/D 交错顺序至少完成第 17 个正式样本，本次未等待长时矩阵完成。
>
> 证据已更新至下文定义的 `W/board_progress_snapshot.json`、`board_progress_watch_20261118.json`、`board_model_execution.json`、`board_layers_negatives.json`、`board_profile_f32_A_t1_r1.log`、`board_profile_f32_A_t1_r1_env.json` 和 `board_continuation_results.json`。解析结果 `layers_negatives_passed=true`、`formal_sample_count=1`，执行与样本验证失败清单为空；`five_sample_matrix_passed=false`、`environment_review_passed=null`、`performance_passed=false`。**单个样本不构成五次稳定性、收益或 P0/P1 性能验收结论**；共享 trace、单冻结输入、完整 Python/WAV 双门禁及 P7 线程覆盖限制继续保留。下文各时间戳为历史快照，先前“正式样本 0”的记录不代表当前状态。

适用对象：`src/model/vllm_asr.c`、`src/model/vllm_asr_audio.h`、`src/model/vllm_asr_w8_rvv.h`、
`src/model/vllm_asr_wav.h` 以及 `tools/asr/` 下的门禁与单测。

本文档定义「每一项优化做到什么程度算通过」。所有验收结论必须由可复现的命令与落盘产物支撑，
不接受口头结论，也不接受「看起来快了」这种判断。

---

## 0. 不变量（任何阶段都不得违反）

| 编号 | 约束 | 判定方式 |
|---|---|---|
| I1 | 编译时间禁用 FMA 收缩 | 编译命令必须包含 `-ffp-contract=off` |
| I2 | 浮点累加顺序与标量参考一致：输出 lane 分块 → k 递增 → bias 最后加 | 代码评审 + `asr_w8_test.c` 位级 `memcmp` |
| I3 | C 端不引入 Python/ONNX Runtime/torch 依赖 | `asr_c_wav_gate.py` 的 `binary_imports.log` 断言 |
| I4 | VQF 全文件 FNV 校验、形状白名单、图哈希校验保持原样 | `asr_c_gate.py` 的负例用例全绿 |
| I5 | 每条优化路径必须有编译期逃生舱（宏或 `VLLM_ASR_SCALAR`） | 逃生舱关闭后回到标量路径且门禁全绿 |

红线：I1 与 I2 是独立硬约束；违反它们不保证立即产生 token 或容差失败，但仍不得验收。
C 标量/优化内核的位级相等与 Python 参考的 `max_abs ≤ 1e-3`、token 完全相等分别验收，不能互相替代。
全文收益百分比和倍数均为预注册目标，不是已测收益；没有板端数据时不得标为完成。

---

## 1. 阶段总览

| 阶段 | 内容 | 预期收益 | 风险 | 前置 |
|---|---|---|---|---|
| P0 | 基线冻结 + 剖析 | 无（建立度量） | 低 | 无 |
| P1 | 打开 RVV / CONV_PACK 构建开关 | 高 | 中（需位级回归） | P0 |
| P2 | encoder attention 向量化 | 高 | 中 | P1 |
| P3 | `aconv` 卷积本体向量化 / 去冗余 im2col | 中高 | 中 | P1 |
| P4 | LM logits 投影与 argmax 优化 | 中高 | 中 | P1 |
| P5 | W8 tile 转置与 `qvalue` 取模消除 | 中 | 低 | P1 |
| P6 | `affn_pack` 持久化 | 中（短音频/多次调用） | 中（内存） | P1 |
| P7 | 并行策略（chunk/层/token 级） | 中（多核） | 中高 | P1 |

排序依据：P1 不写算法、只改参数，是唯一「低成本高上限」项，必须先做并把收益量化，
后续阶段的优先级要按 P0/P1 的真实剖析数据重新校准，而不是照本文档的静态判断。

---

## P0 基线冻结与剖析（所有验收的参照系）

### 目标
得到一份可复现的、分 stage 的耗时基线，作为后续每个阶段「提速 X%」的唯一起点。

### 执行步骤
1. 固定输入：1 段真实音频（建议 5–8 秒，含语音内容）+ 门禁自带的 9 个边界 WAV。
2. 固定编译参数（现状）：`-std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off`，
   默认无显式 `-march`、无 ASR 宏，但编译器默认 ISA 可能启用 RVV，不能据此称为标量。
   另建显式 `-DVLLM_ASR_SCALAR` 对照并记录编译器预定义宏。F32/W8 模型分别建基线，
   同一对比必须使用相同模型 SHA256；现有 Python 参考支持 RowW8，同 W8 VQF 可重新导出 map/assets 并运行门禁。
   同模型 C/Python 容差、C 标量/RVV 位级一致性与 W8 相对 F32 量化质量分别验收。
3. 固定线程：`VLLM_THREADS=1` 与 `VLLM_THREADS=8` 各跑一轮。
4. 打开剖析：`VLLM_ASR_PROFILE=1`，采集全部 19 个叶子桶（含 unclassified/load/assets/trace/text）。
   `encoder aggregate=1` 是汇总桶，不计入求和；total 包括加载、资产、特征、trace/有限性检查和文本处理。
   trace 开关与输入模式必须相同；若另报纯推理耗时，必须明确所加叶子桶及排除项。
5. 每档重复 5 次，取中位数；同时记录 `decode_steps`、`generated_tokens`、`selected_tokens` 用于确认
   各次运行的解码长度一致（长度不同则耗时不具可比性）。

### 交付产物
- `baseline/profile_<arch>_t<n>_r<i>.log`：5 次原始 stderr。
- `baseline/baseline.md`：中位数表，列为 19 个 stage + total + decode_steps。
- `baseline/env.txt`：`cc --version`、`uname -a`、`/proc/cpuinfo` 的 ISA 字段（含 VLEN 若可读）、
  模型文件 SHA256。
- `baseline/binary_sha256.txt`：被测二进制哈希。

### 验收标准
- [ ] 同一档 5 次重复的 total 极差 ≤ 3%（超出则说明机器状态不稳，需重测或换机）。
- [ ] 19 个 stage 秒数之和与 total 的差异 ≤ 2%（超出说明剖析点有遗漏，属待查缺陷）。
- [ ] `decode_steps` 与 `generated_tokens` 在 5 次之间完全一致。
- [ ] 基线二进制的 `asr_c_gate.py` 与 `asr_c_wav_gate.py` 全绿（证明基线本身是合格版本）。

### 通过后动作
把 P1–P7 的「预期收益」列按实测占比重写一版，标注每项的理论上限（该 stage 占 total 的百分比）。
这一版表格是后续所有「提速 X%」指标的来源，必须落盘。

---

## P1 打开 RVV / CONV_PACK 构建开关

### 现状
两个门禁默认未传 ISA/ASR 开关；现已支持可选 `--cflags`，默认编译命令不变。
F32 RVV 路径由 `__riscv_vector && !VLLM_ASR_SCALAR` 控制；W8 RVV 另需
`VLLM_ASR_W8_RVV`，卷积打包另需 `VLLM_ASR_CONV_PACK` 且未启用 CONV_SCALAR。
`affn_pack` 还用于 FFN 等路径，不能笼统说全部未编译。
参数经 `shlex.split` 拆分，仅接受 `-march/-mabi/-mcpu/-mtune=VALUE` 和现有路径宏
（SCALAR/W8_RVV/CONV_PACK/CONV_SCALAR/FFN_SCALAR/TAIL_SCALAR/QKV_SCALAR/OUT_SCALAR，允许 `=1`）。
不开放任意编译参数：fast-math、浮点收缩覆盖、响应文件、编译器转发参数均被拒绝。
示例：`--cflags="-march=rv64gcv -mabi=lp64d -DVLLM_ASR_W8_RVV -DVLLM_ASR_CONV_PACK"`。
两个门禁均编译后立即本机执行，不是跨架构 runner；交叉构建必须单独完成，再到目标板运行。

### 改动范围
- `tools/asr/asr_c_gate.py`、`tools/asr/asr_c_wav_gate.py`：新增 `--cflags`（或 `--march`）参数，
  默认值保持现状，保证默认路径行为不变。
- 新增板上运行的验收脚本 `tools/asr/asr_bench.py`（若最终选择脚本形式），只负责
  编译 + 跑 profile + 汇总，不改生产源码。

### 执行步骤
1. 交叉编译验证：x86 主机用 `riscv64-linux-gnu-gcc -march=rv64gcv -DVLLM_ASR_W8_RVV -DVLLM_ASR_CONV_PACK`
   能编过（`-Werror -pedantic` 不许有告警）。
2. 板上运行四档组合（另保留不传参数的默认命令基线）：
   - A：`-march=rv64gcv -DVLLM_ASR_SCALAR`（显式标量参考）
   - B：`-march=rv64gcv`（F32 RVV 路径，无额外 ASR 宏）
   - C：`-march=rv64gcv -DVLLM_ASR_W8_RVV`
   - D：C + `-DVLLM_ASR_CONV_PACK`
   F32/W8 均跑各自模型的 Python/完整 WAV 门禁；W8 使用重新导出的 map/assets。
   同一 W8 模型在 A/B/C/D 间做 C 位级对拍，另做 W8 相对 F32 的量化精度验收。
   F32 输入不能证明 W8 内核收益；不得将 F32-A 与 W8-C 混成编译开关收益。
3. 每档记录：hidden max_abs、projector max_abs、每步 logits max_abs、token 序列、
   以及 P0 定义的 19 个 stage 耗时。

### 验收标准
功能：
- [ ] A/B/C/D 四档的 token 序列与 Python 参考逐 token 完全相等。
- [ ] 四档的 `mask` 390 项完全一致。
- [ ] 四档 `hidden` / `projector` / 每步 `logits` 的 `max_abs ≤ 1e-3`。
- [ ] 同模型 C 标量/优化内核必须通过位级比较；Python 数值容差另验。WAV 门禁的同一冻结 Mel 回归要求字节相等，不能用 1e-3 替代。

性能：
- [ ] C 相对 A 的 total 提速 ≥ 1.3×（若显著低于此值，先查 `quant_linear` 白名单是否覆盖了主要形状，
      白名单位于 [vllm_asr.c](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr.c#L145-L159)）。
- [ ] D 相对 C 不出现回退（任何 stage 恶化 > 5% 需给出解释并登记）。

工程：
- [ ] 默认参数（不传 `--cflags`）下的编译命令与 P0 基线逐字相同（用字符串比对断言）。
- [ ] `asr_w8_test.c` 在 C/D 两档编译下全部通过，包含 `--bad-scale` 的 nan/zero/negative/inf 四个负例。

### 回滚
`-DVLLM_ASR_SCALAR` 一行即可回到参考路径；该逃生舱需在本阶段结束时实测验证一次。

---

## P2 encoder attention 向量化

### 现状
`audio_encode()` 内 18 层 × 14 头 × 104 窗口三重标量循环，double 累加，逐元素 `expf`，
见 [vllm_asr_audio.h](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr_audio.h)。

### 改动范围
仅 encoder attention 的 QK^T、softmax、PV 三段；不得改动 QKV 投影与 FFN。

### 验收标准
功能与精度：
- [ ] `VLLM_ASR_PROFILE` 中 `attention` 桶的耗时下降 ≥ 40%（相对 P1-D 档）。
- [ ] 端到端 token 序列不变；`hidden` max_abs ≤ 1e-3。
- [ ] 新增一个注意力级单测（可复用 `asr_ffn_test.c` 的骨架）：对同一份 q/k/v 输入，
      标量路径与向量路径逐元素 `max_abs ≤ 1e-4`，且 softmax 归一化后每头和为 1±1e-5。
- [ ] 累加顺序与标量一致（lane 分块、k 递增），由该单测的位级比较断言。

性能：
- [ ] 14 头循环不被串行化打断；若引入多线程，必须证明 `VLLM_THREADS=1` 下不退化。
- [ ] 不使用 `-ffast-math`，不允许把 `expf` 换成近似实现（除非单测证明误差在 1e-4 内）。

风险登记：
- softmax 中 `max` 归约若用向量归约，需保证 NaN/Inf 输入下的行为与标量一致（用负例覆盖）。

---

## P3 `aconv` 卷积本体向量化

### 现状
`aconv_out` 已走 RVV m4；前三层卷积仍是「逐输出点手工构造 `col[in]` → `aline`/`affn_down`」，
`col` 每个输出点重建一次。

### 验收标准
- [ ] `conv0`/`conv1`/`conv2` 三个桶合计耗时下降 ≥ 30%（相对 P1-D）。
- [ ] 卷积层输出与标量路径逐元素 `max_abs ≤ 1e-4`，用 `asr_conv_out_test.c` 的同类骨架新增
      三层卷积的单测（现有 `asr_conv_out_test.c` 只覆盖 conv_out，需补 conv0–2）。
- [ ] 输入边界（`length` 极小、`frames` 为 0、`mask` 全零的 chunk 跳过路径）不出现越界：
      用 `-fsanitize=address,undefined` 编译跑一遍边界用例，零报告。
- [ ] `VLLM_ASR_CONV_SCALAR` 逃生舱关闭新路径后，门禁仍全绿。

---

## P4 LM logits 投影与 argmax

### 现状
`step()` 末尾 `linear(logits,z,w[0],D,VOC,NULL)` 为 151936×512，`linear_row` 是 double 标量；
argmax 为 `for(t=0..VOC)` 全扫并跳过 5 个音频占位 id。

### 验收标准
精度（此阶段最容易踩线，指标要更严）：
- [ ] 每步 `logits` 的 `max_abs ≤ 1e-4`（比门禁 1e-3 更严，避免后段 decode 漂移累积）。
- [ ] 每个 decode 步选出的 token 与参考完全一致；`generated_tokens` 数量一致。
- [ ] argmax 的并列（tie）行为与标量完全一致：构造一个含大量相等 logits 的合成用例，
      验证「取首个最大值」在小 id 与大 id 两侧都成立。
- [ ] 占位 id 151646–151650 的跳过逻辑在向量化后仍精确生效（用合成 logits 把这 5 个 id 置为全局最大，
      断言不会被选中）。

性能：
- [ ] `decode` 桶耗时下降 ≥ 25%（相对 P1-D）。
- [ ] `prefill` 桶不退化（> 3% 恶化需说明）。

约束：
- [ ] 不允许「跳过 logits 只算 argmax」这类不等价改写（会破坏 trace 输出与门禁对拍）。
- [ ] trace 模式下 dump 的 `VOC` 个 float 必须与标量路径逐字节相同（trace 是门禁的输入）。

---

## P5 W8 tile 转置与 `qvalue` 取模

### 现状
- 非转置分支在 [vllm_asr_w8_rvv.h](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr_w8_rvv.h#L21-L33)
  用 `int8_t tile[32][W8_LANES]` 做 512 次标量逐字节搬运。
- `qvalue()` 每元素一次 `index%rows` 与 `index/rows`，见
  [vllm_asr.c](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr.c#L114-L117)。

### 验收标准
- [ ] `asr_w8_test.c` 的形状矩阵 `{1,1},{31,17},{33,31},{513,527},{896,896},{896,3584},{3584,896},{1024,4096},{4096,1024},{7680,896}`
      在 transposed × wide × bias 全组合下仍是**位级** `memcmp` 相等。
- [ ] `--bad-scale` 四个负例（nan/zero/negative/inf）仍被拒绝。
- [ ] 若改为「加载时一次性转置」，需证明加载后内存增量 ≤ 原权重体积的 1.0 倍，
      且 `load` 桶耗时增幅 ≤ 20%（用 P0 的 load 桶对比）。
- [ ] `qvalue` 改动后 `qlinear_row`/`qaline_row` 输出位级不变（同一 `asr_w8_test` 覆盖）。

性能：
- [ ] `norm_qkv` + `ffn` + `tail_mlp` 合计下降 ≥ 10%（相对 P1-D），或在剖析中证明该路径已被 P1 完全覆盖而无需再改。

---

## P6 `affn_pack` 持久化

### 现状
18 层 encoder + 4 层 tail + 3 层 conv，每层重建打包缓冲，峰值约 24.5 MiB。

### 验收标准
- [ ] 单次推理的 `encoder_other` 与 `tail_mlp` 中的打包开销归零或 < 1% total。
- [ ] 峰值 RSS 增量 ≤ 现有峰值的 1.2 倍；用 `tools/bench/rss_stat.sh` 记录前后峰值。
- [ ] 同一进程内连续跑 3 次推理，第 2、3 次的耗时不超过第 1 次的 105%（证明打包确实只做一次）。
- [ ] 释放路径无泄漏：ASan/LSan 全绿。
- [ ] `VLLM_ASR_CONV_PACK` 关闭时行为与 P1-A 基线一致。

---

## P7 并行策略

### 现状
`asr_parallel` 只按单次 matvec 的输出块切分，门槛为「每参与者 ≥ 32768 次乘加且至少一个输出块」。
`audio_encode` 的 30 个 chunk、每层 token 循环均为天然并行维度但未使用。

### 验收标准
- [ ] `VLLM_THREADS=1` 时所有阶段耗时与 P6 结束态差异 ≤ 3%（并行不得污染单线程）。
- [ ] `VLLM_THREADS=8` 时 total 相对单线程加速比 ≥ 2.0×（低于此值说明同步开销吃掉收益，
      需调整粒度或改任务级并行）。
- [ ] 8 线程下的 token 序列与单线程完全一致（并行不得引入非确定性）。
- [ ] 线程数 2/4/8/16 四档的 total 单调不增（允许 ±5% 抖动）；出现明显倒挂需登记原因。
- [ ] `asr_tp_test.c` 固定请求 8 线程，注入创建失败覆盖实际参与者 1–8；它不读取 VLLM_THREADS，不能冒充线程环境变量矩阵。另用生产程序验证 1/2/4/8。
- [ ] 嵌套并行保护：`vllm_tp_worker_id()<0` 判断保持，worker 内部不得再次 fork。

---

## 2. 全量回归矩阵（每个阶段结束都要跑）

| 用例 | 命令 | 通过判据 |
|---|---|---|
| hidden 回归 + 负例 | `python tools/asr/asr_c_gate.py --vqf <model> --audio <wav> --out <dir> --cc gcc` | 退出码 0，所有 max_abs ≤ 1e-3，负例全部被拒 |
| WAV 端到端 | `python tools/asr/asr_c_wav_gate.py --vqf <model> --audio <wav> --out <dir> --reference <ref> --cc gcc` | 文本与参考完全相等，mel 模式字节级相等 |
| W8 内核位级 | 编译 `tools/asr/asr_w8_test.c` 并运行 | 全形状 memcmp 相等，`--bad-scale` 四负例通过 |
| FFN 单层 | 编译 `asr_ffn_test.c`，运行 `<exe> <model.vqf> <map> <fc1_input.bin> <fc2_input.bin>` | 生产内核 memcmp 位级相等；需真实捕获输入 |
| conv_out 单层 | 编译 `asr_conv_out_test.c`，运行 `<exe> <model.vqf> <map> <conv_flat.bin>` | 11 个形状 memcmp 位级相等；输入为 7680 个 FP32 |
| 多线程 | 编译并运行 `tools/asr/asr_tp_test.c`（该文件已包含线程池实现，不另链接） | 固定请求 8、实际 1–8 的创建降级/嵌套/关闭测试全绿 |
| W8 格式 | `python tools/asr/asr_w8_format_test.py <asr_w8_test.exe>` | 5 个格式负例及 4 个 scale 负例拒绝 |
| 内存 | `-fsanitize=address,undefined` 跑 WAV 全用例 | 零报告 |

补充要求：
- [ ] 每次阶段验收都保存完整 stdout/stderr 与产物目录，不得只保留「通过」两个字。
- [ ] 每个阶段用同一段音频、同一模型文件、同一线程数，否则性能数字不可比。

---

## 3. 性能验收的统一方法

1. 计时口径：使用 `VLLM_ASR_PROFILE=1` 的输出；total 自身包含文件加载、资产和 trace 等，不等于纯推理。外部墙钟仅作补充，不混用；求和只取 19 个叶子桶，不重复加入 aggregate encoder。
2. 采样：每档 5 次取中位数；报告中同时给出最小/最大值。
3. 回归判定：某 stage 恶化 > 5% 且绝对值 > 10 ms，即视为回退，必须解释或修复。
4. 端到端指标：单独记录「5 秒音频的 total 秒数」与「每 token 解码秒数」，写入阶段报告首页。
5. 机器状态：每次测量前记录 CPU 频率策略与热状态；跨会话的绝对值不算证据，只认同会话 A/B。

---

## 4. 阶段报告模板（每阶段一份，落盘 `docs/asr_perf/<阶段>.md`）

```
阶段：P?
日期：
代码基线：git commit / 工作区 diff 摘要
编译参数：（完整命令行原文）
机器：CPU / ISA（含 VLEN）/ 内存 / 调度策略
输入：音频 SHA256 / 采样率 / 时长 / 帧数
线程：VLLM_THREADS=?

分数表：
| stage | 基线(s) | 本阶段(s) | 变化 | 占 total |
| ...   |         |           |      |          |
| total |         |           |      |          |

精度：
| 项 | max_abs | 阈值 | 结论 |
| hidden |  | 1e-3 |  |
| projector |  | 1e-3 |  |
| logits(每步最大) |  | 1e-4 |  |
| token 序列 | 完全相等 / 不等 | — |  |

回归矩阵：8 项逐项 PASS/FAIL（附日志路径）
逃生舱验证：关闭开关后的门禁结论
风险与遗留：
结论：通过 / 有条件通过（附条件）/ 不通过
```

---

## 5. 风险登记表

| 编号 | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 向量化改变累加顺序，位级对拍失效 | 高 | 严格保持 lane 分块 → k 递增 → bias 最后；每阶段必跑 `asr_w8_test` |
| R2 | `-march=rv64gcv` 后 VLEN 相关的 `vsetvl` 行为在目标板上与仿真不同 | 高 | 板端实测为准；仿真只用于编译验证，不作为性能证据 |
| R3 | argmax 并列行为改变导致 token 漂移 | 高 | 合成 tie 用例 + 5 个占位 id 置最大值的用例 |
| R4 | 持久打包推高内存，触发板端 OOM | 中 | 记录 RSS 峰值；提供按需释放开关 |
| R5 | 任务级并行引入非确定性 | 中 | 8 线程与单线程 token 序列必须完全一致 |
| R6 | 门禁脚本改动误伤默认路径 | 中 | 默认编译命令字符串与基线逐字比对 |
| R7 | 单阶段收益被另一阶段回退抵消 | 中 | 每阶段独立 A/B，禁止多阶段合并验收 |

---

## 6. 验收门（Definition of Done）

一个阶段只有在下列全部满足时才算通过：

1. 第 2 节的 8 项回归全绿，且有日志。
2. 第 3 节口径下的性能收益达到该阶段写明的百分比阈值。
3. 精度项达到该阶段阈值（注意 P4 的 1e-4 比通用 1e-3 更严）。
4. 逃生舱实测有效。
5. 阶段报告按模板落盘。
6. 风险登记表已更新（新增风险或关闭风险都要写）。

整体收尾：P1–P7 全部通过后，出一份汇总报告，包含
「基线 vs 最终」的 19 个 stage 对照表、端到端 5 秒音频耗时对照、峰值 RSS 对照、
以及每个优化项的「实测收益 / 理论上限」比值——用于判断是否还有下一轮空间。

---

## 7. 交付物清单

| 类别 | 路径 | 说明 |
|---|---|---|
| 基线 | `baseline/` | 5 次 profile 原始日志、环境信息、二进制哈希 |
| 各阶段报告 | `docs/asr_perf/P1.md` … `P7.md` | 按第 4 节模板 |
| 回归日志 | `<阶段目录>/gate_*.log` | 8 项矩阵逐项日志 |
| 汇总 | `docs/asr_perf/summary.md` | 基线 vs 最终对照 |
| 代码改动 | `src/model/*`、`tools/asr/*` | 独立可回滚的改动；提交/推送须有明确授权，本轮未执行 |

## 8. 本轮实际进度（2026-11-17）

状态：P0 准备与 P1 参数入口完成；**P0 基线冻结、P1 板端精度/性能验收尚未完成**。
没有新增内核、没有修改浮点默认参数，没有生成或宣称任何性能基线/收益。

### 环境与资产核实

- 在工作区、目标子目录及 `E:/`、`E:/项目/` 祖先规则入口未找到 AGENTS.md、CLAUDE.md 或 `.trae/rules`；不据代理口述追加规则。
- 本机 Windows x64：Python 3.12.9，MinGW GCC 16.1.0，objdump 可用。
  NumPy/tokenizers/torch/transformers/scipy/onnx/onnxruntime 可导入定位；soundfile/librosa 未安装。
  门禁 `--help` 实际导入成功；不据 soundfile 缺失直接判定现有 WAV 路径不可运行。
- WSL Ubuntu 已配置 `riscv64-linux-gnu-gcc` 15.2.0，交叉构建可用。
- `../modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/` 有 F32 ONNX、tokenizer、feature 配置、embedding 与 projector。
  三个 F32 主图检查未发现外置 initializer 引用。源模型存在，不应写成“无模型”。
- 真实音频 `../modls/iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav`：
  PCM16、单声道、16000 Hz、88747 帧、5.5466875 秒。
- 在项目/model 目录及 TEMP 的 asr* 目录搜索未发现配套 ASR F32/W8 VQF、asr_lm.map、alignment.json、mel.bin/output.bin、C WAV assets，
  也没有 FFN/conv_out 捕获输入。本轮未做大模型转换和参考生成，这些是下一步准备工作，不等于源模型缺失。
- SSH config 只有 GitHub 别名；已有项目脚本指向 localhost:2222 隧道及另一历史局域网地址。
  仅对已配置 localhost:2222 做 2 秒 TCP 探测，返回 10035（未建立连接）；未连接历史局域网地址或未配置设备。
  因此板端 ISA/VLEN、频率、热状态、模型及运行时尚未核实，不能形成性能结论。

### 修改与验证

`asr_c_gate.py` 定义共用 `parse_cflags`，`asr_c_wav_gate.py` 复用；不增加模块文件。
默认空列表不改变原命令，显式参数插入在编译器名后，固定严格浮点参数保留。
以下命令从仓库根目录执行；本轮输出保留在执行会话记录，未伪造正式基线日志。
临时二进制在 `%TEMP%/*_p01.exe` 与 WSL `/tmp/`，不提交。

| 验证 | 结果 |
|---|---|
| `python tools/asr/asr_c_gate.py --help` / `python tools/asr/asr_c_wav_gate.py --help` | 两者退出 0，显示 --cflags |
| 内存 AST 编译与 argparse 检查（无新增测试文件） | 两脚本语法通过；3 组合法值、13 组拒绝值通过；错误 CLI 返回 2，先于资产读取/目录创建 |
| AST 提取原 cmd 并应用空参数/非空插入 | 空参数逐元素保持原 argv；非空参数准确插入，保留唯一 -ffp-contract=off |
| MinGW 严格编译 W8、线程池、FFN、conv_out 四个单测 | 全部退出 0 |
| `%TEMP%/asr_w8_test_p01.exe` | 全形状/两布局/FP32与FP64累加等位级 oracle 通过（x64 回退路径） |
| `%TEMP%/asr_tp_test_p01.exe` | create_limit=0..7，actual=1..8，static/dynamic/nested/shutdown 全通过 |
| `python tools/asr/asr_w8_format_test.py <TEMP>/asr_w8_test_p01.exe` | 5 格式 + nan/zero/negative/infinity 四 scale 负例全拒绝 |
| WSL 交叉编译生产 ASR，RVV + W8_RVV + CONV_PACK | 严格编译退出 0，未在 x64 执行产物 |
| WSL 交叉编译 W8 单测 A/B/C/D 四档 | 全部退出 0，仅编译验证，不代表 RVV 位级运行通过 |
| FFN/conv_out 真实资产运行、完整 Python/WAV 门禁、ASan WAV、5 次 profile | 未运行，缺配套资产/参考/捕获输入与可用板端 |

本地编译的公共参数：

```text
-std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off
-D_GNU_SOURCE -Iinclude/core -Iinclude/common -pthread
```

W8/FFN/conv_out 编译形式：`gcc <公共参数> tools/asr/<name>.c src/core/vllm_tp.c -o <TEMP>/<name>_p01.exe -lm`。
线程池单测使用相同形式但不另链接 `src/core/vllm_tp.c`。
生产交叉编译命令（在 WSL、仓库根目录）：

```sh
riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -march=rv64gcv -DVLLM_ASR_W8_RVV -DVLLM_ASR_CONV_PACK src/model/vllm_asr.c src/core/vllm_tp.c -Iinclude/core -Iinclude/common -D_GNU_SOURCE -pthread -o /tmp/asr_p01_rvv -lm
```

W8 四档交叉构建将生产源换为 `tools/asr/asr_w8_test.c`，目标为 `/tmp/asr_w8_p01_rvv`，使用 P1 定义的 A/B/C/D 开关。
一次 PowerShell→WSL shell 循环因引号传递失败；改为 Python subprocess argv 列表后四档全部成功，未修改源码规避告警。

### 同日续办：真实资产生成与本机门禁

本段更新上文“未生成/未运行”的历史状态。现有 ONNX 完整且本机依赖、资源可用，未安装任何依赖。
转换前 E: 可用 109,985,239,040 字节，物理可用内存约 19 GiB；WhisperFeatureExtractor 实际导入成功。
没有改动生产源码或两个门禁的 --cflags，没有提交/推送，也未重新探测板端。

资产根目录 `B = E:/项目/Kestrel_VLLM_RISC-V/modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/`，
验收目录 `R = B/p0_validation/`，均为现有模型资产目录下的持久产物。

| 资产 | 实际位置/内容 |
|---|---|
| F32 VQF | `B/audio8_p0_f32.vqf`，1,302,080,475 字节，480 entries，1189 个完整验证的共享张量 |
| 模型 SHA256 | `5e1de521e7ce6b44697a6b0b79ff16cf3fdc1cc095a7f62938ec9d8033466020` |
| 音频 SHA256 | `a1bd32dc78493c123f9625a66deee562aed2895f53fbc39f2cca3be7e6f4f20f`，输入仍为上文 5.5466875 秒音频 |
| 来源记录 | `R/provenance.json`（绝对路径、字节数、SHA256） |
| map/冻结输入/完整参考 | `R/hidden/{asr_lm.map,mel.bin,input.bin,output.bin,alignment.json}` |
| WAV 资产与报告 | `R/wav/{asr_lm.map,asr_wav.assets,wav_alignment.json,wav_trace.bin,transcript.txt,compile.log,binary_imports.log}` |
| 显式标量参考 | `R/scalar/{alignment.json,mel.bin,input.bin,output.bin,hidden_run.log,mel_run.log}` |
| 实际层输入 | `R/{fc1_input.bin,fc2_input.bin,conv_flat.bin,capture.json}`；从同一冻结 Mel 的 NumPy encoder 捕获第一真实 token，分别 896/3584/7680 FP32，非随机输入 |

执行转换：在仓库根目录运行
`python -B tools/asr/onnx_to_vqf.py --bundle ../modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle --out ../modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/audio8_p0_f32.vqf`。
退出 0，三图导出、两张 LM head 与 embedding 全矩阵共享校验、完整 FNV 写入均完成。
转换原始日志：`C:/Users/Administrator/AppData/Roaming/Trae/ModularData/ai-agent/command-execution/467d6fddd786191b44004006cf5845d9/output.log`。

首次双门禁因为 MinGW 链接器中文输出路径编码失败，原日志保留 `R/{hidden_gate.log,wav_gate.log,execution.json}`。
用可逆 `subst Q: E:/项目/Kestrel_VLLM_RISC-V` 映射解决；Q: 目前保留供复现，不复制模型。
重跑默认门禁时 `VLLM_THREADS=1`、`OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4`，使用默认 32 token 上限且实际遇 EOS 停止。
`R/execution_ascii.json` 记录完整命令和子进程退出码；stdout/stderr 保存 `R/{hidden_gate_ascii.log,wav_gate_ascii.log}`。
临时执行与捕获脚本仅在 `%TEMP%/asr_p0_{continue,layers,sanitize}.py`，Python 使用 `-B`/禁用 bytecode。

| 验证 | 实测结果及日志 |
|---|---|
| 完整 `asr_c_gate.py` 默认参数 | 退出 0；mask、token、所有负例通过；hidden max_abs=5.14984130859375e-05，projector=4.1961669921875e-05，逐步 logits 全部 ≤1e-3；`R/hidden/alignment.json` |
| 完整 `asr_c_wav_gate.py`（未传 --quick） | 退出 0；真实音频 + 9 边界 + extra chunk、7 拒绝例、287 decoder 用例通过；冻结 Mel 字节回归与 hidden 回归通过，Python 误差上界 8.58306884765625e-05；`R/wav/wav_alignment.json` |
| 文本/token | 两门禁文本一致：欢迎大家来体验达摩院推出的语音识别模型。12 个选择 token（含 EOS 151645），完整列表在 JSON |
| 显式 `--cflags=-DVLLM_ASR_SCALAR` hidden/Mel 门禁 | 退出 0，`R/scalar/alignment.json`；原始 stdout/stderr 在 Trae `command-execution/ea19e992bcfb512ea8cef1da086c257a/output.log`；仅证明 x64 逃生舱，尚非板端 RVV 关闭对照 |
| FFN / conv_out 真实输入 | 严格编译与运行退出 0；`R/asr_ffn_test.log`、`R/asr_conv_out_test.log`，生产核 memcmp 位级通过；输入来源见 capture.json |
| W8 位级 / 线程池 / W8 格式 | 全部退出 0；`R/asr_w8_test.log`、`R/asr_tp_test.log`、`R/w8_format.log`、`R/layers_execution.json`；x64 回退路径，不代表 RVV 板测 |
| WSL GCC 15.2 ASan+UBSan | `-fsanitize=address,undefined -fno-omit-frame-pointer` 严格构建；真实 WAV、9 边界、extra chunk 均退出 0，5 malformed WAV 预期退出 1且报正确输入错误；真实 WAV 完整推理退出 0，ASan/UBSan/LSan 零报告；`R/sanitize.log`、`R/sanitize_execution.json` |

WSL 消毒器二进制及临时输出在 `/tmp/asr_p0_sanitize*`，Windows 单测二进制位于 R。
这些运行用于本机正确性准备；单测输出中的计时不作为 P0/P1 性能证据。

### 下一步与实际阻塞

1. F32 资产、冻结 Mel、Python 参考及真实层输入已经就绪，不再列为缺失；同源 W8 VQF 和量化精度验收仍待单独准备，本轮没有把 int8 ONNX 或 F32 输入冒充 W8 验收。
2. 板端仍是此前 localhost:2222 未建立连接的阻塞，本轮按约束未重试也未探测其他地址。恢复后核实 ISA/VLEN/编译器/频率/温度，再传递上述带哈希资产，运行 A/B/C/D RVV 位级与完整门禁。
3. 本机默认双门禁、显式标量 hidden 门禁和 WSL 内存检查已完成；目标板显式标量 WAV/优化档及线程矩阵仍需执行。固定 trace/线程/解码长度后，每档重复 5 次采集 19 叶子桶，才填写 P0 基线与 P1 收益。
4. 无源模型、转换依赖或磁盘资源阻塞；中文输出路径问题已绕过且保留失败证据。当前仍未达到阶段 Definition of Done，所有性能复选框继续保持未完成。

### 同日再续办：同源 W8 与真实 SSH 握手

本段替代上文“W8 尚未准备”和仅 TCP 探测的当前状态，历史记录保留。未修改生产内核，未提交或推送。
`B` 仍为上述 model_bundle，新增验收目录 `W = B/p1_w8_validation/`，继续使用已有 Q: 映射。
临时执行脚本及转换器仅在 `%TEMP%/asr_p01_*`，没有覆盖仓库已有转换器二进制。

执行前检查三个环境变量 `VLLM_WMODE`、`VLLM_VQF_KEY`、`VLLM_VQF_SIGN_PRIV` 均为空，只输出布尔状态。
预注册 `W/preregister.json`：当前真实音频的 W8/F32 token（含 EOS）和归一化文本完全一致作为有限样本量化门槛；
数值误差仅记录，不将 1e-3 用作跨精度量化质量阈值，不外推到其他音频。

已有 `vqf_convert/vqf_conv.exe --help` 未包含 ASR W8 入口，故按当前 build.bat 源文件列表构建临时转换器：

```powershell
# 在 vqf_convert 目录执行，退出 0；保留已有通用转换源码 unused/format 告警，未修改源码掩盖。
gcc -O2 -Wall -Wextra -I src src/conv_main.c src/model_cfgio.c src/model_layers.c src/quant_kernels.c src/qk_repack.c src/vqf_vision.c src/vllm_crypto.c src/vqf_asr.c -o "$env:TEMP/asr_p01_converter.exe" -lm -lbcrypt -lshell32
# B 展开为上述 bundle；完整 argv 和退出码见 W/execution.json
& "$env:TEMP/asr_p01_converter.exe" --model $B --convert-vqf "$B/audio8_p1_w8.vqf" --asr-w8
```

转换退出 0，日志 `W/convert.log`：480 entries、1189 个完整共享张量验证，FNV `086bdfafcd1daa66`。
构建原始日志：`C:/Users/Administrator/AppData/Roaming/Trae/ModularData/ai-agent/command-execution/5b8cafa6936c5abf0052ebcdb6fa1a67/output.log`。

| 模型 | 字节数 | SHA256 |
|---|---:|---|
| `B/audio8_p1_w8.vqf` | 332382327 | `c5f7e4db8db7e5637fc15cfdbd6044b78fd38ad609e1fe23c7f29dcfbe2ac973` |
| 原 `B/audio8_p0_f32.vqf` | 1302080475 | `5e1de521e7ce6b44697a6b0b79ff16cf3fdc1cc095a7f62938ec9d8033466020`（重新核验，未变） |

音频路径、字节数及 SHA256 再次记录于 `W/provenance.json`，音频哈希仍为 `a1bd32dc78493c123f9625a66deee562aed2895f53fbc39f2cca3be7e6f4f20f`。
现有 RowW8 参考直接支持该模型；门禁自行重新生成 `W/hidden/asr_lm.map` 和 `W/wav/{asr_lm.map,asr_wav.assets}`，未复用 F32 的 map/assets。
以下在仓库根执行，`AUDIO` 为前述真实 WAV，`VLLM_THREADS=1`、`OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4`：

```powershell
python -B tools/asr/asr_c_gate.py --vqf "$B/audio8_p1_w8.vqf" --audio $AUDIO --out "$W/hidden"
python -B tools/asr/asr_c_wav_gate.py --vqf "$B/audio8_p1_w8.vqf" --audio $AUDIO --out "$W/wav" --reference "$W/hidden"
```

| 验证 | 真实结果与证据 |
|---|---|
| 同 W8 Python/C hidden + Mel 完整门禁 | 退出 0；mask、token、负例全通过；hidden max_abs `3.9577484130859375e-05`，projector `2.193450927734375e-05`，逐步 logits 最大 `4.00543212890625e-05`；`W/hidden/alignment.json`、`W/hidden_gate.log` |
| 同 W8 完整 WAV 门禁（未传 --quick） | 退出 0；真实音频、9 边界及 extra chunk、7 拒绝例、decoder 检验通过；Python 误差保守上界 `7.915496826171875e-05`；冻结 Mel 字节回归与 hidden 回归通过；`W/wav/wav_alignment.json`、`W/wav_gate.log` |
| x64 显式标量逃生舱 | 严格参数加 `-DVLLM_ASR_SCALAR` 编译退出 0，同冻结 Mel 完整输出与默认 C 字节相等；`W/scalar_comparison.json`、`W/scalar_compile.log`、`W/scalar_mel.log`。这是 x64 回退路径，不能代替 RVV 位级验收 |
| W8 相对 F32 有限样本量化门槛 | 12 个 token（含 EOS）及文本完全相同：欢迎大家来体验达摩院推出的语音识别模型。`W/quantization_comparison.json`，该单样本门槛通过 |
| 跨精度数值差异（描述性） | 同冻结 Mel 的 C traces：hidden max_abs `0.2784614562988281` / RMSE `0.027014253901899883`；projector max_abs `0.26361846923828125` / RMSE `0.02115641907256217`；逐步 logits 最大 `0.8337907791137695`。逐步 logits/KV 的 max_abs、RMSE 全量落盘；不作广泛量化质量结论 |

两个默认门禁的严格编译保留 `-ffp-contract=off`。hidden 二进制 SHA256 为
`5ee95f4c21e30349e70d02749862672ab952b746dcc4c9a94834394bc1139f70`，WAV 二进制为
`94a3fa7f8e03f8fd4e55f75f137a453356d5818404505fefef61ca0b27fcd350`。
显式标量真实 WAV 亦退出 0，trace 和文本与默认 C 字节相等，见 `W/scalar_wav_comparison.json`、`W/scalar_wav.log`；
map、assets、冻结输入、trace 及标量二进制的字节数和 SHA256 汇总于 `W/asset_sha256.json`。
门禁 JSON 的历史描述字符串仍写 `C F32 audio encoder`，本次实际输入以命令、模型哈希及 W8 目录为准；没有因此误判不支持 W8。

板端按 `../记录/RISC-V线上设备连接.md` 使用 IPv4 和真正 SSH 握手：

```powershell
ssh -4 -p 2222 -o BatchMode=yes -o ConnectTimeout=7 -o ConnectionAttempts=1 root@127.0.0.1 "uname -a"
```

退出 255：`banner exchange: Connection to UNKNOWN port -1: Connection refused`。这是实际 SSH 失败，
不是裸 socket 等 banner 超时，也未到认证阶段。命令及错误保存于 `W/preregister.json`；原始日志在
`C:/Users/Administrator/AppData/Roaming/Trae/ModularData/ai-agent/command-execution/4ee577ca222536ed958fb50ec1979fbf/output.log`。
未改隧道、重启服务或访问其他设备；未上传或占用板端。ISA/VLEN、空间、负载、频率和热状态尚不能核实。

剩余验收：板端恢复后，先核实身份与资源，再做同模型标量/RVV 位级及线程矩阵；正确性通过后，
按 F32/W8 实际适用档分别进行同会话每档 5 次 profile，记录全部 19 叶子桶、解码长度和波动。
同模型同代码路径的冗余宏组合不重复充当独立性能证据，F32 与 W8 不混算开关收益。
本轮没有板端 profile、RVV 运行或 1.3× 收益证据，P0/P1 仍未通过完整验收；P2–P7 未开始。

### 同日新授权板端实测（进行中）

用户提供 `../远程板端.txt`，现已使用该文件中的授权地址及账户连接，旧 localhost 阻塞不再适用。
凭据只由工作区外 Python/Paramiko 临时脚本读取，没有写入命令行、日志或本文。
连接匹配已有 known_hosts，主机公钥指纹 `SHA256:CXeeK3vQu0jT3jOP+h/wnt51CS2zikIcwNear7x70QU`。
独立板端目录 `/mnt/ssd/kestrel/asr_validation_20261117_2318`；没有覆盖已有项目或终止其他进程。
证据目录仍为 `W = B/p1_w8_validation/`。

- 真实架构 `riscv64`，Spacemit X60，8 核，ISA 含 V；原生 GCC 13.2.0、double ABI（lp64d）。
- RVV intrinsic 实测 `VLEN=256 bits`。默认编译器宏没有 `__riscv_vector`；优化档显式 `-march=rv64gcv -mabi=lp64d`。
- 首次检查内存可用 7,382,990,848 字节、SSD 可用 30,513,844,224 字节、无 swap；performance、1.6 GHz、36/38°C。
- 负载均值 2.32/2.24/2.23，进程快照没有其他高 CPU 推理任务；传输与编译期间不采集正式性能。
- `W/remote_probe_20261117.json`、`remote_resources.log` 保存环境；`remote_source_verified.json` 保存逐文件远端 SHA256 校验；`remote_binary_identity.log` 保存 VLEN 与全部二进制 SHA256。

| 已执行板端门禁 | 结果 |
|---|---|
| W8 A/B/C/D 严格原生编译 | 四档退出 0，全部保留 `-ffp-contract=off`，无 fast-math |
| W8 10 形状 × transposed/wide/bias 全组合 | 四档在实际 1 与 8 线程均位级通过；`remote_w8_run_*.log`、`remote_w8_t8_*.log` |
| 格式及 scale 负例 | 每档 5 格式 + 4 scale 全拒绝；`remote_w8_format_absolute_*.log` |
| 线程池创建失败降级 | actual=1..8，static/dynamic/nested/shutdown 全通过；`remote_tp.log` |
| 默认及 A/B/C/D 生产编译 | 五档严格编译退出 0；`remote_asr_compile_*.log` |

完整编译/执行 argv 见 `W/remote_execution_20261117.json`、`remote_models_execution.json` 与 `remote_extra_execution.json`。
板端复现形式（在独立目录执行）：

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -D_GNU_SOURCE -Iinclude/core -Iinclude/common -pthread -march=rv64gcv -mabi=lp64d -DVLLM_ASR_W8_RVV -DVLLM_ASR_CONV_PACK tools/asr/asr_w8_test.c src/core/vllm_tp.c -o w8_D -lm
VLLM_THREADS=1 ./w8_D
VLLM_THREADS=8 ./w8_D
python3 tools/asr/asr_w8_format_test.py /mnt/ssd/kestrel/asr_validation_20261117_2318/w8_D
```

首次格式负例因 Path 将 `./w8_A` 化为 `w8_A`、未从当前目录搜索可执行文件而失败；改为绝对路径后四档全部通过，原失败日志保留。
一次模型启动早于资产传输完成，退出 1，属于编排先后问题；不得计为模型正确性结果。
完整 Python 双门禁依赖在板端仅 NumPy 可用，缺 torch/transformers/tokenizers/scipy；冻结资产 C 对拍可独立执行，但不能冒充完整 Python/WAV 门禁。
17 项模型/输入/参考资产已传输完成且 SHA256 校验一致，清单见 `W/remote_asset_manifest.json`。冻结参考对拍仍在执行，正式五次 profile 尚未开始；当前没有可报告的性能中位数或 1.3× 收益，P0/P1 尚未通过，P2–P7 未开始。

首个板端 F32-A、实际 1 线程冻结 Mel 用例退出 0。与既有主机 C 参考相比，hidden 最大绝对误差 `3.0994415283203125e-05`、projector `5.14984130859375e-05`、logits 最大 `3.0934810638427734e-05`、KV 最大 `1.5974044799804688e-05`；mask 及 12 个选中 token 均一致。结合原 Python/C alignment 的三角误差上界为 `9.34600830078125e-05 < 1e-3`，仅证明该冻结用例的数值检查通过，不代替完整 Python/WAV 双门禁。证据为 `W/board_first_f32_scalar_comparison.json`、`board_f32_A_t1_mel.bin` 和 `board_f32_A_t1_mel.log`。

该正确性运行的单次 total 为 `432.855102875 s`，decode_steps=11、generated_tokens=11、selected_tokens=12；19 个叶子桶求和相对误差为 0（排除 encoder aggregate）。这是单次观察，不能用于五次中位数或优化收益结论。对应 WAV 用例也已退出 0，单次 total `425.937652483 s`；相对同一冻结参考的 hidden/projector 最大绝对误差分别为 `3.147125244140625e-05`/`1.71661376953125e-05`，Python 误差上界 `8.296966552734375e-05`，mask、12 个选中 token 和 transcript 一致。见 `W/board_comparisons.json`。A 档记录中的 scalar_equal=true 是自身基准比较，不能视作 RVV 已通过。Mel SHA256 为 `5d8652af393ff6b684df83f5fa201f78ed75a55fe381f90795ec240c58609423`，WAV trace SHA256 为 `e78d9c055440317fc5e4c145231581cfb45dcea9806a6b2329ed8b69a749bab4`。后续 default 档正在执行。复现首个 Mel 用例：

```sh
cd /mnt/ssd/kestrel/asr_validation_20261117_2318
VLLM_THREADS=1 VLLM_ASR_PROFILE=1 ./asr_A assets/f32/model.vqf assets/f32/asr_lm.map assets/f32/mel.bin f32_A_t1_mel.bin
```

续跑实测：未重启原模型编排。板端为 8 核 SpacemiT X60 / riscv64，使用 `csrr vlenb` 实读 32 字节，即 **VLEN=256 bits**；原生探针源码、命令和退出 0 记录见 `W/board_vlen_probe.json`。

F32 default、实际 1 线程的 Mel/WAV 均退出 0，与 A 档 trace 字节相同；default 不能当作 RVV。真正同 ISA 的 A/B 均以 `-march=rv64gcv -mabi=lp64d -ffp-contract=off` 编译，B 的冻结 Mel 已退出 0：完整 8123976 字节与 A 相同，hidden/projector/12 步 logits/KV/token 最大差全为 0，SHA256 `5d8652af393ff6b684df83f5fa201f78ed75a55fe381f90795ec240c58609423`。证据 `W/board_f32_AB_exact.json`；随后 B WAV 也完成完整 trace 对拍，8123976 字节与 A 全相等，各字段最大差为 0，SHA256 `e78d9c055440317fc5e4c145231581cfb45dcea9806a6b2329ed8b69a749bab4`。因此该真实模型、该音频、实际 1 线程的 A/B Mel+WAV 位级检查通过，不能外推其余档位及 8 线程矩阵。

提前执行的层级和负例流程已完成：2 项严格编译、F32/W8 各 FFN 和 conv_out 共 4 项层级检查、两模型 × A/B/C/D × 9 类负例共 72 项拒绝检查全部通过，见 `W/board_layers_negatives.json`。FFN 真实输入每方向仅 1 token、各重复 3 次；F32 conv_out 为 11 个形状，W8 仅真实 7680→896 形状。W8 其余尾形状覆盖仍以独立 W8 内核测试为准。

验收脚本发现并修正了两处 W8 不适配（仅远端验证目录副本，未改生产内核）：FFN 的 `affn_pack` 对 W8 返回原句柄，测试每轮 `free(packed)` 导致 double free，应使用 `affn_free`；conv_out 原参考把量化句柄当 FP32 数组，应通过 `qvalue` 按逻辑行索引读取，并避免套用不适用的合成输出宽度。原失败、修正后源码和执行证据为 `board_w8_ffn_initial_failure.json`、`board_w8_convout_early.json`、`board_w8_convout_validated.json`、`board_asr_*_validated.c`。修正后的 W8 层级流程再次完整通过，没有据此认定生产 RVV 数值失败。FFN 计时未开启时输出 0/NaN，只算正确性，不算性能。

续跑性能数据汇总为 `W/board_continuation_results.json`：只计 19 个叶子桶，排除 encoder aggregate。已有正确性用例的计时均为单次观察，部分与层级检查并行，**正式 profile 样本数仍为 0**，五次中位数、波动、1.3× 收益与完整线程矩阵尚未通过验收。原 `run_models.py` 将继续其余正确性矩阵，任何字节不一致自动禁止后续性能运行。

2026-11-18 00:19:57（Asia/Shanghai）再次只读核查：原 PID 57789 自 11-17 23:30:41 启动，已运行约 49 分 15 秒，未重启。完成 16/36 次正确性推理、8/18 组 Mel/WAV 对拍，均退出 0；F32 A/default/B/D 在实际 1/8 线程的 Mel/WAV 全部通过，16 份完成日志的线程数、解码计数及 19 桶闭合检查全部通过。当前为 `w8_A_t1_mel`（PID 85346，已运行约 1 分 13 秒，CPU 100%），剩余 20 次正确性推理（含当前运行），之后还有层级负例流程和 90 次正式 profile。F32 D、实际 1 线程 Mel/WAV 已与 A 的完整 trace SHA256 一致，并通过冻结主机参考误差和 transcript 检查；D 的 total 单次观察为 `80.866728974 / 81.605902993 s`。F32 A、实际 8 线程的 Mel/WAV 也已通过，日志明确 `requested=8 actual=8`；其 Mel/WAV SHA256 均与单线程 A 一致。证据为 `W/board_comparisons.json`、`board_model_execution.json`、`board_f32_D_t1_*.log`、`board_f32_A_t8_*.log`。

已读取远端真实编排并保存为 `W/board_run_models.py`：F32 使用 A/default/B/D 四档，W8 使用 A/default/B/C/D 五档，各覆盖 1/8 线程、Mel/WAV 两种输入，共 `(4+5)×2×2=36` 次推理。随后自动再次执行 `run_layers_negatives.py`，成功后按模型、线程、repeat、档位依次跑 WAV profile，即 `(4+5)×2×5=90` 个正式样本。无须另起 resume 或模型脚本。该矩阵不覆盖 P7 的 2/4/16 线程要求，不能据此宣告完整 P7 通过；提前完成的层级负例仍会被原编排复跑，未擅改运行中任务。

profile 的环境及计时分别写入板端 `/mnt/ssd/kestrel/asr_validation_20261117_2318/profile_*_env.json`、`profile_*.log`，执行记录和状态写入 `model_execution.json` / `model_status.json`；本地收集到 `W/board_*`。profile 共用 `profile_trace.bin` 和 `profile_text.txt`，只保留最后一次输出，不能当成所有样本独立 trace 证据。最终后处理仍须逐样本核验实际线程数、退出码、解码长度、19 叶子桶闭合及环境，再按 18 组各自五次计算 median/min/max 和同会话 A/B/C/D 比值。已补齐本地临时解析脚本 `asr_continue_parse.py` 的 18 组五次完整性、退出码、实际线程、11/11/12 解码计数、环境记录存在性和 19 叶子桶闭合检查；仅五次齐全且通过才输出 median/min/max、波动及相对 A 比值。当前正式样本仍为 0，`five_sample_matrix_passed=false`，`performance_passed=false`；环境频率与温度的可比性仍需最终审阅，不因脚本完成而自动认定性能通过。

2026-11-18 01:23:51（Asia/Shanghai）持续低频采集确认：原 PID 57789 未重启，已完成 **36/36 次正确性推理、18/18 组完整对拍、0/90 次正式 profile**；相较此前 01:06:07 的 24/36，本次完成剩余 W8 D 单线程及 A/default/B/C/D 八线程的全部 Mel/WAV 十二次推理、六组对拍。36 份完成日志均退出 0，requested/actual 线程、decode/generated/selected=11/11/12、19 个指定叶桶闭合检查均通过，`correctness_matrix_passed=true`，无执行或样本检查失败。此处完成的是冻结输入 C 对拍矩阵，不是完整板端 Python/WAV 双门禁。

**新增 W8 RVV 生产路径证据：C 与 A、实际 1 线程的冻结 Mel 和 WAV 完整 trace 位级相等。** 01:05:40 独立下载 A/C 四份 trace 到本地复核，两个模式各 8123976 字节；整文件字节、390 项 mask、hidden、projector、12 步 logits/KV/token 全部相等，数值字段最大差均为 0，解析恰好覆盖完整文件。证据为 `W/board_w8_AC_exact.json` 与 `board_w8_{A,C}_t1_{mel,wav}.bin`，不是 A 自身比较或单次计时推断。Mel/WAV SHA256 分别为 `bd308c2d8bc4cf8087d44a6c839c4559bceb039b4521d7ec10179871ba06c72d` / `f90364bf207cf16891964f3bc60ae9773930cf20e7609130ab73490702979d83`。C 的冻结主机参考检查及 WAV transcript 检查通过，Python 三角误差上界分别为 `9.679794311523438e-05` / `8.726119995117188e-05`，均小于 1e-3；此结论仅限该 W8 模型、该音频和实际 1 线程，不外推 D 或 8 线程，也不代替完整双门禁。

本次新增独立下载复核：01:14:11 的 W8 A/D 单线程、01:22:26 的 A/C 八线程、01:24:46 的 A/D 八线程，Mel/WAV 整文件均为 8123976 字节且位级相等，390 项 mask、hidden、projector、12 步 logits/KV/token 全字段相等。原始证据为 `W/board_w8_AD_exact.json`、`board_w8_AC_t8_exact.json`、`board_w8_AD_t8_exact.json` 和对应 `board_w8_{A,C,D}_t{1,8}_{mel,wav}.bin`。这将既有单线程 C 证据扩展到本次明确测过的 D 及八线程 C/D，但仍仅限同一冻结音频。

01:28:36 最新快照：原 PID 57789 已运行约 1 小时 57 分 53 秒，状态 running，已自动进入 `run_layers_negatives.py` 复跑（PID 122740），尚未记录该步骤退出；子进程 PID 125788 正在运行 F32 B `negative_mel_truncated_features.bin` 拒绝检查，CPU 100%，证据 `W/board_layers_rerun_health.json`，未见挂起证据。剩余 **层级负例复跑及 90 个正式样本**。本地低频采集器取得全部剩余正确性后已停止，板端原编排保持运行。快照见 `W/board_progress_snapshot.json`，本轮连续采集时间序列见 `W/board_progress_watch_20261118.json`，完整误差和哈希见 `W/board_comparisons.json`，样本及缺失清单见 `W/board_continuation_results.json`。不把旧的 78 项通过记录冒充本次自动复跑完成。本次没有修改板端测量配置、生成性能结论或放宽门禁；后续性能结论或测量变更交父代理合规审计。

本次完善已有本地后处理 `asr_continue_parse.py`：核验具体 19 叶桶集合及非负有限计时、排除 encoder aggregate；正确性整矩阵必须同时具备 36 份合格执行日志和 18 组位级对拍；正式样本要求环境字段非空。五次齐全后按同模型同线程计算 A 与 default/B/C/D 的 total median/min/max、绝对极差及相对极差、每生成 token 的 decode 秒数，并列出叶子 stage 中同时恶化 >5% 且 >10 ms 的回退。新增执行失败、样本失败、缺失正确性和正式样本标签清单，供失败时仅恢复缺失部分，当前没有启动恢复任务。

环境证据仍有明确范围限制：原 runner 每样本采集 loadavg、thermal、scaling_cur_freq，**没有逐样本 governor**；初始 performance 策略快照不能自动证明每个正式样本的策略，`environment_review_passed=null`，仍需最终审阅。2026-11-18 00:32:00.940307+08:00 板端只读采样已落盘 `W/board_environment_observation.json`：两个 thermal zone 均为 42000（42°C），8 个 CPU 的 scaling_cur_freq 均为 1600000 kHz、governor 均为 performance，policy0 结果一致，loadavg 为 3.39/3.59/4.30，无读取错误。因此当前 runner 的 thermal/frequency 路径有效，但本次瞬时证据不能追溯填充过去样本或证明未来 profile 环境。正式样本共用输出文件、仅一段冻结音频及每精度一个模型、不含 P7 的 2/4/16 线程、缺完整板端 Python/WAV 双门禁依赖等限制保留。本轮仅适用本计划真实 I1–I5，不增加额外 M/O 限制；尚未生成通过阶段报告，P0/P1 与性能验收均未标通过。

最小补验准备（以下为此前待办说明；可执行准备见下一节，不与现有任务并行占板）：先让原矩阵和 90 个样本完成，再依据原始逐样本环境和同组五次 total 极差 ≤3% 审阅稳定性；本次解析器增加 `stability_passed`（五次齐全后的极差/中位数 ≤3%）、`plan_closure_passed`（具体 19 桶集合及非负有限计时、闭合 ≤2%），另保留原 `strict_closure_screen_passed`（1e-6）更严筛查，不混淆两者；W8 D 相对 C 的所有 >5% stage 回退单独列入 `stage_regressions_gt5pct_vs_C`，不附加 10 ms 豁免。`layers_negatives_passed` 还须本次 runner 的 `layers_negatives` 退出 0，防止旧通过 JSON 冒充当前复跑成功。确需补测时只重测环境证据不足或波动超标的同模型同线程对比组，每档完整五次并保留同会话 A 基线；任何测量配置变更先交父代理完成所需审计，不修改正在运行的 runner。完整门禁仍须补足可执行 Python 参考依赖后的 `asr_c_gate.py` / `asr_c_wav_gate.py`（含 9 个边界 WAV、binary_imports、格式/图/形状负例及逃生舱覆盖），不能用已有 78 项层级负例或单冻结输入替代；P7 的 2/4/16 线程留到对应阶段。本次未启动补验、未改生产代码，结果 JSON 保持性能未通过。

### 2026-11-18 串行补验工具与本地实跑

新增 `tools/asr/asr_supplement.py`，只编排现有门禁及冻结边界资产，不改生产内核。`prepare` 复用既有 WAV 门禁的 9 个边界、extra chunk、5 个 malformed WAV、2 个损坏 assets、冻结 features 和 Python 误差报告，保存 29 个文件的 SHA256。输入和预期已实际落盘 `W/supplement_fixtures/`；非随机重造、非仅命令建议。`boundary` 逐个串行调用既有二进制，核验完整 features 长度、头部、有限值及 Python 三角误差上界 ≤1e-3；7 个负例要求退出 1 且具体错误文本匹配。

本机实跑 `W/wav/asr.exe`：**17/17 通过**，最终严格版结果和逐例 stdout/stderr 在 `W/supplement_local_strict/{results.json,*.log,*.features}`；首次较宽退出检查的运行另保留 `W/supplement_local/`。这只是 Windows 冻结边界回放验证，不能当作板端运行或完整 Python/WAV 双门禁。

`boundary --runner-root` 与 `full --runner-root` 均在启动计算前要求原目录 `model_status.json` 为 complete、90 个唯一预期正式标签齐全且全部执行记录退出 0。保护逻辑本地验证：running 拒绝、complete+89 拒绝、complete+90 接受。`--local-check` 仅允许 Windows 本地回放。输出目录必须不存在，避免覆盖旧证据。板端补验尚未上传/启动；正式采样期间不执行补验、不安装依赖。

本机已执行（仓库根，W 展开为上述目录）：

```powershell
python -B tools/asr/asr_supplement.py prepare --source "$W/wav" --out "$W/supplement_fixtures"
python -B tools/asr/asr_supplement.py boundary --local-check --exe "$W/wav/asr.exe" --fixtures "$W/supplement_fixtures" --out "$W/supplement_local_strict"
```

正式采样完成后，将工具及 `supplement_fixtures` 传入原验证目录；逐个执行 default/A/B/C/D 的边界回放，不能并行。示例（R 为原板端验证目录；其他档替换 A 及输出名）：

```sh
python3 tools/asr/asr_supplement.py boundary --runner-root "$R" --exe "$R/asr_A" --fixtures "$R/supplement_fixtures" --out "$R/supplement_boundary_A"
```

完整双门禁入口 `full` 先运行原 `asr_c_gate.py`、成功后运行原 `asr_c_wav_gate.py`，不传 quick，使用本次 hidden 输出作为 WAV reference，线程固定 1、关闭 profile，逐门禁日志和 argv/退出码落盘。依赖仍须在采样结束后解决（现缺 torch/transformers/tokenizers/scipy）；入口本地仅验证保护逻辑，不声称已执行完整门禁。F32 用 A/default/B/D，W8 用 A/default/B/C/D，各输出独立目录；A 的 SCALAR 是逃生舱覆盖，B/C/D 参数按 P1。示例：

```sh
python3 tools/asr/asr_supplement.py full --runner-root "$R" --vqf "$R/assets/f32/model.vqf" --audio "$R/assets/audio.wav" --out "$R/supplement_full_f32_A" --cflags="-march=rv64gcv -mabi=lp64d -DVLLM_ASR_SCALAR"
```

边界包来自 W8 WAV 资产；F32/W8 完整门禁仍各自导出 map/assets，不用该包替代模型专属完整门禁。旧 governor 和共享 trace 限制不变，不补造历史记录；所有性能结论继续等待五次及环境审阅。
