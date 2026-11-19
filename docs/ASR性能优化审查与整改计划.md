# ASR 性能优化审查与整改计划

## 阶段性收口确认（2026-11-18）

用户已明确确认“收口”。本阶段工作关闭，不再自动启动复测或扩大优化范围；本次仅确认交付状态，不修改原始证据、冻结门槛或历史判定。总体计划不标记为全部完成，性能验收不标记为通过。

- 已交付：资产复用与功能门禁、环境记录和新采样协议工具、限定复测执行记录及失败证据封存。本轮采样前正确性 8/8 通过。
- 实测观察：F32 单线程 A/D 各完成五次，total 中位数为 440.502958179s / 81.833737735s，加速比 5.382901605。环境证据存在缺口，该收益不等同于正式性能验收通过。
- 未通过项：限定矩阵仅完成 10/40 正式样本；第三次预热达到 75°C，触发冻结的严格温度门槛，矩阵失败。环境覆盖与亲和证据缺口保留；限定范围整体验收为 false。
- 证据保留：Audio8 模型目录内 `model_bundle/p1_w8_validation/final_20261118_run3`，包括原始记录、汇总、169 项 SHA256 清单及独立补充核验 `followup_verification.json`（不在原清单内）。终态复核时板端执行进程及本地收集器均已退出。
- 后续事项：散热改善、环境与亲和监测补齐、完整矩阵复测、多模型、2/4/16 线程及 LM 优化递延，需后续重新启动工作。本次收口不代表这些事项已完成。

以下为执行与审查历史；历史中的推进安排、运行中状态和待办不构成继续执行指令，最终交付口径以本节及 run3 终态复核为准。

## 整改执行更新（2026-11-18）

当前授权范围冻结为 Audio8 同一音频、F32 A/D 与 W8 A/C、1/8 线程共 8 组，每组 1 次预热及 5 次正式，共 48 次执行；间隔至少 3 秒，同 boot，不清缓存。新采样稳定性采用极差/中位数单线程 ≤3%、八线程 ≤5%。收益不回退目标在正式采样前登记为 W8 C/A ≥1.3、F32 D/A 单线程 ≥5.0、八线程 ≥1.8；未解释的阶段回退阻止性能通过。R8 多模型、2/4/16 线程和 LM 优化递延，不属于本阶段收口范围。旧采样仍按原 3% 阈值解释，不回算通过，也不覆盖历史证据。

### run3 终态复核（2026-11-18 15:29:01，Asia/Shanghai；失败封存，不作性能验收收口）

只读复核确认板端 `status=failed`、`matrix_exit_code=1`，原控制 PID521692 及 run3 完整命令匹配进程均不存在；本地原收集器 PID15980 已退出，原日志最后记录为成功封存。未重启矩阵、未改 C 或阈值、未补样、未提交。

正式执行完成 10/40（F32 A/D t1 各5次），预热执行完成 3/8，其中只有2次协议通过；最后 F32 A t8 warmup 虽原始 exit=0、线程横幅8/8且 token/text/trace 校验通过，但 `passed=false`。其周期环境记录在 `time_ns=1794985046891051743` 观测 `/sys/class/thermal/thermal_zone1/temp=75000`，触发冻结策略的严格 `<75000` 判定，矩阵因此失败。全部13条已执行记录原始 exit 均为0，不能据此掩盖环境失败。

原 `sha256_manifest.json` 的169项本地文件全部匹配；其中所有 `board/` 文件与当前远端 SHA256 逐项一致。重放结果与 `summary_recomputed.json` 完全一致；其与板端 `results.json` 的35项差异仅为缺失样本错误中的 Windows/Linux 路径字符串，原 `audit.summary_matches_board=false` 保留，不改写原证据。已执行样本的请求/实际线程、12个 token、文本及 trace 有限值/长度/token 校验均通过，trace SHA256 均为 `e78d9c055440317fc5e4c145231581cfb45dcea9806a6b2329ed8b69a749bab4`。

F32 A t1 预热周期环境最大间隔2.143942970s，超过2s覆盖判据；F32 D t1 r3 实际绑核证据为 unknown。故两组环境结论均为 null。单线程 A/D total 中位数分别440.502958179s/81.833737735s、加速比5.382901605，极差/中位数分别0.997459%/0.452282%，只作为已完成部分的观察值；1/8线程稳定性3%/5%、F32收益5/1.8、W8收益1.3等冻结门槛均未调整，限定矩阵整体验收仍为 false。

补充独立复算现有三个样本 A t1 r1、D t1 r3、A t8 warmup 的19叶桶，总和分别440.033599167s、81.913035748s、103.576116158s，均与 total 相等、相对误差0。此为失败调查补充抽样；原指定 D t8 r3、W8 C t8 r5 未产生，原三样本验收失败记录保持不变。详细复核新增于同一归档的 `followup_verification.json`；原清单不覆盖此新增文件，不覆盖或重签原封存。不同模型家族、2/4/16线程及LM优化继续明确递延。

### run3 最终证据核验（自动封存，2026-11-18）

独立证据目录：`model_bundle/p1_w8_validation/final_20261118_run3`（Audio8 模型目录内）。原始板端目录完整下载至 `board/`；`audit.json`、`summary_recomputed.json`（若矩阵已启动）、`independent_three_sample_closure.json`（若矩阵已启动）及 `sha256_manifest.json` 保存复算和文件哈希。历史结果保留。

本轮实际状态：`failed`；正确性记录 8/8，正式 10/40，预热 3/8。严格 FP 构建身份核验：`true`；正确性：`true`；限定范围验收：`false`。false 为未通过，null 为证据不足，均不得写作性能通过。

| 组 | total中位数(s) | 极差/中位数 | A基线加速比 | 稳定性 | 环境 | 回退检查 | 收益 |
|---|---:|---:|---:|---|---|---|---|
| f32__audio1__A__t1 | 440.502958179 | 0.009974591732967624 | 1.0 | true | null | true | true |
| f32__audio1__D__t1 | 81.833737735 | 0.0045228235351848466 | 5.382901604781502 | true | null | true | true |
| f32__audio1__A__t8 | null | null | null | null | null | null | null |
| f32__audio1__D__t8 | null | null | null | null | null | null | null |
| w8__audio1__A__t1 | null | null | null | null | null | null | null |
| w8__audio1__C__t1 | null | null | null | null | null | null | null |
| w8__audio1__A__t8 | null | null | null | null | null | null | null |
| w8__audio1__C__t8 | null | null | null | null | null | null | null |

逐组错误、环境缺口与未解释回退见 `summary_recomputed.json`，失败不补样、不择优。Windows 复算仅对深拷贝中的板端绝对路径做语法适配，不修改冻结配置；数值正确性上界来自本轮已保存参考对拍，未重新运行模型。

本阶段只涵盖同一 Audio8 音频、F32 A/D 与 W8 A/C、1/8 线程；不同模型家族、2/4/16 线程、LM 优化明确递延。完成采样与封存不等于性能通过，以下执行中说明保留为历史记录。

### 限定复测执行中（历史进度记录；最终状态以上节为准）

2026-11-18 14:04:15（Asia/Shanghai）只读续接核实：run3 已进入 `matrix_running`。完整命令确认控制 PID 521692 仍为 `/mnt/ssd/kestrel/asr_final_20261118_run3/job.py`；子进程 PID 536836 为本轮 `asr_A`，正在执行 `matrix/f32__audio1__A__t1/warmup`。本地 PID 15980 完整命令仍为既有 `asr_final_collect_run3.py`，收集器存活。相比 13:48:55，正确性记录由 3/8 增至 8/8，`correctness_complete.json` 已保存八个组合且标记 passed；逐份核对八份 execution/correctness 记录，退出码均为 0，线程标记匹配各自 1/8 配置，token 一致，记录中的数值误差上界 F32 为 0.00008296966552734375、W8 为 0.00008726119995117188，均小于 0.001。这里是已保存正确性记录检查，尚非终态原始 trace、二进制及哈希的独立复算。A/C/D 构建记录均退出 0，历史与重建 SHA 各自一致。矩阵已有 config/start 和首个预热目录，但尚无已完成 execution.json，已完成正式 0/40、预热 0/8（1 次预热运行中）；job.log 为空，本次只读命令无 stderr，未发现已记录错误。本地预定 `final_20261118_run3` 归档仍不存在，故 audit、summary_recomputed、三份正式样本的 19 叶桶闭合及最终样本完整性、线程/token/环境/收益独立验收尚未完成，不能宣称性能通过。本次未重启矩阵、未改 C、未放宽门限、未提交；原收集器继续跟踪，历史记录保留。多模型、2/4/16 线程和 LM 优化继续明确递延。以下为历史观察。

2026-11-18 13:48:55（Asia/Shanghai）只读续接核实：已按完整命令确认 PID 521692 为本轮目录的 `python3 -B .../job.py`，子进程 PID 528748 为本轮 `asr_D` 正在执行 `correctness_f32_D_t8`；本地 PID 15980 的完整命令确认为既有 `asr_final_collect_run3.py` 收集器。状态为 correctness、completed=3；已保存 `f32_A_t1`、`f32_D_t1`、`f32_A_t8` 三份正确性记录，退出码均为 0、线程标记分别匹配 1/1/8，记录中的数值误差上界均为 0.00008296966552734375，小于 0.001。三份记录的 token、trace SHA 与 text SHA 相同，但此处仅核对已保存记录，尚未进行终态原始文件独立复算。A/C/D 构建记录退出码均为 0，历史与重建 SHA 分别一致；终态下载后仍需对实际二进制复核。板端尚无 matrix 或 correctness_complete.json，本地预定 final_20261118_run3 归档尚不存在：正式 0/40、预热 0/8，最终稳定性、收益、环境、三样本闭合和验收结论均待定，不应将正确性耗时当作正式性能数据。未重复启动或终止任务，未修改 C 或原样本；既有收集器继续运行。下述 13:42 记录保留为历史，不据其耗时估算承诺结束时间。

2026-11-18 13:42（Asia/Shanghai）续接核实：原 PID 521692 仍在运行，状态已从 build_identity D 进入首个正确性组合 `f32_A_t1`，子进程 PID 522785，尚无正式矩阵样本，job.log 未见异常。没有重复启动、编译、上传或终止板端任务。本地临时脚本 `C:/Users/Administrator/AppData/Local/Temp/asr_final_collect_run3.py` 每 60 秒只读监测；仅在终态且控制进程退出后下载至模型证据根的独立 `final_20261118_run3/board`，随后复算同版协议 summary、核验数量/正确性、独立复算三个正式样本叶桶闭合并生成 SHA 清单。此为后续自动处理安排，不代表下载或最终验收已经完成；运行时间按历史样本估算约 3 小时，最终以实际终态和封存产物为准。

当前板端目录 `/mnt/ssd/kestrel/asr_final_20261118_run3` 的脱离会话控制进程 PID 为 521692。run2（PID 515903）已在首个正确性样本的参考误差检查阶段因编排脚本未向提取函数注入 `json` 而失败退出，尚未开始正式矩阵；其日志与产物原样保留。修正命名空间后在 run3 从头重建、对拍与采样，不复用失败轮样本。run2 三个严格 FP 重建二进制 SHA 均与历史对应档一致。前置任务串行复现历史 A/C/D 严格 FP 编译参数（`-ffp-contract=off`，无 fast-math），保存历史/重建二进制 SHA 与历史源码哈希，随后为本轮 8 个组合补做独立冻结参考数值对拍；全部通过后自动执行冻结的 48 次矩阵。`status.json`、`job.log`、`build_identity.json`、`independent_expectations.json`、`frozen.json` 和后续 `matrix/*/execution.json` 是进度依据；不得仅因进程已启动宣称验收完成。

run1 在编译和采样前因旧拓扑识别逻辑拒绝重复 core_id 而失败，原目录 `/mnt/ssd/kestrel/asr_final_20261118_run1` 保留。板端两个集群的 core_id 均为 0–3，但 thread_siblings_list 为独立的 0–7；修复后按 sibling 集合不相交确认独立物理核，同时保存原 package/core_id。此失败没有正式样本，不进行择优补样。

预检确认 8 核、约 25GB 可用磁盘、Python/taskset/gcc 可用，历史 runner 和 R4 run2 均 complete；一次系统更新检查 CPU 占用已在启动前复查消失。两个 cluster thermal zone 最低 passive trip point 均为 75000 毫摄氏度，当前约 39000；温度阈值由真实 sysfs 导出，不能自行编造。协议执行期间约每秒记录 boot、温区/trip points、频率和 governor，每 100ms 观察所有可见引擎 TID 掩码，按 executable 区分 taskset 启动阶段；缺测和覆盖缺口保留 unknown。完整性、稳定性、环境、回退和收益使用失败优先的三态聚合，原始 speedup 独立保留。14 项模拟测试通过。该记录是执行进展，最终样本数量、封存、独立三样本闭合复算及性能判定仍须以实际完成结果补充。

### R1/R2 协议工具初版记录（2026-11-18；以下为限定复测前的历史状态）

本节记录初版工具及其当时缺口；其中“仅端点”“固定 null”“11 项测试”“不执行矩阵”等描述均不代表当前版本。当前版本的周期采集、三态判定、14 项测试及 run2 进度以上节和板端实际产物为准。多模型家族扩展仍递延。

新增标准库工具 `tools/asr/asr_protocol.py` 与模拟测试 `tools/asr/test_asr_protocol.py`，历史 `board_run_models.py` 和历史解析结果保持原状。版本 1 固定同一 boot_id、每个模型×音频×档位×线程组一次预热后五次正式，所有样本间隔至少 3 秒；没有 M 编号冷缓存规则，不重启、不清缓存、不改 governor。新输出根目录必须不存在，失败不补样、不择优，直接停止矩阵并保留失败/中断状态；强杀未写 finish 时汇总为 running_or_interrupted。每个样本有独立 trace/text/log/execution、before/after 环境及实际进程线程亲和观察。

真实执行只支持 Linux；启动核验配置 CPU 均在允许亲和集合且对应不同物理核。示例单核 CPU0/八核 CPU0–7 仅为草案，必须在目标板核验。显式设置 VLLM_THREADS/VLLM_ASR_PROFILE/OMP_NUM_THREADS，限制 BLAS/MKL 为 1，taskset 绑定子进程；记录线程 banner 与轮询所得实际线程掩码。短进程可能没有有效亲和观察，报告保留原因，不能作为绑定已证明。环境采集每次前后 governor、频率、loadavg、温区、时间及读取失败原因；当前仅有端点快照，缺全程频率证明且温度阈值未冻结，所以 environment_review_passed 固定 null，不伪造环境通过。

解析使用 C 实际输出的 19 叶桶与 encoder/total 汇总，拒绝重复、缺失、非有限、负值和闭合误差 >2%；检查完整 token 序列、逐字文本、解码计数、trace 布局/token/浮点有限性。此检查不替代 R4 数值参考双门禁。每组保留正式五次原值及各桶 median/min/max/range_over_median/population_rsd，预热排除；稳定性单线程 3%、八线程 5%。相对 A 通用阶段回退按 >5% 且 >10ms 登记，D/C >5% 单独登记。完整性、稳定性、环境、回退、收益独立报告，证据不足不输出 performance=true；未解释回退仍阻止通过。

运行接口（所有示例 JSON/输出均放 TEMP，板上正式 out 必须是另一个全新路径）：

```text
python tools/asr/asr_protocol.py freeze --config <TEMP>/draft.json --out <TEMP>/frozen-v1.json
python tools/asr/asr_protocol.py run --config <TEMP>/frozen-v1.json --out <TEMP>/new-run
python tools/asr/asr_protocol.py summarize --input <TEMP>/new-run --out <TEMP>/new-summary.json
python -B -m unittest discover -s tools/asr -p test_asr_protocol.py -v
```

草案 JSON 结构如下（路径必须替换为真实板端绝对路径，冻结时计算 SHA256；预期值必须来自已接受的独立参考，不能从待测结果倒推）：

```json
{
  "schema_version": 1,
  "protocol": {"warmups": 1, "repeats": 5, "interval_seconds": 3, "cache": "same_boot_warm"},
  "models": [{
    "id": "model_f32", "family": "actual_model_family", "tiers": ["A", "D"],
    "model": {"path": "/absolute/model.vqf"},
    "map": {"path": "/absolute/asr_lm.map"},
    "wav_assets": {"path": "/absolute/asr_wav.assets"},
    "expected": {"audio1": {"text": "accepted transcript without final newline", "tokens": [151645], "decode": [0, 0, 1], "projected_count": 1}}
  }],
  "audios": [{"id": "audio1", "audio": {"path": "/absolute/audio.wav"}}],
  "tiers": [{"id": "A", "binary": {"path": "/absolute/asr_A"}}, {"id": "D", "binary": {"path": "/absolute/asr_D"}}],
  "threads": [{"threads": 1, "cpus": [0]}, {"threads": 8, "cpus": [0,1,2,3,4,5,6,7]}]
}
```

`decode` 顺序为 decode_steps/generated_tokens/selected_tokens；tokens 包含停止 token，projected_count 用于验证当前 C trace 布局。冻结输出为每项资产补充 sha256，运行前后重新验证资产；运行目录复制完整配置并记录其 SHA256。模型/音频/档位 ID 禁止双下划线（保留作唯一标签分隔符）。所有模型必须有同线程 A 对照。配置可扩展多个模型家族及音频，但现有 F32/W8 精度对不代表多模型扩展完成，报告对此始终保守标记 false。

本次验证：11 项标准库 unittest 全部通过，包含模拟完整 runner、预热排除、拒绝覆盖、失败不补样、中断/缺 finish、boot 变化、损坏 profile/trace、SHA 变化、总体 RSD、单/八线程 3%/5%、通用与 D/C 回退门槛、多模型配置，以及一个真实 Python 子进程的显式环境/日志测试。两个新增 Python 文件 py_compile 通过；测试产物位于 TEMP 并自动清理，字节码缓存指定 TEMP。Linux taskset/物理拓扑/sysfs 真实采集仍需板端执行时验证。

尚待冻结/执行：真实多模型家族和音频集合、各档编译二进制与全部资产 SHA256、每对模型音频的预期文本/token/计数/projected_count、目标板物理拓扑、温度阈值和全程温频证据审阅方案。当前完成工具实现及 TEMP 模拟，不执行昂贵正式矩阵，不宣称 R1/R2 板端验收或 R8 完成。R4 run2 功能通过及历史 performance_passed=false/environment_review_passed=null 原样保留。

R4 本地实现已完成：hidden/WAV 两门禁支持同 VQF、同音频绑定的清单复用，验证源文件与副本 SHA256；hidden 原始 Python 参考被冻结为 hidden_reference.npz，包含 hidden、projector、逐步 logits/KV 与 21 个 mask 边界参考。默认生成和复用执行相同 C 检查及负例；supplement full 已贯通 --assets-dir，并支持 --export-manifest。

本地真实 W8 验证：默认 full 与复用 full 的 hidden/WAV 子进程全部退出 0；52 项资产检查通过；287 个 WAV 解码用例通过。hidden 误差 3.9577484e-5，hidden 输入回归误差 1.2159348e-5，WAV Python 误差上界 7.9154968e-5，均满足原门限。参考、输入、hidden 输出、WAV trace、mel/hidden 回归输出与文本逐字节一致。阻断 torch、transformers、tokenizers、asr_vqf_run、onnx_to_vqf 导入后复用 full 仍通过；仍需 NumPy，未加载 SciPy。

本地 F32 default full 与阻断重依赖的复用 full 均已完成，hidden/WAV 子进程退出码均为 0，原始记录分别位于 C:/Users/Administrator/AppData/Local/Temp/asr_r4_f32_full_default_20261118_run1 与 asr_r4_f32_full_reuse_20261118_run1。W8 默认记录位于 asr_r4_full_default_20261118，复用记录位于 asr_r4_full_reuse_20261118。旧 WAV-only 清单须重导才能支持 hidden 门禁；清单必须来自可信导出，哈希不提供签名认证。

本次原生板端目录为 `/mnt/ssd/kestrel/asr_r4_20261118_run1`，长期证据目录为 `E:/项目/Kestrel_VLLM_RISC-V/modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/p1_w8_validation/r4_board_20261118_run1`。执行前已通过原 runner complete、90 个唯一正式标签及全部退出码 0 的保护，核实无并行采样任务，NumPy/gcc/objdump 与空间满足要求；上传的 70 个输入/源码文件 SHA256 全部校验通过。采用默认原生编译和 VLLM_THREADS=1，未使用 Linux local-check，未运行 benchmark、未改 C、未提交。

W8 原生 full 已退出 1（1281.25 s）：hidden 子门禁通过，WAV 特征/decoder 检查以及 WAV 数值上界、token、文本断言通过后，在 `asr_c_wav_gate.py:193` 的 `mel_regression.bin` 与冻结参考 `output.bin` 逐字节比较处失败。未修改门槛或替换参考，不将其判定为通过。F32 原生 full 同样退出 1（1660.58 s），hidden 子门禁退出 0，WAV 子门禁在相同逐字节断言处失败；后续 WAV 门禁内的 hidden 回归未执行。W8/F32 独立 boundary 均退出 0，各 17/17 项通过（10 个正例、7 个负例），不替代 full 验收。

两套 mel 输出与冻结参考均为 8123976 字节，mask/token 相同；离线解析得到 W8/F32 最大浮点绝对差分别为 5.7220458984375e-5 / 5.14984130859375e-5，不同字节数分别为 2213418 / 2127086。这是辅助诊断，不改变逐字节回归失败结论，也未定位差异根因。详见长期证据中的 `mel_regression_diagnostic.json`。完整日志、子进程 execution、boundary results、二进制依赖记录及本地 F32 复用记录已封存，`evidence_sha256.json` 记录文件摘要，`input_hashes.json` 保留模型、音频、资产和源码身份。R4 整体仍未验收通过，既有 performance_passed=false、environment_review_passed=null 不变。

2026-11-18 run2 修复与真实重跑已完成，R4 本次功能门禁通过：冻结 manifest/reference 和 Python 三角误差上界保持原逻辑，仅将 Mel 逐字节回归基准分离为本次成功 hidden 门禁输出。新增 `--c-baseline`、成功后生成的模型/音频/平台/输入/输出 SHA256 证明以及错误基准、自比较负例；`full` 在 hidden 退出 0 后传入本轮目录。未改 C，未放宽逐字节断言或 1e-3。本地 `asr_r4_run2_{w8,f32}_{default,reuse}` 四个 full 和两组资产/基准负例均退出 0。板端 `/mnt/ssd/kestrel/asr_r4_20261118_run2` 在 runner complete/90 唯一标签/全部退出 0 及无并行验证检查后串行执行，W8/F32 的 full、prepare、boundary 六项全部退出 0，最终 status=complete。两模型本轮 hidden/output.bin 与 WAV mel_regression.bin 已实际逐字节核对相等，后续 hidden regression 均实际完成；hidden 数值/token/拒绝测试/21 个 mask 边界、WAV 特征/287 个 decoder 用例/全部拒绝测试及每模型 17 项 boundary 均通过。W8/F32 的 WAV Python 误差上界分别为 8.726119995117188e-5 / 8.296966552734375e-5，均小于 1e-3。305 个证据文件（实际命令、日志、报告、产物及输入哈希）已封存至相邻长期目录 `E:/项目/Kestrel_VLLM_RISC-V/modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/p1_w8_validation/r4_board_20261118_run2`，摘要见 `evidence_sha256.json`；自动收集与最终断言脚本退出 0。run1 失败记录保留。未测性能、未提交，既有 performance_passed=false、environment_review_passed=null 不变。

下文历史审查中的因果归因（自动向量化导致卷积回退、W8 打包必然增加流量、qvalue 必然是 LM 慢的根因）仅为待验证假设，不能作为已定位根因；既有中位数本身不能证明这些因果。W8 D/t1 decode 为 19.4952 s，并非所有档位恒定在 18.58 s。unclassified 增加 1.3 ms 未达到原计划“>5% 且 >10 ms”的回退双门槛。不得据此直接互斥 W8_RVV/CONV_PACK 宏或宣称 R5 已解决。

> 审查日期：2026-11-18（Asia/Shanghai）
> 审查对象：`src/model/vllm_asr.c`、`vllm_asr_audio.h`、`vllm_asr_w8_rvv.h`、`vllm_asr_wav.h`、`tools/asr/*`、
> `tools/build/build_riscv.sh`，以及板端正式采样产物目录 `W/`。
> 审查依据：`W/board_continuation_results.json`（`collected_at=2026-11-18T08:33:52.895687+08:00`，
> `runner_status.status=complete`）、`W/board_layers_negatives.json`、`W/board_model_execution.json`、
> 90 份正式日志与对应环境 JSON。
> 本文是 [ASR性能优化落地验收计划.md](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/docs/ASR性能优化落地验收计划.md)
> 的**后续整改文件**，不修改、不覆盖原计划中的历史进度记录。

---

## 0. 历史审查结论摘要（以下数据截至旧 90 次采样；不代表本轮状态）

**一句话结论：优化存在实测正收益，但当前证据链不允许判定为验收通过。**

- 正收益成立，且幅度远高于预注册目标：F32/1 D 档 **5.42×**、F32/1 B 档 **2.69×**、F32/8 D 档 **1.89×**、
  F32/8 B 档 **1.64×**、W8/1 C 档 **1.79×**、W8/8 C 档 **1.63×**（均相对同线程 A 档中位数）。
- 但 `performance_passed=false`、`environment_review_passed=null`，且 18 组中 **7 组**稳定性不达标
  （极差/中位数 > 3%），其中 6 组集中在 8 线程。
- `five_sample_matrix_passed=true` **只表示每组五次样本齐全、检查项完整**，不表示性能或环境验收通过；
  不得引用为达标证据。
- 另有 3 处结构性缺口：板端未执行完整双门禁、`prefill`/`decode` 两个大桶全程未被优化、
  单一冻结音频/模型导致泛化性未知。

因此本轮整改分两条线并行：**证据合规线（R0–R2、R4）**负责把"已有结论"变成"可采信结论"；
**内核收益线（R3、R5–R7）**负责把剩余 37%–59% 的端到端耗时上限打开。

---

## 1. 证据基线与可引用范围

### 1.1 已通过、可直接引用的部分

| 证据 | 规模 | 判定字段 |
|---|---|---|
| 正确性推理矩阵 | 36/36 | `correctness_matrix_passed=true`、`missing_correctness_labels=[]` |
| W8 全档位位级对拍 | 18/18，每份 8123976 字节 | 390 项 mask、hidden、projector、12 步 logits/KV/token 最大差全为 0 |
| 层级负例 | 78/78，整体退出 0 | `layers_negatives_passed=true` |
| 正式性能采样 | 90/90（18 组 × 5 次） | `formal_sample_count=90`、`missing_formal_labels=[]`、`execution_failures=[]` |
| W8 C vs A ≥1.3× | 通过 | `w8_C_vs_A_1_3x_passed=true` |
| 19 叶桶闭合 | 18/18 组 | `plan_leaf_closure_passed=true` |

这五项构成本轮审查的**事实底座**：数值差异不是测量事故或实现错误，而是真实的内核行为差异。

### 1.2 不可引用或需降级引用的部分

| 项 | 状态 | 处理规则 |
|---|---|---|
| `board_progress_snapshot.json`（`status=running`、`formal_count=25`） | 过期运行中快照 | **禁止**与最终 90/90 混用；仅作过程记录 |
| `environment_review_passed` | `null` | 不得写"环境已审阅"；R2 完成后才能改判 |
| `profile_trace.bin` | 多份样本共用，仅保留最后一次 | 不得声称"每个样本都做过 trace 校验"；R2 补齐 |
| 逐样本 governor | 未采集 | 只可引用启动时单次频率快照；不得外推为全程策略 |
| 板端 Python/WAV 门禁 | 旧 90 次采样时未执行；R4 run2 后已通过 | 旧性能矩阵不能追溯补成全链验收；R4 新功能证据见上文 |

---

## 2. 审查结果

### 2.1 18 组正式采样总表

判据：每组 5 次，`stability_passed` 要求极差/中位数 ≤ 3%；`speedup_vs_A` 为同线程 A 档中位数之比。

| # | 模型 | 线程 | 档位 | total 中位数 (s) | 极差/中位数 | 稳定性 | vs A | 关键回退 |
|---|---|---|---|---|---|---|---|---|
| 1 | f32 | 1 | A | 437.8286 | 0.96% | 通过 | 1.0000 | — |
| 2 | f32 | 1 | default | 435.0203 | 1.15% | 通过 | 1.0065 | — |
| 3 | f32 | 1 | B | 162.6411 | **3.01%** | **未通过** | **2.6920** | conv1 +10.25 s、conv2 +2.75 s |
| 4 | f32 | 1 | D | 80.7463 | 0.37% | 通过 | **5.4223** | — |
| 5 | f32 | 8 | A | 104.1099 | **4.94%** | **未通过** | 1.0000 | — |
| 6 | f32 | 8 | default | 103.6783 | **5.88%** | **未通过** | 1.0042 | — |
| 7 | f32 | 8 | B | 63.3315 | 1.33% | 通过 | **1.6439** | projector +0.061 s |
| 8 | f32 | 8 | D | 55.2009 | 0.68% | 通过 | **1.8860** | projector +0.086 s |
| 9 | w8 | 1 | A | 393.1194 | 0.87% | 通过 | 1.0000 | — |
| 10 | w8 | 1 | default | 368.1759 | 2.99% | 通过 | 1.0677 | — |
| 11 | w8 | 1 | B | 393.3396 | 1.09% | 通过 | 0.9994 | — |
| 12 | w8 | 1 | C | 219.0104 | 1.29% | 通过 | **1.7950** | — |
| 13 | w8 | 1 | D | 220.3551 | 1.21% | 通过 | 1.7840 | vs C **0.9939**；projector +0.047 s |
| 14 | w8 | 8 | A | 80.2285 | 0.72% | 通过 | 1.0000 | — |
| 15 | w8 | 8 | default | 77.8379 | **5.19%** | **未通过** | 1.0307 | — |
| 16 | w8 | 8 | B | 80.2654 | **10.78%** | **未通过** | 0.9995 | — |
| 17 | w8 | 8 | C | 49.2351 | **7.62%** | **未通过** | **1.6295** | — |
| 18 | w8 | 8 | D | 51.0522 | **6.41%** | **未通过** | 1.5715 | vs C **0.9644**；conv1 +1.47 s、projector +0.012 s、unclassified +0.0013 s |

档位定义沿用原计划：A=标量基线、default=不加显式 ISA 宏、B=`-march=rv64gcv`、
C=B+`-DVLLM_ASR_W8_RVV`、D=C+`-DVLLM_ASR_CONV_PACK`。

### 2.2 结论 C1–C8

**C1｜历史 W8 C 档有正收益，B 档在该音频下未见明显收益。**
W8 上 B 档相对 A 为 0.9994（单线程）/ 0.9995（八线程），C 档为 1.7950 / 1.6295。
该结果支持继续复测 `VLLM_ASR_W8_RVV` 路径，但不能据单个工作负载断言 ISA 编译选项普遍无效，也不能排除编译器与显式内核的交互贡献。

**C2｜F32 的收益主要来自 `VLLM_ASR_CONV_PACK`，而非 RVV。**
F32/1 D 档把 conv1 从 64.09 s 降到 9.10 s、conv2 从 16.32 s 降到 2.38 s、conv_out 从 102.42 s 降到 0.83 s；
B 档相对 A 的 conv1 +10.25 s、conv2 +2.75 s。自动向量化造成回退是待验证假设，尚无汇编或硬件计数器证据确认；历史数据支持本轮选择 D 复测，不足以证明普遍交付策略。

**C3｜历史优化档中 `prefill`、`decode` 仍占较大耗时。**
F32/1 各档 prefill 约 18.19 s / decode 约 9.64 s；W8/1 A 档 prefill 约 60.60 s / decode 约 18.58 s，D 档 decode 为 19.4952 s，不能称全档位恒定。
在 F32/1/D（80.75 s）中，`load 19.70 + prefill 18.19 + decode 9.64 = 47.53 s`，占 **58.9%**；
在 W8/1/D（220.36 s）中为 `2.91 + 60.64 + 19.50 = 83.05 s`，占 **37.7%**。
这些占比说明仅优化其他阶段的收益受剩余耗时限制；桶级时间本身不能证明 P4 的实施状态，也不能定位 LM 内部瓶颈。LM 优化不属于本次限定复测。

**C4｜W8 的 LM 路径存在结构性劣化，prefill/decode 是 F32 的 3.33× / 1.93×。**
同为单线程 A 档：prefill 60.60 vs 18.19，decode 18.58 vs 9.64。
W8 的 `load` 为 2.91 s（F32 为 19.69 s），但 load 差异不能单独确定 LM 慢的原因。
候选解释包括 `qvalue()` 的 `index%rows` 每元素取模与 [vllm_asr_w8_rvv.h](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr_w8_rvv.h#L21-L33)
非转置分支的 512 次标量逐字节转置。这正是原计划 P5 的两个目标，尚未实施。

**C5｜W8 D 档相对 C 档是净回退，CONV_PACK 在 W8 下不成立。**
单线程 0.9939、八线程 0.9644；八线程 conv1 从 11.58 s 涨到 13.05 s（+12.66%，+1.47 s），
projector +18.6%，unclassified +65%。
打包增加搬运或内存流量属于待验证假设，尚无带宽或向量利用率证据。unclassified 增加仅约 1.3ms，未达到 >5% 且 >10ms 的双门槛。本轮限定 F32 D、W8 C 复测，不据此修改 C 宏组合或宣布普遍交付结论。

**C6｜8 线程测量噪声超标，且随绝对耗时缩短而放大。**
7 组超标中 6 组在八线程（最高 w8/8/B 10.78%），仅 f32/1/B（3.01%）在单线程。
八线程下最小 total 已达 49–51 s，主机侧负载与温频漂移对本就不稳定的并行区更敏感。
但注意：**噪声大不等于可以放宽阈值**——必须先补齐 R1/R2 的协议证据，再决定是否预注册分线程阈值。

**C7｜线程扩展性已被串行段压死，八线程不是免费的。**
以 D 档为例：F32 从 80.75 s（1 线程）到 55.20 s（8 线程），仅 **1.46×**；
同期 A 档为 4.21×、B 档 2.57×。W8 的 D 档为 4.32×（220.36→51.05）。
F32/8/D 中 `load 19.70 s` 占 35.7% 且完全不并行。
即并行收益已被 Amdahl 上限锁住，继续加线程无意义，必须先削减串行段（`load` 的权重/资产解析、`prefill` 的逐位置计算）。

**C8｜泛化与覆盖缺口是"结论边界"，不是"实现缺陷"。**
单一冻结音频 + 单一模型对、未测 2/4/16 线程、板端未跑完整双门禁。
这三点不否定已有数值，但决定了现有结论只能标注为
"在 `W/` 记录的冻结夹具与 1/8 线程下的实测"，不能外推为产品级性能承诺。

### 2.3 为什么 `performance_passed` 仍为 `false`

按原计划第 3、6 章，性能验收同时要求：(a) 每组 5 次且稳定性达标；(b) 环境审阅通过；
(c) 阶段回退已解释或修复。当前三条依次为：**7 组不达标**、**null**、**W8 D→C 回退未解释**。
三者任意一条成立即不得判通过，因此该字段为 `false` 是正确的，不是脚本 bug，也不应改判。

---

## 3. 整改项

优先级：P0 = 阻塞验收结论；P1 = 阻塞收益上限；P2 = 收尾与文档。
所有整改项均须遵守原计划第 0 章不变量 I1–I5，不得以放宽判据换取通过。

### R0｜证据单一真源与时效治理（P0）

- **问题**：`board_progress_snapshot.json` 曾出现 `status=running`/`formal_count=25` 的过期状态，
  与 `board_continuation_results.json` 的 90/90 并存，存在被误引用的风险。
- **证据**：C1–C8 全部依赖 `runner_status.status=complete` 这一条；快照文件无 `is_final` 标记。
- **整改动作**：
  1. 在汇总产物中新增显式字段：`is_final: true/false`、`superseded_by`、`evidence_scope`。
  2. 所有下游文档引用时只允许指向 `is_final=true` 的产物；过期快照改名加 `_stale_` 前缀保留。
  3. 建立 `docs/asr_perf/README.md` 作为证据索引，逐项列出文件、SHA256、判定字段、可否引用。
- **验收判据**：索引覆盖 `W/` 下每个证据文件；任一文档中的数字都能在索引内定位到唯一来源文件与行区间。
- **依赖**：无。

### R1｜八线程测量协议重建（P0）

- **问题**：6/8 的八线程组极差 > 3%，最高 10.78%；当前协议无法区分"真实抖动"与"环境漂移"。
- **证据**：C6；`range_over_median` 在八线程组为 4.94%–10.78%，单线程组为 0.37%–3.01%（含 B 档 3.01%）。
- **整改动作**：
  1. 每个正式样本固定 CPU 亲和（`taskset -c` 限定 8 个物理核），并在样本环境 JSON 中记录实际掩码。
  2. 每组采样前丢弃 1 次预热样本（不计入 5 次），用于消除页缓存与 TLB 冷启动差异。
  3. 同组 5 次之间强制 ≥3 s 间隔，记录每次的 loadavg、温区、`scaling_cur_freq`、`governor`。
  4. 输出统计除中位数外，增加 min/max/极差/相对标准差，并保留 5 个原始值（当前已保留，需在报告中显式列出）。
  5. 若整改后仍有组超标，先定位到具体叶桶，不得直接放宽阈值。
- **验收判据**：18/18 组 `stability_passed=true`；八线程组极差 ≤ 3%；
  且每个样本的环境 JSON 含 governor 与亲和掩码字段。
- **依赖**：R2（同一次上板一并采集，避免重复长跑）。

### R2｜逐样本 governor 与独立 trace 补齐（P0）

- **问题**：`environment_limitations` 明确指出 runner 不记录逐样本 governor；
  `profile_trace.bin` 共用，90 个样本中只有最后一次 trace 可查。
- **证据**：`environment_review_passed=null`；`environment_limitations[0..1]`。
- **整改动作**：
  1. runner 在每个样本前后各采集一次 `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
     与 `scaling_cur_freq`，写入该样本专属环境 JSON。
  2. 每个样本使用独立输出子目录（如 `profile/<label>/`），使 `profile_trace.bin` 按样本隔离。
  3. 环境审阅脚本按"全部样本 governor 一致 + 温区在阈值内 + 频率未触顶降频"给出布尔判定，
     取代当前手工 `null`。
- **验收判据**：`environment_review_passed=true`；样本级 trace 文件数 = 90；
  且任取 3 个样本可独立完成 19 叶桶闭合复算。
- **依赖**：R1。**注意**：不得为满足本条而重启板端或清缓存伪装冷启动；若协议变更，必须在报告中标注实际协议。

### R3｜`prefill` / `decode` 桶优化（P1，即原计划 P4）

- **问题**：两桶在全档位恒定不变，占 F32/1/D 的 34.5%、W8/1/D 的 36.4%，是端到端上限。
- **证据**：C3、C4；叶桶数据见第 2.1 节来源文件的 `leaf_median_seconds`。
- **整改动作**：
  1. `step()` 末尾的 151936×512 logits 投影改为 RVV 实现，保持 double 累加与
     "输出 lane 分块 → k 递增 → bias 最后加"的顺序。
  2. argmax 向量化，保留跳过 151646–151650 五个占位 id 的语义与"取首个最大值"的 tie 行为。
  3. 对 W8 路径单独评估 `qvalue()` 的取模消除（见 R6），二者合并测量。
- **验收判据**：
  - 每步 `logits` 的 `max_abs ≤ 1e-4`，且 12 步 token 序列与参考完全一致。
  - trace 模式 dump 的 VOC 个 float 与标量路径**逐字节相同**。
  - `decode` 桶相对本阶段起点下降 ≥ 25%；`prefill` 不退化（>3% 需说明）。
  - 合成用例：把 5 个占位 id 置为全局最大，断言不被选中；构造大量相等 logits，断言 tie 行为一致。
- **依赖**：R1（否则收益无法与噪声区分）。

### R4｜板端完整双门禁落地（P0，合规）

- **问题**：`scope_limitations[2]` 指出板端缺完整 Python/WAV 门禁依赖，现有结论只有"冻结 C trace 对拍"。
- **证据**：`asr_c_gate.py` 依赖 `numpy`、`scipy`；`asr_c_wav_gate.py` 无条件调用 `export_assets`，
  而 [asr_c_assets.py](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/tools/asr/asr_c_assets.py#L7-L10) 依赖
  `numpy`、`tokenizers`、`transformers`、`torch`。riscv64 上完整安装该依赖链不现实。
- **整改动作（两条互斥路线，需选择其一并记录理由）**：
  - **路线 A（依赖补齐）**：若板端可安装 riscv64 版 numpy/scipy 且能离线安装 tokenizers 相关包，
    直接执行 `asr_supplement.py full`；不引入任何代码改动。
  - **路线 B（资产复用）**：为 `asr_c_wav_gate.py` 增加 `--assets-dir` 复用路径，
    读取在 x64 由**同一 VQF** 导出的 assets/reference，并对每个文件做 SHA256 断言；
    断言失败即拒绝运行。导出函数是 VQF 的确定性函数，因此复用不改变判定强度。
    该路线需**代码评审**并证明：复用路径与重新导出路径在 x64 上产生逐字节相同的资产。
- **验收判据**：板端 `asr_supplement.py boundary` 全绿、`full` 退出 0；
  `binary_imports.log` 仍无 `python/onnxruntime/tokenizers/torch` 依赖；
  9 个边界 WAV + extra chunk + 5 个 malformed + 2 个损坏 assets 全部按预期通过/拒绝。
- **依赖**：无（可与 R1/R2 并行准备，但板上执行需与性能采样错开，不得叠加负载）。

### R5｜W8 档位收敛与 D→C 回退治理（P1）

- **问题**：W8 D 档相对 C 档单线程 0.9939、八线程 0.9644，属净回退；conv1 八线程 +1.47 s。
- **证据**：C5；`stage_regressions_gt5pct_vs_C` 中 conv1、projector、unclassified 三项。
- **整改动作**：
  1. 交付矩阵收敛为：**F32 用 D 档，W8 用 C 档**；`VLLM_ASR_CONV_PACK` 在
     `VLLM_ASR_W8_RVV` 同时开启时默认关闭（编译期互斥），并保留显式覆盖入口以便复测。
  2. 复测 W8/8/C 与 W8/8/D 各 5 次，确认互斥后 D 档不再劣于 C 档。
  3. 若必须保留 W8+D，需给出 conv1 打包路径的搬运量分析并证明可被 `vle8`/`vsext` 摊销。
- **验收判据**：W8 D 相对 C 的 `speedup_vs_C ≥ 1.00`，且 `stage_regressions_gt5pct_vs_C` 为空；
  F32/1 与 F32/8 的 D 档收益不回退（1 线程 ≥5.0×、8 线程 ≥1.8×）。
- **依赖**：R1。

### R6｜W8 `qvalue` 取模与 tile 转置（P1，即原计划 P5）

- **问题**：W8 的 prefill/decode 为 F32 的 3.33×/1.93×，是 W8 路线最大的剩余项。
- **证据**：C4；[vllm_asr.c](file:///e:/项目/Kestrel_VLLM_RISC-V/kestrel_vllm_riscv/src/model/vllm_asr.c#L114-L117)
  每元素一次 `index%rows`/`index/rows`；`vllm_asr_w8_rvv.h` 非转置分支的标量转置。
- **整改动作**：
  1. `qvalue()` 改为预计算行/列（或加载期建立偏移表），消除每元素取模。
  2. 非转置布局在**加载期一次性转置**，使内层循环只用 `vle8`+`vsext_vf4`。
  3. 两条改动分别独立提交、独立测量，避免收益归因混淆。
- **验收判据**：
  - `asr_w8_test.c` 全形状（含 `{7680,896}`、`{896,3584}`、`{4096,1024}`）位级 `memcmp` 相等。
  - `--bad-scale` 四个负例（nan/zero/negative/inf）仍被拒绝。
  - `norm_qkv + ffn + tail_mlp` 合计下降 ≥ 10%。
  - 若采用加载期转置：`load` 桶增幅 ≤ 20%，峰值内存增量 ≤ 原权重体积 1.0 倍。
  - W8/1 的 prefill+decode 合计目标下降 ≥ 15%。
- **依赖**：R3（同属 LM 路径，合并测量可避免互相干扰）。

### R7｜线程矩阵补齐 2/4/16（P1，即原计划 P7）

- **问题**：`scope_limitations[1]` 明确 2/4/16 线程不在 runner 内；现有 1/8 两组无法定位扩展性拐点。
- **证据**：C7；F32 D 档 1→8 线程仅 1.46×。
- **整改动作**：在 R1 的新协议下补测 F32-D 与 W8-C 的 `VLLM_THREADS ∈ {2,4,8,16}`。
  同时验证 `vllm_tp_worker_id()<0` 嵌套保护与 worker 内不再派生。
- **验收判据**：total 随线程数单调不增（允许 ±5% 抖动，倒挂需登记原因）；
  8 线程相对 1 线程 ≥ 2.0×（不满足则说明同步开销吃掉收益，需调整 `asr_parallel` 粒度）；
  同组 token 序列与单线程完全一致。
- **依赖**：R1；建议在 R3/R6 收益落地后再跑，否则拐点会漂移。

### R8｜泛化验证（P1）

- **问题**：`scope_limitations[0]`：每种精度只有一组冻结音频/模型。
- **证据**：C8；现有结论无法覆盖长音频、静音、噪声与不同说话人。
- **整改动作**：新增 ≥5 段音频（含 >30 s 长音频、纯静音、带噪语音、两人对话、样本率 22050 重采样各一），
  每段跑：正确性推理 1 次 + F32-D 与 W8-C 各 5 次性能采样 + 双门禁。
- **验收判据**：全部音频 hidden/projector `max_abs ≤ 1e-3`、token 与参考一致；
  性能相对 A 的收益方向不反转（允许幅度不同）；长音频不出现桶闭合失败。
- **依赖**：R1、R4。

### R9｜收尾：结论口径与交付矩阵固化（P2）

- **整改动作**：
  1. 在验收计划第 3 章预注册分线程阈值：单线程 ≤3%，八线程 ≤5%（**先注册后测量**，不得倒推）。
  2. 交付矩阵写入构建脚本注释与文档：F32 → D 档；W8 → C 档；B 档仅作对照，不交付。
  3. 更新 `performance_passed` 的判定逻辑为四元组
     `(five_sample_matrix ∧ stability_all ∧ environment_review ∧ no_unexplained_regression)`，
     并在产物中同时输出四个分量，避免单一布尔被误读。
- **验收判据**：任取一个历史产物，能复算出与文档一致的四个分量；
  文档中不存在"看起来更快"这类无数值支撑的表述。
- **依赖**：R1–R8 全部完成后执行。

---

## 4. 执行顺序

四条线按依赖串行/并行组合，**同一时间板上只允许一条测量线运行**。

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| S1 | R0（证据索引）+ R2 改造（runner 采集 governor 与独立 trace）+ R4 路线选择与准备 | 索引可用；runner 改动通过本地 dry-run；R4 路线已书面确定 |
| S2 | R1 + R2 上板复测 18 组 | 18/18 稳定性通过；`environment_review_passed=true` |
| S3 | R5（档位互斥）+ R4 上板执行双门禁 | W8 D 不再劣于 C；板端双门禁全绿 |
| S4 | R3 + R6 内核实现与位级回归 | 位级/`max_abs`/token 全绿，且 decode 降 ≥25%、W8 LM 降 ≥15% |
| S5 | S4 收益的 18 组复测 + R7 线程矩阵 | 全部稳定性与单调性判据通过 |
| S6 | R8 泛化 + R9 口径固化 | 多音频全绿；四分量判定落地 |

S1 与 S4 可并行推进（前者只动 runner 与工具，后者只动内核），但**不得同时上板**。

---

## 5. 回归与不得破坏项

每个整改项合并前必须重跑原计划第 2 章 8 项回归矩阵，且额外确认：

- [ ] `-ffp-contract=off` 仍在编译命令中（I1）。
- [ ] 累加顺序未变，`asr_w8_test.c` 位级 `memcmp` 全等（I2）。
- [ ] 二进制依赖中无 `python/onnxruntime/tokenizers/torch`（I3）。
- [ ] VQF 全文件 FNV 校验、形状白名单、图哈希（`da046a67…`、`c03f10a6…`）原样保留（I4）。
- [ ] 每条新优化路径都有编译期逃生舱；关闭后回到标量路径且门禁全绿（I5）。
- [ ] `asr_supplement.py` 的三重保护仍然生效：原目录 `model_status.json=complete`、
      90 个唯一标签齐全、输出目录不存在。

---

## 6. 风险登记

| 编号 | 风险 | 影响 | 缓解 |
|---|---|---|---|
| RR1 | 八线程极差即便加固仍 >3% | R1 无法收口，验收长期阻塞 | 先做 S2 实测；若仍超标，按 R9 预注册分线程阈值并附噪声下界证据 |
| RR2 | R4 路线 B 的资产复用被质疑削弱门禁 | 合规风险 | 在 x64 证明复用/重导资产逐字节相同，并纳入回归矩阵 |
| RR3 | R3 的 logits 向量化破坏 trace 字节一致性 | 门禁输入失效 | trace 路径独立保留标量实现，仅 decode 快路径向量化 |
| RR4 | R6 加载期转置抬高 `load` 桶与内存 | 端到端收益被抵消 | 按判据设 20% / 1.0× 上限，超标即回退到运行期转置 |
| RR5 | 并行粒度调整引入非确定性 | token 序列漂移 | 每次改动都断言 token 序列与单线程一致 |
| RR6 | 单一音频结论被外部误用为产品承诺 | 结论失真 | R8 完成后才允许对外表述性能数字，并附夹具 SHA256 |

---

## 7. 交付物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 本审查与整改计划 | `docs/ASR性能优化审查与整改计划.md` | 本文件 |
| 证据索引 | `docs/asr_perf/README.md` | R0 产出，含 SHA256 与可否引用标记 |
| runner 改造 | 板端编排脚本 + 样本环境 JSON 模板 | R2 产出，含 governor 与亲和掩码 |
| 补验结果 | `W/supplement_boundary_A/`、`W/supplement_full_*/` | R4 产出 |
| 内核阶段报告 | `docs/asr_perf/P4.md`、`P5.md` | R3、R6 产出，按原计划第 4 章模板 |
| 复测汇总 | `W/board_continuation_results_retest.json` | S2/S5 产出，`is_final=true` |
| 泛化报告 | `docs/asr_perf/generalization.md` | R8 产出 |

---

## 8. 待决策项

以下 3 项会改变整改范围，需在 S1 结束前确认，确认结果写入本文件第 8 节并注明日期：

1. **R4 路线**：A（板端补依赖）还是 B（`--assets-dir` 资产复用 + SHA256 断言）。
   B 需要一次代码评审，但可立即执行；A 不改进代码，但依赖 riscv64 轮子的可获得性。
2. **分线程稳定性阈值**：是否接受"单线程 3% / 八线程 5%"的预注册口径。
   接受则 R1 的八线程判据改为 5%，但必须在测量前注册，且需附噪声下界证据。
3. **泛化范围**：R8 的 5 段音频是否足以支撑对外结论，或需扩大到多模型精度对（会显著拉长板上时间）。

---

## 9. 与既有结论的关系

- 本文**不推翻** `W/board_continuation_results.json` 中的任何数值，只重新界定其可引用范围。
- 本文**不接受**把 `five_sample_matrix_passed=true` 或任一 `speedup_vs_A` 数值单独作为验收通过。
- 本文**不修改**原验收计划的历史进度记录；原计划继续作为判据来源，本文作为整改执行依据。
