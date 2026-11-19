# ASR 性能证据索引

> 建立日期：2026-11-18（Asia/Shanghai）
> 本索引记录当前可见的最终 ASR 板端证据及其引用边界。来源文件均保留原路径，不改写、不移动。
> SHA256 是本机索引时对现存文件计算的摘要；它证明后续可检测文件是否变化，**不证明来源真实性、历史采集时哈希或验收通过**。

## 当前最终判定

- 正式样本：90/90，18 组各 5 次；汇总 `runner_status.status=complete`。
- 正确性矩阵：36/36；层级负例：78/78；W8 C 相对 A ≥1.3×：通过。
- `five_sample_matrix_passed=true` 仅表示样本组齐全，不表示性能验收通过。
- `performance_passed=false`；`environment_review_passed=null`。
- 7/18 组极差/中位数 >3%；无逐样本 governor；共享 trace 只留最终样本。
- 限定范围：每精度仅一对冻结音频/模型；不含 2/4/16 线程；板端未运行完整 Python/WAV 双门禁。

## 核心最终证据

| 文件 | SHA256 | 采集/身份 | 关键字段与范围 | 可引用性 |
|---|---|---|---|---|
| `board_continuation_results.json` | `91B65155B01F1539322E343A85C280750002A02756A6731A378B75CB848D48C5` | 最终汇总；`collected_at=2026-11-18T08:33:52.895687+08:00`；runner complete | 第 7373 行起 18 组结果；第 8134–8159 行总判定/限制；90/90、performance=false、environment=null | **最终性能结论唯一真源**；仅支持冻结夹具、1/8 线程下的结果 |
| `board_layers_negatives.json` | `9EB04C0EB46566ACE387FF69BF8D80E70803C0BD036B19D9E594DDFDA227A6F8` | 板端层级/负例结果 | 计划记载 78/78 通过；按原 JSON 逐项复核时以实际字段为准 | 支持已覆盖用例，不替代完整门禁 |
| `board_comparisons.json` | `126D9880E7341CC594B8A8CB178B40F50D3628BA3A5F42548D2E141BCEFB3F43` | 板端正确性比较汇总 | 正确性矩阵 36/36；实际细节以 JSON 为准 | 支持这些冻结正确性对照，不作性能结论 |
| `board_model_status.json` | `BEDDEBACB141FDF84309E392F971CC8BC984DC8702B128DAB3332D289B177AE4` | 当前保存状态 `complete` | 单独不能证明样本完整；须与执行清单、最终汇总交叉核验 | 仅作状态证据 |
| `board_model_execution.json` | `F35FB26A70CEC08C82D9B78CF316C720C0C070903289C5F1FF3FA1A3D4ACD5BC` | 板端执行记录 | 与汇总核对样本退出码、正式标签 | 执行历史，不单独判性能 |

## 位级对拍与硬件证据

| 文件 | SHA256 | 判定范围 | 可引用性 |
|---|---|---|---|
| `board_w8_AC_exact.json` | `3B1242825E3DC5C126AA82FA83F4E6A4E02DC86AA4E0AD18CE98205F16078B47` | W8 A/C 对拍，1 线程 | 仅列明输入与覆盖范围内的位级结果 |
| `board_w8_AD_exact.json` | `5CD664C27CADF28FE599C9848F607461CA293E7B120BE915473F7E3006164EFC` | W8 A/D 对拍，1 线程 | 同上 |
| `board_w8_AC_t8_exact.json` | `6F8328A934B5DBAAEB23663AE4DFD5B69B964C59A29561F27E765AD5DDC8FD5E` | W8 A/C 对拍，8 线程 | 同上 |
| `board_w8_AD_t8_exact.json` | `BBD3683EEE3F1397354EF364EFD789E4C9D601F14DFACE0F6B168BEF835F04EC` | W8 A/D 对拍，8 线程 | 同上 |
| `board_f32_AB_exact.json` | `A78771E49B1559D5CA8D6A53A9A5F7CEB9CEC66A85214C6F09EAF5B7C54618B2` | F32 A/B 对拍 | 不外推到未覆盖档位/形状 |
| `board_vlen_probe.json` | `11BD2F5289E3D9E3F73BEA1E5D84C3FC407EFC1D058FC3697ED8BDC02EB4A2D7` | 板端 RVV VLEN 探针 | 仅支持探针实际记录的硬件信息 |
| `asset_sha256.json` | `D6975453C03D34C2365B7C13D2E9755F1110EFA664F4D5593B4F81FF38D07A47` | 输入、资产及本地二进制哈希清单 | 支持冻结输入身份；不代表性能验收 |

## 正式采样文件集

正式日志和环境 JSON 位于同一目录 `W/`：

- `board_profile_{model}_{config}_t{threads}_r{repeat}.log`：90 个正式 profile 日志。
- `board_profile_{model}_{config}_t{threads}_r{repeat}_env.json`：90 个对应环境文件。
- 标签矩阵：F32 为 A/default/B/D × 线程 1/8 × repeat 1–5（40 组文件）；W8 为 A/default/B/C/D × 线程 1/8 × repeat 1–5（50 组文件）。
- 每个正式样本日志与 env JSON 通过 `{model}_{config}_t{threads}_r{repeat}` 标签配对；汇总中 `missing_formal_labels=[]`、`execution_failures=[]`、`sample_validation_failures=[]`。
- 逐样本 governor 未采集，不能由其他环境字段补推。profile trace 共用且仅保留最终 trace，不能声称 90 份逐样本 trace 已验证。
- 这 180 个文件以精确标签和总数清单索引；本索引未对 180 个文件逐一计算 SHA256。若文件需封存或迁移，应另生成不可覆盖的完整清单，不能把本条描述当作逐文件哈希校验。

## 本地门禁和补验的边界

本地双门禁、WAV 门禁以及补验脚本的本地回放记录在 [ASR性能优化落地验收计划.md](../ASR性能优化落地验收计划.md) 第 450–473、514–525、621–646 行。它们只支持所述本机环境与冻结夹具；本地 `17/17` 补验不等价于板端双门禁。完整 Python/WAV 双门禁在板端尚未执行。

## 过期/非最终证据

| 文件 | 状态 | 引用规则 |
|---|---|---|
| `board_progress_snapshot.json` | 历史运行中快照；曾记录 `running`/部分样本 | 仅过程记录，不得覆盖最终汇总 |
| `board_progress_watch_20261118.json` | 过程监视快照 | 仅过程记录 |
| `early_start_model_execution.json`、`early_start_model_status.json` | 早期启动记录 | 仅过程记录，不与最终 90/90 混作单一状态 |
| 其他 `*_early*`、`*_initial_failure*`、`*_rerun_health*` | 问题定位或中间复跑 | 解释历史，不作为最终验收判定 |

**保留策略：** 不重命名、不移动、不覆盖上述历史文件，以免改变引用与来源；用本索引中的状态标签解决时效歧义。只有确认存在另一个索引外的旧文件时，才考虑归档更名。

## 原始来源

- 最终汇总：`modls/Edge0--Audio8-ASR-0.1B-onnx-runtime/model_bundle/p1_w8_validation/board_continuation_results.json`
- 计划与阶段边界：`kestrel_vllm_riscv/docs/ASR性能优化落地验收计划.md`
- 本轮审查与整改判定：`kestrel_vllm_riscv/docs/ASR性能优化审查与整改计划.md`

本索引不修改原始产物，不改变既有阈值，不将历史 `null` 或失败状态升级为通过。重新采样后应新增独立汇总文件并更新本索引的“当前最终证据”，保留旧条目及其哈希。
