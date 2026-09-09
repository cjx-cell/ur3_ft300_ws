# PAP-MoE 当前任务与接手指南

更新：2026-09-09。**本文是唯一当前状态入口；有日期的实验报告只描述当时合同，不作为默认启动命令。**

## 1. 目标与不可改变的要求

- 目标是 PAP-MoE 整体在指定物理工况（密集接触、视觉失效、安全接触、受力卡死恢复等）优于同信息基线；不是“恢复 Pi0.5 原有能力再加修正”。当前尚未证明这种优势。
- 当前只使用现有 Workspace50，不新增恢复数据、不擅自开始训练。
- PAP 没有 subtask、skill-progress 或独立夹爪/手臂决策头；四专家提供特征，动作专家负责完整动作生成。
- 冻结同数据、归一化、执行合同后做单变量对照。不能把动作差、训练 loss 或开环 MSE 改善直接当作闭环提升。
- 不改原始数据/旧检查点的 processor，不覆盖旧结果，不隐藏失败，不把示范或专家接管视频写成模型成功。

## 2. 当前状态：没有训练；核心验证被复现门槛阻断

最近的四模型配对及完整 Flow 回放已完成：

`artifacts/deployment_validation_20260909_s4_success_lineage/`

- 16/16 闭环：S4 2/4、B 2/4、GT 2/4、Pred 1/4。
- 组合：0001/seed1、0031/seed2、0041/seed1、0041/seed2。按历史 S4 成功选择，只是回归诊断集。
- 四模型各 932 次完整 Flow（RTC/无 RTC）；同观测/噪声/公共尾块匹配。Gate 预测逐值一致，S4 RTC 自回放最大差 0。
- S4→B 在第0块已有约0.020–0.025 rad 的关节差；这是动作映射变化，不证明该变化导致失败。S4→B同时包含接口改变与联合训练，不能归因于单独一项。
- `completion.json`、`final_audit.json` 已完成/通过；不重跑该 pipeline。

之后开始物理条件信息验证：

`artifacts/deployment_validation_20260909_core_information/`

- `manifest.json`：16条各自真实 rollout，191个预先选定观测；包括几何阶段边界、首次闭爪与终止前3块。
- 三组：正常条件；融合强度0；保持当前路由但替换成上一观测的同身份专家特征。后者同时改变专家中融合的视觉/状态/力表征，不能单独证明力模态有效。
- 四模型离线已完成，`status.json=offline_complete_execution_pending`。各模型 `completion.json`、`records.jsonl` 和 NPZ 保存完整 Flow；native 自回放、路由一致、零融合诊断均设有强校验。**尚缺离线效果汇总，不要把差异直接称为准确率。**
- 实际分支 `branches/run.py` 已自然停止：`status=blocked_start_state_mismatch`，只完成 B / 0041 / seed1 的正常前缀重放，终局 timeout；没有执行正常重复或两个干预组。
- 第6块匹配差：关节最大0.000635 rad、夹爪0.001532 rad、peg位置0.525 mm、快力RMS0.108 N；但关节历史差分0.007056 rad超过0.005门槛，peg姿态3.361°超过2°门槛。
- 因此**当前只知道接触/历史状态复现未通过，不能推断物理专家无效**。已记录于 `branches/results.json`、`branches/report.md`。不要自动放宽阈值/重跑同目录。
- 旧暂停调度 PID857240 仍只是保留现场，禁止 SIGCONT 或当作当前任务恢复。进程号仅供核对，执行前检查 cmdline，不能凭历史PID杀进程。

## 3. 下一步执行清单（按顺序）

1. 汇总已完成的191观测条件对照，按模型/几何阶段列完整动作差、夹爪输出、当前/历史特征差；正常输出是参考而非动作真值。只筛选诊断候选，不宣称信息收益。
2. 检查分支复现偏差：用 `branches/grasp_native_worker.jsonl`、原/新 `resident_requests.jsonl`、`diagnostic/controller.jsonl`、`geometry_shadow.jsonl` 对齐第0–6块。区分观测历史的采样时序与peg真实转动；姿态需拆分轴倾斜与轴向旋转，确认圆柱轴对称性是否使某些旋转无关。**先验证几何语义，不能为过门槛任意放宽。**
3. 固定改动前后的复现合同，先做正常前缀重复。已有工具没有精确保存隐藏接触求解器状态/物体速度；仅重置关节位置不够。如果无法可靠匹配，应明确改用多次配对随机化干预，而不是冒称同状态因果实验。
4. 复现通过后再正常/旁路/历史特征分支，每次只干预1块再交回模型。先抓取，再运输/插入，保存实测抬升、滑脱、横向偏差、插入几何和发送/实测跟踪。要证明物理信息价值，不仅是依赖某条网络路径。
5. 有实际证据后再决定训练或接口修改；不默认重训、不恢复基线锚定损失、不同时改接口/Gate/数据。
6. 专项优势验收仍未完成：E2真实视觉失效、受力卡死恢复、安全接触、严格主动插到底/释放稳定，以及同输入非MoE基线。现有训练内5位置不支持多任务泛化声明。

## 4. 模型与训练合同

双相机/语言视觉前缀、关节状态及历史、FT快慢窗口、视觉质量/历史 → 四物理专家；PhysicsGate直接输出50×3因子（b视觉失效、c接触、m可运动性）。原始路由：

`[(1-b)(1-c), b, c(1-m), cm]`，再归一化。

四专家语义：E1正常视觉自由运动；E2视觉退化备援；E3刚性约束接触；E4可运动/顺应接触。视觉失效接触时E2可与E3/E4协同。先验不应直接依赖某个轴孔的阶段/高度/几何；三因子是估计而非完美物理真值。

每个未来动作步使用对应路由，加权专家特征在 **Action Transformer 前** 融合，再用完整 Flow 生成50步7维绝对动作。不是四专家各自产生最终修正动作。没有Forecaster或草稿动作的部署分支。

训练基线后：S2用真实未来路由联合训练四专家与完整动作专家；S3监督PhysicsGate；S4冻结其他模块，用动作损失与路由正则适配Gate。最新B→GT/Pred实验另追加专家+完整动作专家联合训练，Gate/VLM/公共Gate编码器冻结；不要和旧S4只训Gate混淆。

## 5. 数据、权重和环境（必须保留）

根 `/home/ubuntu/ur3_ft300_ws`；ML环境 `/home/ubuntu/miniconda3/envs/pi0-env`；ROS `/usr/bin/python3`。

| 数据角色 | 根目录下路径 |
|---|---|
| 原始50条/14418帧、5位置×10风格 | `pap_moe_framework/datasets/workspace_50_v10_canonical` |
| Pi0.5正确全帧统计视图 | `pap_moe_framework/datasets/lerobot_v3_workspace50_v10_baseline_global_stats_v1` |
| PAP正确全帧统计视图 | `pap_moe_framework/datasets/lerobot_v3_workspace50_v10_full_clean_global_stats_v1` |
| 旧错误分位数视图（只追溯） | `pap_moe_framework/datasets/lerobot_v3_workspace50_v10_full_clean` |

camera0腕部、camera1全局，224×224 RGB；Pi0.5两真实相机加一空相机，tokenizer长度200。state是实测7关节；action是下一个控制目标，连续rad，夹爪范围0–0.8，抓住物体实测约0.63不等于命令也应该固定为0.63。力归一化MEAN_STD；动作/状态使用全帧q01/q99。必须检查实际processor张量，不只JSON类型/代数往返。

下表均在 `outputs/train/`，后缀为 `checkpoints/<step>/pretrained_model`：

| 名称 | 目录 | step |
|---|---|---|
| Pi0.5参考 | `pi05_workspace50_global_stats_expert_only_30k_20260827` | 030000 |
| 原S4 | `pap_corrected_gate_calibration_20260907_104300` | 015000 |
| B | `pap_interface_ab_20260909_B_continuous` | 010000 |
| GT | `pap_route_adaptation_20260909_GT` | 010000 |
| Pred | `pap_route_adaptation_20260909_Pred` | 010000 |

GT/Pred同B起点各追加10000步，455,104,947可训练参数（四专家/动作侧），无新增恢复数据。预训练权重 `ai-models/pi05/pi05_libero_base`。模型/数据未公开下载，不把本地路径当Hub地址。

## 6. 默认工作区 vs 冻结实验版本

- 当前实验必须使用 `artifacts/deployment_validation_20260909_route_adaptation/code/lerobot` 与 `resident_code`。默认 `/home/ubuntu/lerobot` 不含全部后续接口改动；不能直接拿默认入口加载所有候选并宣称相同实验。
- 夹爪修复候选：`artifacts/deployment_validation_20260909_gripper_candidate/install`。实际插件SHA256 `383681a9db7958b838f1a8938c17e694c4f419a7397fa46e6b726fd8aece0ee3`。仅新read累计停滞，避免500Hz write重复统计100Hz旧反馈。默认安装未推广该候选。
- 对照合同：50预测/10执行；arm-only EXP RTC，max10、delay0；0.13rad/0.1s控制器步限；120仿真秒；xy≤8mm、peg_z<0.89、连续5检查。CAD几何影子另记；原成功不等于严格释放稳定。
- 归一化污染、持续力过滤、数据读取竞态、夹爪裁剪、实际加载插件等均有已定位问题及局部修复。不能说“所有工程问题已排除”。旧坏统计结果不和新统计混算，不能给旧模型热换统计。
- 清理删除了未使用的旧SA脚本/注册，运行源文件集合因此改变。旧 `runtime_source_manifest` 中已移走文件应从清理映射恢复到隔离副本再核验；不能就地覆盖历史哈希。下一实验需重新冻结完整代码，不直接重启硬编码旧pipeline。

## 7. 已有证据与结论

| 问题 | 已知事实 | 不能推断 |
|---|---|---|
| 开环好闭环差 | 曾出现阶段性MSE收益但运输/接触转换退化 | 开环收益保证闭环提高 |
| 新模型全失败？ | seed0的GT/Pred各0/5；四成功组合回归为2/4、1/4 | 所有位置能力都丧失 |
| Gate还是动作映射？ | 同输入四模型Gate逐值一致，动作不同 | 只靠差异认定某专家无效 |
| 新模块是否有信息？ | 表征/消融有动作依赖性；执行信息收益未证实 | 去掉条件退化就证明物理泛化 |
| 原S4历史成功是否可复现？ | 旧4/4选择集在修复插件下2/4 | 历史与新分数可直接作因果对照 |
| 最新分支 | 起始关节历史/peg姿态未匹配，干预未执行 | 物理专家造成当前失败 |

主要报告：`reports/PAP_MoE推进顺序与实验结论_简版_20260907.md`（全时间线）、`PAP_MoE部署条件有效性验证_20260909.md`、`PAP_MoE固定融合_真实与预测路由联合适配_20260909.md`、`PAP_MoE_S4成功组合配对与动作分叉_20260909.md`、`PAP_MoE物理条件信息有效性验证_20260909.md`。原始产物仍在artifacts，不删失败。

## 8. 项目整理与公开版本


旧SA仅保留 `LEGACY_SA_MOE.md` 说明；本地49项旧入口/重复资料已移到 `/home/ubuntu/ur3_ft300_retired/20260909`，`maintenance/release_20260909/removal_manifest.json`可恢复，永久删除0项。数据、有效权重和冻结证据保留。

`learning/` 发布锁定上游提交及52个增量/测试文件，不完整复制LeRobot，不迁移当前环境；官方独立插件方案可行但训练器/采样器/RTC等改动仍需切分和回归。上游是公开提交`ba27aab79c731a6b503b2dbdd4c601e78e285048`，不是不可从官方检出的本地HEAD `c48f7bc7`。详见 `LEROBOT_INTEGRATION.md`。许可证/贡献看 `CONTRIBUTIONS.md`，启动看 `RUNNING.md`。

接手时先读本节状态，再检查JSON/实际进程；不要重复过去“阶段已完成但旧追加日志仍写正在训练”的错误。
