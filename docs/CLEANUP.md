# 2026-09-09 项目整理与恢复

## 已执行

- 49项旧SA-MoE模型/配置/推理/评估入口、旧辅助头脚本、重复备份和历史文档入口移出活动工作区；恢复根为 `/home/ubuntu/ur3_ft300_retired/20260909/`。
- 清单：本机 `maintenance/release_20260909/removal_manifest.json`、`completed_moves.jsonl`。其中目录是一项，不能把49项写成49个文件。
- 移除默认LeRobot的SA策略注册；PAP兼容旧配置字段及不可变实验快照不动。配置工厂修改前备份在恢复根 `pre_edit/lerobot/`。
- 旧入口文档RESOURCE_REGISTRY移入恢复根 `retired_navigation/`；全流程历史审计移到 `reports/数据训练推理审计_历史快照_20260906.md`。接手文档完整重写，旧全文在 `pre_edit/workspace/handover.md`。
- README展示真实模型视频、实验局限与贡献；新增运行和许可证/来源说明。学习代码以锁定上游+增量发布，不移动当前环境。
- `.gitignore`隔离artifacts、模型、数据、缓存、备份和密钥类文件。取消tokenizer等资源的Git跟踪只影响公开版本，本机资源保留。

## 明确保留

另移出3项已退役的subtask/progress残留：`pap_moe_framework/tests/test_stage_shadow_metrics.py`（引用的实现此前已不存在）、`patches/pap_moe_force_aware_subtask.diff`、`patches/pap_moe_force_head_joint_mode.diff`。恢复位置为恢复根`obsolete_tests_and_patches/`同名文件；不为让旧测试通过而重新引入被取消的阶段/进度头。

补充移出 `scripts/plot_samoe_training.py` 和 `pap_moe_framework/patches/` 的6份旧v6补丁，恢复位置为 `redundant_patches/`；当前源码已由版本化增量提供，不再保留容易重复套用的旧补丁入口。

根目录的 `run_baseline_gazebo_eval.sh`、`run_baseline_v9_gazebo_eval.sh` 是旧301/13001场景的重复包装器，移到 `retired_navigation/`；统一从scripts导航进入当前Workspace50流程。

## 发布验证

- 公开上游祖先检出 + 增量应用：已在独立 `/home/ubuntu/ur3_ft300_release_checks/lerobot_upstream_20260909` 执行成功，没有覆盖现有环境。
- 独立源码副本的PAP/RTC：347项测试通过；工作区数据/常驻协议：74项通过。
- 发现一条仍断言默认夹爪历史二值化的旧测试，已按当前连续rad合同修正，并保留显式旧模式兼容测试。运行逻辑未改变。
- 运行测试需禁用自动加载ROS的pytest插件，避免Python3.10/3.12混用导致缺失lark；未因此安装或改动当前依赖。
- 发布候选扫描：未发现常见GitHub/HF token或私钥模式，无新增单文件超过10MB，主要文档本地链接存在。模式扫描不是完整历史安全审计。
- 未执行新的训练、全量ROS编译、空白机器端到端部署或闭环收益验证。

Workspace50原始/派生数据、有效训练检查点、失败记录、归一化证据、实际插件和冻结实验快照均保留原路径。`artifacts/`还被大量模型/回放/报告硬编码引用，本次不批量迁移或清空。

这是一轮源代码/文档/公开仓库整理，不是大规模磁盘回收。永久删除0项；可恢复移动不释放占用空间。既有Pi0/ACT/DP和共用控制工具不是旧SA-MoE，不按名字中含pi0就删除。

## 恢复方法

先查manifest的source/destination，在隔离副本检查所需旧项，确认目标不存在后单项移回；不要批量覆盖新文档或源码。旧实验的source_manifest指向已移走文件时，结合此映射核验原件，不改写原始哈希。

发布增量准备测试第一次因误把未推送的本地LeRobot HEAD当作官方提交而失败；失败副本保留在 `/home/ubuntu/ur3_ft300_release_checks/lerobot_20260909`。随后改为公开祖先 `ba27aab79c731a6b503b2dbdd4c601e78e285048` 并纳入该本地提交改动，避免不可下载的上游锁定。
