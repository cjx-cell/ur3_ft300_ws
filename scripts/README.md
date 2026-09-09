# 脚本导航

不要按文件名中的版本号自动选择训练起点。当前实验状态、冻结源码和检查点以 [接手文档](../docs/PAP_MOE新对话接手指南.md) 为准。

| 用途 | 入口 |
|---|---|
| 准备独立学习源码副本 | `prepare_learning_source.py` |
| Pi0.5正确统计基线训练 | `run_pi05_workspace50_official_lowmem_expert_only_30k.sh` |
| ACT / DP | `run_workspace50_corrected_act.sh` / `run_workspace50_corrected_diffusion.sh` |
| PAP前置融合训练配置 | `run_pap_moe_vnext_stage_train.sh` / `train_pap_checked.py` |
| 本机冷加载仿真入口 | `run_workspace50_lerobot_policy_gazebo.sh` |
| 常驻模型 / 请求协议 / 回放 | `resident_policy/` |
| 数据与全帧统计审计 | `audit_workspace50_pipeline.py`、`deployment_validation_20260909/audit_numeric_dataset_chain.py` |
| 实际夹爪 / 执行目标 / 释放质量 | `deployment_validation_20260909/` |
| 完整采集与视频 | `run_pap_moe_v9_exactfit_collection.sh`；数据工具在`../pap_moe_framework/scripts/` |

其他诊断、恢复与带日期的实验脚本保留用于解释已有结果，不等于当前推荐流程。不要直接运行含旧数据路径的历史调度器；也不要将反事实路由/示范回放分支计作原生模型分数。旧SA-MoE和已取消辅助头的独立入口已移除。
