# 运行与资源说明

## 1. 环境与公开资源边界

本机 Ubuntu 22.04 / ROS 2 Humble / Gazebo Fortress，学习环境为Python3.12.13；版本记录见 `learning/environment.json`。ROS Python3.10 与学习环境隔离，通过文件快照/请求协议通信，不混装rclpy到学习环境。

公开代码不含 Workspace50 原始数据、转换后数据、微调检查点、预训练模型与tokenizer。获取权重需遵守 [LeRobot Pi0.5](https://huggingface.co/docs/lerobot/main/en/pi05) 和相应模型仓库的许可/访问条件。没有这些资源时可构建仿真、读代码和跑轻量测试，不能直接复现表格中的模型结果。

本次没有在一台空白机器重新执行全部ROS/训练流程；不要把本机通过的命令说成一键无条件可复现。许多研究脚本保留原绝对路径 `/home/ubuntu/ur3_ft300_ws`、`/home/ubuntu/lerobot` 和conda环境路径，换机器须显式调整；不能用修改环境变量假装所有硬编码都已支持。

## 2. 仿真与相机

先安装ROS2 Humble、Gazebo Fortress/ros_gz、MoveIt2、ros2_control/controllers、colcon、rosdep等。

```bash
cd /path/to/ur3_ft300_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

需要MoveIt/RViz时分别另开终端，source同一环境：

```bash
ros2 launch ur3_ft300_moveit_config move_group.launch.py
ros2 launch ur3_ft300_moveit_config moveit_rviz.launch.py
```

`patches/gz_ros2_control_source.json`锁定夹爪修复候选上游，补丁在同目录；它没有被本次整理自动安装为默认插件。正式对比应使用接手指南中的隔离修复版本并核对运行时映射，不混用apt默认版与候选版。

## 3. 采集、转换与检查

脚本示教入口（新数据目录；不是模型推理）：

```bash
PAP_MOE_PILOT_ROOT=/path/to/new_raw \
  bash scripts/run_pap_moe_v9_exactfit_collection.sh true 1
```

这是既有单条采集入口，**不保证一次调用自动生成规范50条/5位置×10风格**。批量分组与重采编号按manifest管理，不能覆盖现有有效数据。

键鼠入口：`bash pap_moe_framework/scripts/start_teleop_workspace_50.sh /path/to/new_teleop`。此流程的自动home/成功验收仍待专项验证，见接手文档，不作为新正式数据的免检入口。

在学习环境中执行只读样本验收：

```bash
python pap_moe_framework/scripts/validate_multimodel_data_contract.py /path/to/new_raw
python pap_moe_framework/scripts/pap_moe_peg_in_hole_convert_to_lerobot.py \
  --input /path/to/new_raw --validate_only
```

转换full/baseline视图，并在**新视图**计算全局统计：

```bash
python pap_moe_framework/scripts/pap_moe_peg_in_hole_convert_to_lerobot.py \
  --input /path/to/new_raw --output_dir /path/to/new_lerobot_full \
  --repo_id local/ur3_full --model_view full --fps 10 --source_fps 10
python pap_moe_framework/scripts/recompute_lerobot_numeric_stats.py \
  /path/to/new_lerobot_full --output /path/to/new_lerobot_full_global_stats
```

不要重算并覆盖旧检查点对应的统计。baseline视图用同源数据、`--model_view baseline`另行生成。4专家路由/力/历史保留在full视图；DP/ACT/Pi0.5仍使用兼容的图像/状态/动作字段。

## 4. 训练与推理入口

`scripts/prepare_learning_source.py`准备锁定上游+PAP增量源码，不自动启动训练或替换现有环境。之后参考本机入口：

- Pi0.5：`scripts/run_pi05_workspace50_official_lowmem_expert_only_30k.sh`，冻结VLM、全量动作专家，启动前要求正确统计收据。
- ACT/DP：`scripts/run_workspace50_corrected_act.sh` / `run_workspace50_corrected_diffusion.sh`。
- PAP：`scripts/run_pap_moe_vnext_stage_train.sh` 和 `scripts/train_pap_checked.py`；阶段/条件来源须显式配置。新B/GT/Pred实验需按接手指南的冻结代码运行，不能用默认旧入口代替。

现有本机Gazebo模型入口：

```bash
# CHECKPOINT必须是实际存在的兼容pretrained_model目录，包含config/processor/weights
bash scripts/run_workspace50_lerobot_policy_gazebo.sh pi05 "$CHECKPOINT" 1 true
# PAP换为pap_moe；这是默认冷加载入口，不等价于冻结修复插件的正式成绩。
```

冻结版常驻流程及启动合同见 `scripts/resident_policy/` 和接手文档。清理后旧快照继续留在本机artifacts；公开仓库不提供其私有权重，默认启动也不会自动生成评估表。

## 5. 验证入口

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest pap_moe_framework/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest scripts/resident_policy -q
```

在准备好的LeRobot增量源码中：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/policies/pap_moe tests/policies/rtc -q`。关闭自动插件加载是避免ROS Python3.10的launch_pytest污染Python3.12学习环境；不是跳过上述测试。部分测试需可选依赖；单元测试通过不代替Gazebo与真实硬件安全验收。
