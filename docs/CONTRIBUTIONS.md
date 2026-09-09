# 来源、项目贡献与许可证边界

## 上游来源

| 组件 | 上游 / 本地位置 | 本项目工作 |
|---|---|---|
| UR 描述与 Gazebo 框架 | [Universal Robots](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)、`src/ur_description`、`src/ur_simulation_gz` | 组合传感器/夹爪描述、场景、相机和任务控制，非从零机器人驱动 |
| Robotiq 夹爪 | [PickNik ROS 2 Robotiq](https://github.com/PickNikRobotics/ros2_robotiq_gripper) | 仿真联动、碰撞/惯性、控制器参数联调 |
| FT300 | `src/rq_fts_ros2_driver`，其 README 声明沿用 [ROS-Industrial Robotiq](https://github.com/ros-industrial/robotiq) | 传感器集成、力数据采集/滤波/偏置与多时间窗，不宣称原创底层驱动 |
| 相机 / 串口 | [RealSense ROS](https://github.com/IntelRealSense/realsense-ros)、[wjwwood/serial](https://github.com/wjwwood/serial) | 相机朝向、观测映射与同步集成；原库许可证保留 |
| 控制 / 规划 | [ros-controls](https://github.com/ros-controls/gz_ros2_control)、[MoveIt 2](https://github.com/moveit/moveit2) | 启动编排、Servo 键鼠映射、夹爪新读数计数候选修复与诊断 |
| 学习基础设施 | [LeRobot](https://github.com/huggingface/lerobot) | 数据字段/统计审计、采样与优化器分组扩展、PAP 策略注册、训练/推理合同适配 |
| VLA / 基线 | [OpenPI](https://github.com/Physical-Intelligence/openpi) 与 LeRobot 的 Pi0.5、ACT、Diffusion Policy | 使用其实现和预训练能力；本项目开发物理专家、条件接口和验证流程，不宣称发明基础模型 |

## 可展示的项目增量

- `pap_moe_framework/scripts/pap_moe_peg_in_hole_record.py`、转换/统计/路由工具：多模态样本与可追溯数据合同。
- `src/ur_simulation_gz/ur_simulation_gz/scripts/pap_moe_*`：键鼠界面、运动映射和状态诊断。
- `scripts/resident_policy/`、`scripts/deployment_validation_20260909/`：会话隔离、数值回放、控制器/夹爪/释放检查。
- `learning/lerobot_overlay/src/lerobot/policies/pap_moe/`：PAP 模型、PhysicsGate、四专家与训练接口；详细来源由 manifest 标记。


## 许可证

保留各组件原有 LICENSE 和文件版权头。LeRobot 增量涉及的上游文件按其 Apache-2.0 条款分发，见 `learning/LICENSE.lerobot`；其他 ROS 包按各自许可证分发。根仓库不擅自给所有第三方资产重新授权。预训练权重、tokenizer、数据和演示资产的许可与代码许可分开；未在仓库提供的资源不代表可自由再分发。
