# pi0 UR3 Pick-and-Place 全量微调分析报告

## 项目背景

在 UR3 + FT300 力传感器 + Robotiq 2F85 夹爪 + Gazebo Fortress 仿真环境中，使用 pi0 (PaliGemma 2B + Action Expert 300M, 4B 参数) 完成"抓取红色方块放入蓝色碗中"的全量微调。

## 数据和模型

| 项目 | 详情 |
|------|------|
| 数据集 | 49 episodes / 167,962 帧 / 50fps |
| 观测 | camera0 (腕部), camera1 (全局), 7维关节状态 |
| 动作 | 7维绝对关节位置 (6臂+1夹爪) |
| 任务指令 | "pick up the red cube and place it into the bowl" |
| 预训练模型 | pi0_libero_base (6.6GB, LIBERO任务预训练) |
| 硬件 | NVIDIA A100-SXM4-80GB, CUDA 13.0 |

## 训练历史

### V1 (首次尝试)

| 参数 | 值 |
|------|-----|
| Learning Rate | 1e-5 → 2.5e-6 (cosine decay, 30K步衰减到底) |
| Steps | 84,000 |
| Batch Size | 2 |
| 总时间 | ~9h |
| 最终 Loss | 0.012 |
| 推理 MAE | 0.37 rad |

**致命Bug**: `modeling_pi0.py` 中 `_fix_pytorch_state_dict_keys` 未处理 vision encoder 的 `.vision_model.` 层级差异。预训练权重中 437 个 vision key 无法匹配到模型结构，`load_state_dict(strict=False)` 静默跳过。视觉编码器以随机初始化训练，仅在 1 epoch 内无法学到有意义的视觉特征。

### V2 (修复后)

| 参数 | 值 |
|------|-----|
| Learning Rate | 2e-5 → 2.5e-6 (cosine decay, 80K步衰减) |
| Steps | 84,000 |
| Batch Size | 2 |
| 总时间 | ~8.5h |
| 最终 Loss | 0.006 |
| 推理 MAE | 0.34 rad |

**修复内容**: 在 `modeling_pi0.py:1099-1108` 中添加了 key remapping:
- `.vision_model.` → `.` (437 个 vision keys)
- 无 `model.` 前缀 → 添加 `model.` 前缀

验证结果: Vision missing: 0/437, 视觉权重来自预训练 SigLIP。

### V1 vs V2 对比

| 指标 | V1 | V2 |
|------|-----|-----|
| Vision权重 | ❌ 随机初始化 | ✅ 预训练 |
| 最终Loss | 0.012 | 0.006 |
| 推理MAE | 0.37 rad | 0.34 rad |
| 改进幅度 | — | 8.1% |
| Gazebo表现 | 随机游走 | 随机游走 |

## Gazebo 实测表现

1. **V2模型**: 机械臂从初始位姿开始移动，但无明显目标方向，在"上方"区域徘徊，无法靠近方块。
2. **基础模型 (pi0_libero_base)**: 行为与V2类似，漫无目的游走。
3. **夹爪状态机**: 原始脚本的状态机 (CLOSING/OPENING 2秒冻结) 在测试中被禁用后，模型输出仍然无效。
4. **累积漂移**: 推理时每个动作偏差 ~0.3rad，误差累积使状态快速偏离训练分布，形成恶性循环。

## 根因分析

### 1. 数据量与模型规模不匹配

| 项目 | 实际配置 | 推荐配置 |
|------|---------|---------|
| Episodes | 49 | 100+ |
| 训练 Epoch | 1 | 3-5 |
| Batch Size | 2 | 4-8 (显存允许) |
| 训练步数 | 84K | 200K+ |

1 epoch + 4B参数 = 优化远远不够。

### 2. Diffusion Loss 不可靠

pi0 训练的 loss 是**噪声预测 MSE** (diffusion模型去噪目标)，不是动作空间误差:
- Loss 从 1.5 降到 0.006 看起来收敛良好
- 但动作 MAE 高达 0.34 rad，模型无法产生有意义的轨迹

**低 loss ≠ 学到了任务**

### 3. 逐帧动作变化极小

数据集特点: `action[t] ≈ state[t]` (50fps下帧间位移仅 0.0007 rad)

| 方法 | MAE |
|------|-----|
| 零增量 (action=state) | 0.0007 rad |
| 均值预测 | 0.2374 rad |
| V2模型 | 0.3363 rad |

模型比"不动"差了 480 倍。扩散模型从每一帧都近乎相同的噪声分布中学习，难以形成连贯的动作轨迹。

### 4. 目标难以从MAE评估

逐帧MAE是误导性指标——"什么都不做"反而是最优策略。真实评估应该在Gazebo仿真中看任务完成率。

## 建议改进方向

1. **增加训练数据** — 收集更多 demonstrations (100+ episodes)
2. **增加训练轮次** — 3-5 epochs，让模型充分接触数据
3. **增大 Batch Size** — 尝试梯度累积或使用多GPU
4. **尝试 LoRA** — 只微调少量参数可能更稳定
5. **调整动作表示** — 考虑使用相对动作 (delta) 而不是绝对位置，增加帧间变化
6. **降低帧率** — 将50fps降采样到10fps，增大帧间动作差异
7. **评估改为任务完成率** — 直接在Gazebo中跑rollout统计成功率

## 相关文件

| 文件 | 说明 |
|------|------|
| `V1_POSTMORTEM.md` | V1 详细事后分析 |
| `modeling_pi0.py` | 含 `_fix_pytorch_state_dict_keys` 修复 |
| `outputs/train/ur3_pi0_full_v2/` | V2 训练结果 |
| `src/ur_simulation_gz/.../ur3_pi0_inference.py` | 推理脚本 |
| `src/ur_simulation_gz/.../ur3_pi0_ros_side.py` | ROS 桥接脚本 |

## 结论

全量微调 pi0 (4B参数) 在 49 episodes / 1 epoch 配置下无法学会 UR3 Pick-and-Place 任务。主要瓶颈是数据量不足和训练轮次不够，而非代码bug (V1的vision权重bug已在V2修复)。建议至少收集 100+ demonstrations 并训练 3-5 epochs。
