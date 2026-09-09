# Rollout 失败恢复数据（独立链路）

本目录只定义 `模型 rollout → 失败触发 → 专家原地接管 → 恢复成功` 数据。
它不读取、不覆盖、也不转换旧的完整专家轨迹目录。

旧 `pap_moe_peg_in_hole_record.py` 中的人工 XY 偏移恢复是原设计的一部分，
必须原样保留。它采集的是“受控人工偏移状态 → 专家恢复”的覆盖数据；本目录
新增的是“模型真实 rollout 状态 → 专家接管”的纠偏数据。二者都有效，但数据
来源和训练用途不同，不得合并成同一种 `trajectory_scope`，也不得用新链路替换旧链路。

固定边界：

- 新采集：`trajectory_scope=rollout_failure_recovery_v3_multitask_full_modalities`
- 历史 v2 全模态 episode 保持只读，通过离线迁移/重标注继续使用，不要求重采
- 独立输出根目录：`pap_moe_framework/datasets/raw_rollout_recovery_v2_full_modalities/`
- 每帧必须原生包含双相机、当前关节、`10x7` 关节历史、FT300 当前值、
  `64x6` 快窗、`50x6` 慢窗、视觉质量和全 1 的模态有效位
- 每帧还必须包含 policy/expert/executed/requested/controller 动作、接管标记、
  语义子任务、通用 Skill-Progress 与四专家物理阶段监督
- 通用 Skill-Progress 采用跨任务共享局部阶段：`enter/approach/align/interact/`
  `stabilize/verify/exit/recover`，同时保存 `[0,1]` 连续进度、阶段切换就绪度、
  标签有效位、来源和置信度；禁止再把它实现成只适用于抓取的 progress head
- policy roll-in 帧若没有可靠阶段标注，必须 `skill_progress_valid=false`，训练时用
  mask 排除；专家 chunk 的阶段标注通过不可覆盖的 sequence 文件与实际动作对齐
- `stage` 是 Physics Gate/物理专家监督，由 FT300 接触变化等可观测量生成，不能
  直接复制 semantic subtask，也不能把 Gazebo peg/hole 真值作为模型输入
- 历史窗口从 ROS 原生回调缓存因果重采样并保存来源时间戳；出现未来样本直接判废
- 开始收录前必须积满 1 s 关节历史、0.64 s FT300 快窗和 5 s FT300 慢窗；
  schema 会核对时间覆盖，不允许用重复当前值伪造历史窗
- 任一必需模态缺失、无效或相机质量门禁失败，整条 episode 判废，不允许补零入库
- 动作必须拆分为 `policy_action`、`expert_action`、`executed_action`
- `intervention_mask` 之前只能执行 policy 动作，之后只能执行 expert 动作
- 只允许一次连续的 policy → expert 接管，不能来回切换
- 失败局部恢复episode只保留接管前1秒policy上下文和接管后的全部expert动作；更早的完整rollout
  由Gazebo artifact/video单独审计，不作为恢复训练target重复入库
- 专家动作必须来自控制器明确发布的命令，禁止用下一帧实测关节状态代替
- `grasp_lift` 成功样本必须有位姿确认的物理抓取，以及接管后至少 5 cm 的抬升
- `align_precontact/contact/insertion` 使用各自的 XY、FT300 接触和最终插入几何门禁，
  不错误要求已经抓住 peg 的 contact recovery 再向上抬升 5 cm
- 相机时间戳与状态采样的最大偏差默认不超过 150 ms
- `peg_attached` 字段只表示 peg 与双指夹爪的位姿关系确认了物理抓持，绝不表示
  DetachableJoint 或 `/peg/attach`；正式运行仍必须保持 `attached=False`

验证命令：

```bash
/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  pap_moe_framework/scripts/validate_rollout_recovery.py \
  pap_moe_framework/datasets/raw_rollout_recovery_v2_full_modalities
```

在 baseline 和 MoveIt 专家尚未发布三类显式动作事件前，采集器不得保存训练样本。

## 当前独立链路

正式 baseline runner 默认关闭 recovery。采集时使用一个全新的 per-run session：

```bash
PI05_ROLLOUT_RECOVERY_SESSION_DIR=auto \
  ./scripts/run_pi05_v9_absolute_gazebo_eval.sh false 13001
```

runner 会在本次 Gazebo artifact 下建立 `rollout_recovery_session/`。ROS 执行侧在
`action_events/` 原子发布每个 action chunk，事件同时保存：

- 来源模式和单调序号；
- 原始请求的 semantic action；
- ROS 安全限幅后的 semantic `policy_action` 或 `expert_action`；
- 实际选中的 `executed_action`；
- 发送给 FollowJointTrajectory 的物理夹爪/关节目标。

独立 recorder 直接订阅关节状态、双相机和 FT300，不调用旧完整轨迹 recorder：

```bash
/usr/bin/python3 pap_moe_framework/scripts/record_rollout_recovery.py \
  --session-dir <artifact>/rollout_recovery_session \
  --outcome-file <artifact>/rollout_recovery_outcome.json \
  --output pap_moe_framework/datasets/raw_rollout_recovery_v2_full_modalities/<episode>/data.npz \
  --source-policy-checkpoint <checkpoint> \
  --episode-id <episode> \
  --recovery-phase contact
```

专家控制器每次发布明确的 7D semantic action chunk（夹爪命令必须为全开0.0或全闭0.8；
实际接触夹持角由物体决定），然后操作员
发出一次性接管。接管在本 episode 内不可撤销：

```bash
/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  pap_moe_framework/scripts/control_rollout_recovery.py publish-expert \
  --session-dir <session> --chunk <expert_chunk.npy> \
  --sequence 0 --publisher moveit_recovery

/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  pap_moe_framework/scripts/control_rollout_recovery.py takeover \
  --session-dir <session> --trigger 'contact alignment drift' --requester operator
```

后续 expert chunk 的 `--sequence` 必须严格递增。恢复结束后显式写 outcome；没有 outcome
或 schema 校验失败时 recorder 不会生成正式 `data.npz`：

```bash
/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  pap_moe_framework/scripts/control_rollout_recovery.py outcome \
  --outcome-file <artifact>/rollout_recovery_outcome.json \
  --value success --operator operator
```

这套 IPC 只负责忠实记录和选择明确来源的动作，不根据 peg/hole 在线真值生成纠偏动作。
peg/hole 位姿只进入数据质检和成功门禁。

ROS-side 只有在对应 FollowJointTrajectory 返回成功后，才会把该 expert source sequence
原子写入 `output/expert_execution_ack.json`。自动专家每次只允许一个未确认 chunk：必须等到
完全相同 sequence 的执行完成 ACK，才能读取新的机器人状态、重新求 IK 并发布下一 chunk。
ACK 不表示动作刚被接收或派发，而表示整段轨迹已经执行结束；超时必须将本次 recovery 标为
失败，禁止继续覆盖 `expert_chunk.npz`。

每个 policy/expert action event 还对应一个不可覆盖的
`completion_events/completion_<event sequence>.json`。recorder 不使用理想的 0.1 s 墙钟
推算在途动作，而是在 trajectory 成功结束后，用真实 dispatch→completion 墙钟区间重建该
chunk 的采样时刻。这可处理 Gazebo 低于实时速度的情况，避免把最后的 FT300 接触帧错误地
判为与动作不同步。outcome 到达后 recorder 保留 1 s 收尾窗口，再冻结并验证 episode。

旧 `raw_rollout_recovery_v1` 仅用于复现实验和历史 D1 读取。验证旧数据时必须显式传入
`--allow-legacy-missing-modalities`；该选项严禁用于 D2 或任何后续新数据入库。

正式 raw schema 默认拒绝 `>80 N` 连续 3 帧及以上的持续过载；单帧 DART 瞬时冲击仍允许
保留并在质检报告中披露。专家下降步长必须在接近孔口后收小，不能依靠 evaluator 结束仿真来
停止。采 insertion episode 时应设置 `--success-peg-z-m`，让专家先依据几何成功写 outcome，
再由 Gazebo 的连续成功计数结束 episode。
