# 模型常驻优化：独立候选实现（2026-09-08）

最新：resident_v1在第15次尝试遇到读取观测时ID变化，保护退出。新增`observation_snapshot.py`只在推理前有界重读稳定观测，完整paired ID仍用于回复/推理后校验；ctime单独变化不作为新内容触发。稳定损坏观测与持续变化仍失败关闭。22项测试通过。`formal_batch.py --continue-from`校验权重、旧源快照及原始结果，继承所有有效成绩，不选择成功/覆盖失败；新目录`paired_comparison_20260908_resident_v2_stable_read`补缺并续跑。原无效记录保留且不计模型失败。该续跑含两版读取逻辑，报告须标注来源。

目的：同一检查点跨 episode 仅加载一次权重。Gazebo 仍逐条冷启动。
**最新：在线接入及工程验收已完成，已启用独立的常驻正式批；不宣称成功率严格等价。**

正式批`artifacts/paired_comparison_20260908_resident_v1/`使用只读code快照，三模型统一运行模式，旧冷启动成绩不混入。入口`formal_batch.py`要求显式提供审阅过的三模型验收记录。Pi0.5夹爪全50步限幅已对齐原main，PAP仍仅限幅前10步，RTC归一化leftover不变；11项轻量测试通过。Pi0.5原main相同观测21块最大5.96e-8rad，S3 94块差异0，S4前2块差异0；每组均完成两条Gazebo重启且模型常驻的试验。有限验收不构成成功概率等价证明。以下为开发过程记录，旧“未接入”状态不再代表现状。

最新调度更正：用户要求当前单条结束即验证。`after_trial.py`替代原整批等待程序；只暂停父调度shell，现有trial独立自然结束并清理，然后跑离线验证。新状态目录`artifacts/resident_policy_after_trial_20260908_v1/`。不自动恢复正式批或写入其结果表；当前条的结果归并会在原shell恢复时执行。

## 已实现

- `session.py`：串行 begin/infer/end；服务端生成 episode token；严格递增请求序号；旧会话、重复和乱序请求拒收。部分推理失败立即废弃会话。
- `backend.py`：Pi0.5 / PAP-MoE 单检查点常驻；每条重新创建 checkpoint 自带 processors，清空 policy 队列、PAP 视觉记忆及 RTC leftover，重建 RTC，再复现原脚本预热和 seed。保留预热产生的视觉历史，以匹配当前冷启动行为。
- 固定原有 50/10、RTC EXP / max10 / arm-only、连续夹爪 0..0.8；调用原脚本的输入转换和动作反归一化函数。不改控制器、动作限幅、场景、任务成功标准或模型结构。
- `worker.py`：仅支持保存观测 NPZ 的离线 JSON-lines 服务，未连接机械臂或 Gazebo。
- `validate.py`：新进程 A 对照常驻 A→B→A；至少两个动作块，覆盖第二块 RTC；比较归一化完整输出、50 步物理动作、10 步执行前缀、PAP 路由。PAP 另对照原正式脚本保存的动作与预处理结果，防止新实现自证。
- `after_batch.py`：等待当前冻结 45 次评估正常完成并释放 GPU，再顺序验证三个检查点；源代码或权重记录变化、评估无效、GPU 被占用或数值不一致均停止，不自动重试。

## 验收界限

会话逻辑测试通过不等于真实模型输出等价。数值阈值：归一化动作 1e-5、物理动作 1e-4 rad、路由 1e-6（均为最大绝对差）。仅接受相同 shape 和有限值；报告实际误差，不把低于阈值表述为严格逐位一致。

当前 baseline 正式批次没有保存逐块模型输入，因此 baseline 可先在相同 PAP 保存传感器输入上验证冷/热进程一致，但尚缺原 baseline main 的独立数值参考；不能据此声称其在线路径完全验证。

数值验收之后仍需：实现复用原校验的在线观测/动作适配器，保证 paired request ID 与 episode token 双重隔离、就绪时刻一致；同一场景/种子做冷/热闭环配对，检查实际轨迹、时序、传感器历史、成功判据及耗时。通过前不启用正式常驻评估，不混合两版结果。所有后续模型对比统一使用同一已验收模式。

运行轻量测试：

```bash
/usr/bin/python3 -m unittest discover -s /home/ubuntu/ur3_ft300_ws/scripts/resident_policy -p test_session.py -v
```

本目录不在当前冻结批次的运行源码集合内；原训练、推理、启动器均未修改。
