# LeRobot 能否剥离：结论与发布方式

## 结论

可以把 PAP-MoE **作为独立策略包**组织，但不应删除 LeRobot 基础设施后从零重写训练/数据系统。本机除了新增策略，还修改了 train config、训练器、采样、优化器分组、processor、Pi0.5 与 RTC；只复制模型文件会漏依赖。当前不是低风险的简单搬文件任务，暂不迁移运行环境。

官方已有先例和推荐路径：[Adding a Policy](https://huggingface.co/docs/lerobot/main/en/bring_your_own_policies) 支持 `lerobot_policy_*` 的 out-of-tree 插件。它适用于稳定的策略/配置/processor 扩展，但不能自动承载我们对训练器和底层 Pi0.5 的全部修改。必须核对锁定版本是否具备相应插件 API，而不是跟随最新 main 升级。

## 本次采用：上游版本锁定 + 项目源码增量

`learning/upstream.json` 锁定来源和提交，`learning/lerobot_overlay/` 只保存相对上游改变的文件及 PAP 新模块/测试，不把整个 LeRobot 上游仓库复制进 UR 工作区。不改变当前 `/home/ubuntu/lerobot` 安装路径、权重或历史冻结快照。

该增量的 PAP/Pi0.5/RTC 等研究实现来自 2026-09-09 路由适配冻结版本；旧 SA-MoE 注册在公开增量中移除。manifest 记录每个文件来源和 SHA256。准备独立源码副本：

```bash
# 目标必须不存在；不会覆盖当前 ~/lerobot 或自动安装依赖
python3 scripts/prepare_learning_source.py --destination /path/to/new/lerobot-pap
# 在独立学习环境内，核对 learning/environment.json 后再安装：
python -m pip install -e /path/to/new/lerobot-pap
```


## 真正插件化的后续工作

1. 将 PAP 配置、模型、processor 移到独立包并保留保存检查点的注册名兼容。
2. 把采样和优化器参数组扩展转成显式可调用入口，避免依赖训练器内部补丁。
3. 将 Pi0.5/RTC 必要变化拆成小补丁或组合适配器，而非复制整份模型实现。
4. 在新环境逐项验收：数据逐值/统计一致、checkpoint strict load、同输入完整 Flow、RTC 多块重置、采集/转换和配对 Gazebo。

工作量按这些验收项计，不在未完成依赖切分前承诺几小时完成。现阶段研究尚有接口/部署问题，迁移与模型实验应分开做，避免同时改变框架与执行合同。
