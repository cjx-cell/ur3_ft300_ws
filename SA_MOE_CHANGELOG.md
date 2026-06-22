# SA-MOE Changelog

## v8 — Residual Addition (2026-06-22 最终版)

### 唯一核心改动
**数据流从 condition prepend 改为 residual addition**

```
v7: SA-MOE → condition tokens → prepend suffix → Gemma → action
                                    ↑ frozen backbone 稀释信号
v8: SA-MOE → residual: α * expert_out[:, -50:] + suffix_out[:, -50:] → action
                                    ↑ 直接加到 action decoder 输出，不经过 backbone
```

证据：[pi0_force.py:286-287](file:///home/ubuntu/ur3_ft300_ws/ForceVLA/src/openpi/models/pi0_force.py#L286-L287)
```python
limoe_out = self.limoe(jnp.concatenate([prefix_out, force_tokens], axis=1))
v_t = self.action_out_proj(limoe_out[:, -50:] + suffix_out[:, -50:])
```

### SA-MOE 三个组件全部 ACTIVE（保留）
1. **StageGate** (attention pool) → 选择 expert（per-sample routing）
2. **TransformerExpert** (2-layer self-attn) → 阶段特化处理
3. **ModalGate** → α 调制 residual 强度（高 α=视觉阶段加更多 SA-MOE，低 α=力阶段靠 backbone）

### 改动列表
- `_forward_sa_moe`：输出 expert_out [B, N+1, D]（不再是 condition tokens）
- `forward`：移除 condition token 注入，改为 `α * expert_out[:, -50:] + suffix_out[:, -50:]`
- `denoise_step`：接收 sa_residual 参数，在 suffix_out 之后加上
- 删除：condition_proj, condition_tokens 相关代码
- 保留：StageGate, ExpertLibrary, ModalGate, TransformerEncoder（全部 active）

### 参数
172M 可训练

---

## v7 — ForceVLA-style data flow (2026-06-22)

### 架构变更
1. **E_VL 全保留**：不再 mean-pooling，~560 个 VLM token 全部保留
2. **多层 Transformer 融合**：2 层 self-attention 处理所有 [E_VL | force_token]
3. **StageGate attention pool**：4 个可学习 query cross-attend 替代 mean pool
4. **Transformer Expert**：2 层 self-attention 替代 3 层 MLP
5. **多 token condition**：4 个 condition token 注入 suffix（替代 1 个）
6. **删除 force_out_proj**：力只作为输入，不预测
7. **删除 ModalityWeightedFusion**：融合在 Transformer 内部完成
8. **真实 stage 标签**：用数据里的 `observation.stage`（C++ 打的），不再用 `generate_stage_labels` 伪标签
9. **减少损失函数**：5 个→2 个（flow-matching + stage_loss，balance 可选）

### 遇到的问题及修复
1. **dtype 不匹配**：PaliGemma 输出 float32，SA-MOE 是 bf16。修复：`_forward_sa_moe` 入口 + `forward()` 中 E_VL/force 统一 cast 到 target_dtype
2. **LeRobot 归一化 stage 标签**：`observation.stage` 被 dataset 当成 float32 归一化。mean=1.95, std=1.64。0→-1.19, 4→1.25。修复：`_extract_stage_labels` 中反向计算 `stage_raw = norm * std + mean`，round + clamp 到 0-4
3. **报错 Value -1**：归一化后的 stage 0 约等于 -1.19，被 clamp 到 -1。修复同 #2
4. **训练速度慢**：160M 可训练参数，~1.7 step/s（预计 ~8h/50K steps）

### 配置文件变更
- 新增：`transformer_encoder_layers=2`, `expert_transformer_layers=2`, `num_condition_tokens=4`
- 删除：`force_loss_weight`, `alpha_aux_loss_weight`, `fusion_dim`, `stage_label_start_step`

---

## v1-v6 — Original SA-MOE (2026-06-20 ~ 2026-06-22)

### 迭代历史
- **v1**：绝对动作，100K steps，loss 2.2→1.4。问题：action=state 导致模型学不到动作
- **v2**：Delta 动作，5K steps test。loss 1.99→1.62
- **v3**：Delta 动作，100K from v2 checkpoint。loss flat at 1.60
- **v4**：Delta + class_weights (0.5,1,6,6,0.5)，50K steps

### 核心问题（已诊断）
1. **伪标签问题**：`generate_stage_labels()` 用启发式猜 stage，没用 C++ 真实标签
2. **E_VL mean pooling 丢信息**：560 个 token → 1 个向量
3. **Expert 太简单**：3 层 MLP 不够学 5 种操作策略
4. **力融合太浅**：只在池化后的 2 层 Transformer 交互
5. **力预测无用**：`force_out_proj` 推理时从不使用
6. **StageGate 二分类**：class imbalance 导致只预测 approach/confirm

### 训练记录
| Date | Model | Steps | Loss | Stage Acc | Notes |
|------|-------|-------|------|-----------|-------|
| 0620 | ur3_sa_moe_v1 | 100K | 2.2→1.4 | n/a | 绝对动作 |
| 0620 | ur3_sa_moe_delta | 5K | 1.99→1.62 | n/a | Delta 动作 |
| 0620 | ur3_sa_moe_delta_full | 100K | 1.60 flat | ~77% (binary) | 从 5K 续训 |
| 0622 | ur3_sa_moe_delta_balanced | 16K | ? | ? | class_weights，未完成 |
