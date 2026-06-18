# SA-MOE (Stage-Aware Mixture of Experts) 集成 Pi0 方案文档

> 状态：代码已实现，工厂已注册，待训练验证
> 目标：精密装配任务（接近→对准→插入→拧紧），双输出（关节动作 + 期望力/力矩）

---

## 1. 方案演进：从计划到实现

### 1.1 计划阶段 (2026-05-22)

计划文件：`/home/ubuntu/.claude/plans/home-virtual-elephant.md`

核心决策：
- 注入方式：**Prefix Token 注入**——SA-MOE 融合特征作为额外 prefix token 插入 Pi0 prefix embeddings 最前面
- 视觉特征：复用冻结的 SigLIP，`GlobalAvgPool` → `[B, 1024]`
- 力觉特征：新增 `ForceEncoder(6 → 1024)`
- 训练策略：冻结 Pi0 全部参数，仅训练 SA-MOE 模块
- 4 阶段门控：接近 → 对准 → 插入 → 拧紧

### 1.2 与原始计划的差异

| 项目 | 计划 | 实现 | 原因 |
|------|------|------|------|
| 专家数 | 4 | 4 | 一致 |
| 专家输入 | `[Zv, Zf, Zs, Zt]` 拼接 (4×1024=4096D) | `[Zv, Zf, Zs, Zt]` 拼接 (4096D) | 一致，全模态上下文 |
| ModalGate 输入 | `stage_probs + Z_expert` | `stage_probs + Z_expert + expert_idx` (加 stage prior bias) | 增强阶段先验注入 |
| Alpha smoothing | 一阶低通 (γ=0.85) | EMA 在 `select_action` 中 | 推理平稳性 |
| 损失函数 | L_action + λ1·L_stage + λ2·L_balance + λ3·L_force | 同 + L_alpha_aux | 加了 alpha 辅助损失引导模态权重 |
| 阶段标签 | 伪标签（夹爪开度+力幅值） | `stage_loss=0`（暂未监督） | 先无监督运行，后续加标签 |
| Pi0 参数 | 全部冻结 | 全部冻结 (`requires_grad=False`) | 一致 |
| LR | 1e-4 | 1e-4 (via `sa_moe_optimizer_lr`) | 一致 |
| fusion 方式 | αv·Zv + αf·Zf 外部融合 | 专家内部融合 + 外部 α/β 用于监控 | 专家学到更丰富的特征交互 |
| 特征维度统一 | 隐式假设同维度 | 显式 `vis_proj`, `state_to_D`, `text_to_D` | ViT/SigLIP 实际维度可能与 config 不同 |

### 1.3 旧原型 vs 新实现

旧原型 (`pi0/sa_moe_pi0.py`, `pi0/sa_moe_modules.py`)：
- 5 专家（多了"校验"阶段）
- `AtomicSkillExpert` 内部做 α/β 加权融合
- 调用不存在的 `forward_vlm()` / `forward_dit_and_decoder()` 方法
- 无法实际运行

新实现 (`sa_moe_pi0/` 目录)：
- 4 专家，匹配 4 装配阶段
- Expert 处理完整 `[Zv, Zf, Zs, Zt]` 拼接
- ModalGate 独立于 Expert，输出可监控的 α/β
- 完整集成 Pi0 的实际前向流程 (`embed_prefix` → `embed_suffix` → `paligemma_with_expert.forward`)
- 支持 training 和 inference（`sample_actions` + flow-matching denoising）

---

## 2. 文件结构

```
lerobot/src/lerobot/policies/
├── pi0/                              # 原生 Pi0（不修改）
│   ├── configuration_pi0.py          # PI0Config
│   ├── modeling_pi0.py               # PI0Pytorch, PI0Policy, PaliGemmaWithExpertModel
│   ├── processor_pi0.py              # make_pi0_pre_post_processors()
│   ├── sa_moe_modules.py             # [旧原型] 不兼容，保留参考
│   ├── sa_moe_pi0.py                 # [旧原型] 不兼容，保留参考
│   └── sa_moe_config.yaml            # [旧原型] yaml 配置
│
├── sa_moe_pi0/                       # ★ 新 SA-MOE 模块（独立子包）
│   ├── __init__.py                   # 导出 SAMoEPi0Config, SAMoEPi0Policy
│   ├── configuration_sa_moe_pi0.py   # SAMoEPi0Config (继承 PI0Config)
│   ├── sa_moe_modules.py             # ForceEncoder, StageGate, ModalGate, ExpertLib, Losses
│   ├── modeling_sa_moe_pi0.py        # SAMoEPi0Model(PI0Pytorch) + SAMoEPi0Policy(PI0Policy)
│   └── processor_sa_moe_pi0.py       # make_sa_moe_pi0_pre_post_processors()
│
├── __init__.py                       # 添加了 SAMoEPi0Config 导出
└── factory.py                        # 3 处注册点：policy_class, config, processors
```

### 2.1 工厂注册（factory.py 修改点）

```python
# 注册点 1: get_policy_class()
elif name == "sa_moe_pi0":
    from .sa_moe_pi0.modeling_sa_moe_pi0 import SAMoEPi0Policy
    return SAMoEPi0Policy

# 注册点 2: make_policy_config()
elif policy_type == "sa_moe_pi0":
    from .sa_moe_pi0.configuration_sa_moe_pi0 import SAMoEPi0Config
    cfg = SAMoEPi0Config.from_pretrained(pretrained_path, **kwargs_overrides)

# 注册点 3: make_pre_post_processors()
elif isinstance(policy_cfg, SAMoEPi0Config):
    from .sa_moe_pi0.processor_sa_moe_pi0 import make_sa_moe_pi0_pre_post_processors
    processors = make_sa_moe_pi0_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)
```

---

## 3. 框架图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SA-MOE Pi0 推理框架                                         │
│                                                                                      │
│  ┌─────────────────────── OBSERVATION INPUTS ────────────────────────────────────┐  │
│  │                                                                                │  │
│  │  Camera0 [B,3,480,640]    Camera1 [B,3,480,640]                               │  │
│  │      │                         │                                               │  │
│  │      ▼                         ▼                                               │  │
│  │  ┌──────────────────────────────────────┐                                      │  │
│  │  │    Frozen SigLIP (embed_image)       │  ← Pi0 原生，不训练                    │  │
│  │  │    → patches [B, N_patches, vis_dim] │                                      │  │
│  │  └────────────┬─────────────────────────┘                                      │  │
│  │               │ GlobalAvgPool (mean over patches)                              │  │
│  │               ▼                                                                │  │
│  │         Zv_raw [B, vis_dim]  ───────── vis_proj ──────►  Zv [B, D]  ──────┐   │  │
│  │                                                                             │   │  │
│  │  Force [B, 6]                                                              │   │  │
│  │      │                                                                      │   │  │
│  │      ▼                                                                      │   │  │
│  │  ┌──────────────────────────────┐                                           │   │  │
│  │  │  ForceEncoder                │                                           │   │  │
│  │  │  6→256→512→1024 (MLP+LN)     │                                           │   │  │
│  │  └────────────┬─────────────────┘                                           │   │  │
│  │               ▼                                                             │   │  │
│  │          Zf [B, D]  ────────────────────────────────────────────────────┐   │   │  │
│  │                                                                          │   │   │  │
│  │  State [B, 7] → pad → state_proj → Zs_raw [B, expert_dim]               │   │   │  │
│  │      │                                                                   │   │   │  │
│  │      └──────────── state_to_D ──────────────►  Zs [B, D]  ──────────┐   │   │   │  │
│  │                                                                       │   │   │   │  │
│  │  Language tokens [B, 48] → embed_language_tokens →                    │   │   │   │  │
│  │      │                                                                │   │   │   │  │
│  │      └──────────── text_to_D ──────────────►  Zt [B, D]  ────────┐   │   │   │   │  │
│  │                                                                     │   │   │   │   │  │
│  └─────────────────────────────────────────────────────────────────────│───│───│───│───│──┘  │
│                                                                         │   │   │   │   │     │
│  ┌────────────────────── SA-MOE MODULES (可训练) ───────────────────────┤───│───│───│───│──┐  │
│  │                                                                      │   │   │   │   │  │  │
│  │    Zv ───────────────────────────────────────────────────────────────┼───│───│───│───│──│──┤
│  │    Zf ───────────────────────────────────────────────────────────────│───│───│───│───│──│──┤
│  │                                                                       │   │   │   │   │  │  │
│  │    ┌──────────────────────────────────────────────────────────────┐   │   │   │   │   │  │  │
│  │    │  StageGate                                                   │   │   │   │   │   │  │  │
│  │    │  Input:  [Zv | Zf] → [B, 2D]                                 ◄───┼───┼───│───│───│──│──┤  │
│  │    │  MLP:    2D → 256 → num_experts(4)                            │   │   │   │   │   │  │  │
│  │    │  Output:  stage_probs [B,4], expert_idx [B]                   │   │   │   │   │   │  │  │
│  │    └──────────────────────┬───────────────────────────────────────┘   │   │   │   │   │  │  │
│  │                           │ expert_idx                                │   │   │   │   │  │  │
│  │                           ▼                                           │   │   │   │   │  │  │
│  │    ┌──────────────────────────────────────────────────────────────┐   │   │   │   │   │  │  │
│  │    │  AtomicSkillExpertLibrary (Top-1 稀疏激活)                    │   │   │   │   │   │  │  │
│  │    │  Input:  concat[Zv, Zf, Zs, Zt] → [B, 4D]                    ◄───┼───┼───┼───┼───┼──┘  │
│  │    │                                                               │   │   │   │   │      │
│  │    │  expert[0]  ← approach  ← 3-layer MLP + residual              │   │   │   │   │      │
│  │    │  expert[1]  ← align     ← 3-layer MLP + residual              │   │   │   │   │      │
│  │    │  expert[2]  ← insert    ← 3-layer MLP + residual              │   │   │   │   │      │
│  │    │  expert[3]  ← tighten   ← 3-layer MLP + residual              │   │   │   │   │      │
│  │    │                                                               │   │   │   │   │      │
│  │    │  Output:  Z_expert [B, D] (only selected expert's output)     │   │   │   │   │      │
│  │    └──────────────┬──────────────────────┬────────────────────────┘   │   │   │   │      │
│  │                   │ Z_expert             │ expert_idx                  │   │   │   │      │
│  │                   ▼                      │                             │   │   │   │      │
│  │    ┌────────────────────────────────────┐ │                             │   │   │   │      │
│  │    │  ModalGate                         │ │                             │   │   │   │      │
│  │    │  Input:  [stage_probs | Z_expert]  ◄┘                             │   │   │   │      │
│  │    │  MLP:    (4+D) → 256 → 2                                         │   │   │   │      │
│  │    │  + stage_prior_bias[expert_idx]  ← 先验注入                       │   │   │   │      │
│  │    │  Output:  αv [B] (视觉权重), βf [B] (力觉权重)                     │   │   │   │      │
│  │    │  Constraint: α+β=1, α∈[0.05, 0.95]                               │   │   │   │      │
│  │    └────────────────────────────────────┘                               │   │   │   │      │
│  │                                                                         │   │   │   │      │
│  │              ┌──────────────┐                                           │   │   │   │      │
│  │              │ sa_moe_proj  │  ← Linear(D → vis_dim)                   │   │   │   │      │
│  │              │ Z_expert →   │                                           │   │   │   │      │
│  │              │ sa_moe_token │  [B, 1, vis_dim]                         │   │   │   │      │
│  │              └──────┬───────┘                                           │   │   │   │      │
│  │                     │                                                   │   │   │   │      │
│  │              ┌──────┴───────┐                                           │   │   │   │      │
│  │              │ force_out    │  ← Linear(D → 6)                         │   │   │   │      │
│  │              │ Z_expert →   │                                           │   │   │   │      │
│  │              │ F_desired    │  [B, 6] 期望力/力矩                        │   │   │   │      │
│  │              └──────────────┘                                           │   │   │   │      │
│  │                                                                         │   │   │   │      │
│  └─────────────────────────────────────────────────────────────────────────┘   │   │   │      │
│                                                                                 │   │   │      │
│  ┌────────────────────── PREFIX/SURFIX ASSEMBLY ────────────────────────────┐   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │  原始 Pi0 prefix:                                                        │   │   │   │      │
│  │    embed_prefix(images, lang) → [image_patches... | lang_tokens...]      │   │   │   │      │
│  │                              [B, N_prefix, vis_dim]                      │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │  ★ 增强 prefix (SA-MOE token 插入最前面):                                 │   │   │   │      │
│  │    [sa_moe_token] + [image_patches...] + [lang_tokens...]                │   │   │   │      │
│  │    [B, 1+N_prefix, vis_dim]                                              │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │  SA-MOE token 的 attention mask:                                          │   │   │   │      │
│  │    - pad_mask  = 1 (是有效 token，不是 padding)                            │   │   │   │      │
│  │    - att_mask  = 0 (双向注意力，可以看到所有 token)                         │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  └──────────────────────────────────────────────────────────────────────────┘   │   │   │      │
│                                                                                 │   │   │      │
│  ┌────────────────────── JOINT TRANSFORMER ─────────────────────────────────┐   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │   Suffix:  embed_suffix(state, x_t, time)                                │   │   │   │      │
│  │     → [action_expert_tokens...] + [noise_tokens...] + [time_token]       │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │   Full sequence = [Prefix | Suffix]                                      │   │   │   │      │
│  │     = [SA-MOE | image_patches | lang | action_expert | noise | time]     │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │         ┌──────────────────────────────────────────┐                     │   │   │   │      │
│  │         │  PaliGemmaWithExpertModel                 │                     │   │   │   │      │
│  │         │  Joint Attention (prefix ↔ suffix)        │                     │   │   │   │      │
│  │         │  每层 SA-MOE token 与所有 token 双向交互    │                     │   │   │   │      │
│  │         └──────────────────┬───────────────────────┘                     │   │   │   │      │
│  │                            │ suffix_out                                  │   │   │   │      │
│  │                            ▼                                             │   │   │   │      │
│  │         ┌──────────────────────────────────────────┐                     │   │   │   │      │
│  │         │  action_out_proj                          │                     │   │   │   │      │
│  │         │  suffix_out → v_t [B, 50, 32]             │                     │   │   │   │      │
│  │         │  → truncate to [B, 50, 7] (UR3 joints)    │                     │   │   │   │      │
│  │         └──────────────────────────────────────────┘                     │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  └──────────────────────────────────────────────────────────────────────────┘   │   │   │      │
│                                                                                 │   │   │      │
│  ┌────────────────────── OUTPUTS ───────────────────────────────────────────┐   │   │   │      │
│  │                                                                          │   │   │   │      │
│  │  action [B, 50, 7]  ← joint position velocity field (flow-matching)      │   │   │   │      │
│  │  force_pred [B, 6]  ← desired F/T from SA-MOE                            │   │   │   │      │
│  │  stage_probs [B, 4] ← stage classification (for monitoring)              │   │   │   │      │
│  │  alpha [B], beta [B]← modality weights (for monitoring)                  │   │   │   │      │
│  │                                                                          │   │   │   │      │
│  └──────────────────────────────────────────────────────────────────────────┘   │   │   │      │
│                                                                                 │   │   │      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 模块详解

### 4.1 ForceEncoder — 力/力矩编码器

**文件**: `sa_moe_modules.py:21-53`

```
输入:  force [B, 6]   (Fx, Fy, Fz, Tx, Ty, Tz)
结构:  Linear(6→256) → ReLU → LayerNorm → Linear(256→512) → ReLU → LayerNorm → Linear(512→1024)
输出:  Zf [B, 1024]
```

**作用**: 将 6 维 F/T 传感器原始信号映射到与视觉特征相同的 1024 维空间，使力觉信息能以相同"语言"被下游模块消费。

**初始化策略**: 最后一层用 `xavier_uniform(gain=0.01)` + `zero bias`——训练初期力特征接近零，不破坏 Pi0 已有的视觉推理能力。

### 4.2 StageGate — 阶段分类器 + Top-1 路由器

**文件**: `sa_moe_modules.py:60-101`

```
输入:  Zv [B, D], Zf [B, D]
拼接:  [Zv | Zf] → [B, 2D]
结构:  Linear(2D→256) → ReLU → LayerNorm → Linear(256→4)
输出:  stage_probs [B, 4]   (Softmax 概率)
       expert_idx  [B]       (argmax, 0-3)
```

**作用**: 基于视觉+力觉联合特征判断当前装配阶段。视觉可以看到空间关系（是否接近目标？），力觉可以检测接触状态（是否碰到零件？）。两者互补。

**4 个阶段对应**:

| idx | 阶段 | 特征 | 视觉主导? | 力觉主导? |
|-----|------|------|-----------|-----------|
| 0 | approach 接近 | 夹爪张开，向目标移动 | ✓ (α≈0.9) | |
| 1 | align 对准 | 末端靠近目标，微调位姿 | ✓ (α≈0.7) | |
| 2 | insert 插入 | 零件接触，力信号上升 | | ✓ (α≈0.2) |
| 3 | tighten 拧紧 | 大扭矩，力饱和 | | ✓ (α≈0.1) |

### 4.3 AtomicSkillExpertLibrary — 原子技能专家库

**文件**: `sa_moe_modules.py:108-232`

包含 4 个 `AtomicSkillExpert`，每个是一个 3 层 MLP + 残差连接：

```
输入:  concat[Zv, Zf, Zs, Zt] → [B, 4096]   (4×1024)

Layer 1: Linear(4096→1024) → GELU → Dropout(0.1) → LayerNorm
Layer 2: Linear(1024→1024) → GELU → Dropout(0.1) → LayerNorm + Residual
Layer 3: Linear(1024→1024) → LayerNorm

输出:  Z_expert [B, 1024]
```

**作用**: 每个专家学习对应阶段的专属特征变换。专家通过 **Top-1 稀疏激活** 路由——每个样本只激活一个专家。这保证了：

- **阶段专业化**: expert[0] 只看到 approach 阶段的数据，学到"如何接近目标"的特征
- **参数效率**: 4 个专家共享输入但各自独立，总参数 ≈ 4×4M = 16M
- **灾难性遗忘防护**: Pi0 主干冻结，专家只学习增量特征

**初始化策略**: `gain=0.01` + `zero bias`，确保初始状态下接近恒等映射，不破坏 Pi0 已有表征。

### 4.4 ModalGate — 模态自适应门控

**文件**: `sa_moe_modules.py:239-299`

```
输入:  stage_probs [B, 4] + Z_expert [B, D]
拼接:  [stage_probs | Z_expert] → [B, 4+D]
结构:  Linear(4+D → 256) → ReLU → LayerNorm → Linear(256 → 2)
       + stage_prior_bias[expert_idx] 添加到 vision logit
       → Softmax → clamp(0.05, 0.95)
输出:  α [B] (视觉权重), β [B] = 1-α (力觉权重)
```

**作用**: 根据阶段概率和专家特征，动态决定当前应该依赖视觉还是力觉。阶段的先验偏置（stage_priors）注入到最终层的 vision logit 上：

```
stage_priors = [2.0, 1.0, -2.0, -2.0]
# expert[0]=approach: +2.0 偏置 → 推高 vision logit → α 更大 → vision-heavy
# expert[3]=tighten:  -2.0 偏置 → 压低 vision logit → α 更小 → force-heavy
```

**约束**: α 始终 clamped 在 [0.05, 0.95]，防止某一模态完全坍塌。

### 4.5 投影层

| 层 | 文件位置 | 输入 | 输出 | 作用 |
|----|---------|------|------|------|
| `vis_proj` | `modeling_sa_moe_pi0.py:104` | `[B, vis_dim]` | `[B, D(1024)]` | 将 SigLIP 实际输出维度投影到统一 D 维 |
| `state_to_D` | `modeling_sa_moe_pi0.py:105` | `[B, expert_width]` | `[B, D]` | 将 action_expert state 投影到 D |
| `text_to_D` | `modeling_sa_moe_pi0.py:106` | `[B, expert_width]` | `[B, D]` | 将语言 embedding 投影到 D |
| `sa_moe_proj` | `modeling_sa_moe_pi0.py:139` | `[B, D]` | `[B, vis_dim]` | SA-MOE token → prefix embedding 空间 |
| `force_out_proj` | `modeling_sa_moe_pi0.py:144` | `[B, D]` | `[B, 6]` | 专家特征 → 期望力/力矩预测 |

---

## 5. 数据流详解

### 5.1 训练前向传播

```
Step 1: Flow-Matching Setup (Pi0 原生)
  noise ~ N(0,I)                                    → [B, 50, 32]
  time ~ Beta(α,β)                                  → [B]
  x_t = time·noise + (1-time)·actions               → [B, 50, 32]   (noisy actions)
  u_t = noise - actions                              → [B, 50, 32]   (target velocity)

Step 2: 多模态特征提取
  Zv = vis_proj(mean(SigLIP(images)))               → [B, 1024]     ← 冻结 SigLIP
  Zf = ForceEncoder(force)                           → [B, 1024]     ← ★ 可训练
  Zs = state_proj(padded_state)                      → [B, 1024]     ← 冻结
  Zt = text_to_D(mean(embed_lang(lang_tokens)))     → [B, 1024]     ← 冻结

Step 3: SA-MOE 前向
  stage_probs, expert_idx = StageGate(Zv, Zf)        → [B,4], [B]
  z_concat = [Zv | Zf | Zs | Zt]                    → [B, 4096]
  Z_expert = ExpertLibrary(z_concat, expert_idx)     → [B, 1024]
  α, β = ModalGate(stage_probs, Z_expert, idx)      → [B], [B]
  sa_moe_token = sa_moe_proj(Z_expert)               → [B, 1, vis_dim]

Step 4: Prefix 增强
  prefix_embs, pad_masks, att_masks = embed_prefix(images, lang)
  prefix_embs = concat([sa_moe_token, prefix_embs], dim=1)    ← 插入最前面
  pad_masks  = concat([1, pad_masks], dim=1)
  att_masks  = concat([0, att_masks], dim=1)   ← SA-MOE token 双向注意力

Step 5: Suffix + Joint Transformer
  suffix_embs = embed_suffix(state, x_t, time)
  full_seq = [prefix | suffix]
  suffix_out = PaliGemmaWithExpertModel(full_seq)    → [... | chunk_size tokens]
  v_t = action_out_proj(suffix_out[:, -chunk_size:]) → [B, 50, 32]

Step 6: 输出 + 损失
  action_loss = MSE(v_t[:,:,:7], u_t[:,:,:7])         ← 流匹配损失
  force_pred = force_out_proj(Z_expert)               → [B, 6]
  stage_loss = CrossEntropy(stage_probs, labels)       ← 暂为 0
  balance_loss = -entropy(mean(stage_probs))           ← 防止专家坍塌
  alpha_aux_loss = MSE(α, target_α[expert_idx])       ← 引导 α 到阶段预期值
  total = action_loss + 0.2·stage + 0.01·balance + 0.1·alpha_aux + 0.1·force
```

### 5.2 推理前向传播 (sample_actions)

```
Step 1: 提取 SA-MOE 特征（与训练相同）
  Zv, Zf, Zs, Zt → stage_probs, expert_idx → Z_expert → sa_moe_token
  α, β = ModalGate(...)

Step 2: 编码 prefix（含 SA-MOE token）+ KV cache
  prefix_embs = embed_prefix(images, lang)
  prefix_embs = concat([sa_moe_token, prefix_embs])
  _, past_key_values = PaliGemmaWithExpertModel(prefix, use_cache=True)

Step 3: Flow-Matching 去噪循环 (num_inference_steps 步)
  x_t = noise ~ N(0,I)                               → [B, 50, 32]
  for step in range(num_inference_steps):
      t = 1.0 + step * (-1.0/num_steps)               → [B]
      v_t = denoise_step(state, prefix_kv, past_kv, x_t, t)
      x_t = x_t - (1/num_steps) * v_t                ← Euler 积分

Step 4: 输出
  actions = x_t[:, :, :7]                             → [B, 50, 7]
  force_pred = force_out_proj(Z_expert)               → [B, 6]
  stage_probs, α, β                                   → 用于监控
```

### 5.3 Batch 数据格式

```python
batch = {
    # 视觉 (与 Pi0 一致)
    "observation.images.camera0":  Tensor[B, 3, 224, 224],   # SigLIP 预处理后
    "observation.images.camera1":  Tensor[B, 3, 224, 224],

    # 状态
    "observation.state":           Tensor[B, 7],              # 关节位置 (padded → 32)

    # 力/力矩 (新增)
    "observation.force":           Tensor[B, 6],              # Fx,Fy,Fz,Tx,Ty,Tz

    # 语言
    "observation.language.tokens":         Tensor[B, 48],
    "observation.language.attention_mask": Tensor[B, 48],

    # 动作
    "action":                       Tensor[B, 50, 7],         # (padded → 50,32)

    # 期望力标签 (可选)
    "action.force":                 Tensor[B, 6],             # 接触阶段 GT
}
```

---

## 6. 训练策略

### 6.1 参数冻结策略

```
Pi0 全部参数:     requires_grad = False  (约 2.3B)
  ├── SigLIP vision encoder
  ├── PaliGemma language model
  ├── Action Expert (Gemma 300M)
  ├── state_proj, action_out_proj
  ├── embed_prefix, embed_suffix
  └── PaliGemmaWithExpertModel

SA-MOE 模块:      requires_grad = True   (约 20M)
  ├── ForceEncoder        (6→256→512→1024)      ~2.8M
  ├── StageGate           (2048→256→4)          ~0.5M
  ├── ExpertLibrary       (4×[4096→1024→1024])  ~16.8M
  ├── ModalGate           (4+1024→256→2)        ~0.3M
  ├── vis_proj            (vis_dim→1024)        ~1.0M (if needed)
  ├── state_to_D          (expert_dim→1024)      ~1.0M (if needed)
  ├── text_to_D           (expert_dim→1024)      ~1.0M (if needed)
  ├── sa_moe_proj         (1024→vis_dim)         ~1.0M
  └── force_out_proj      (1024→6)               ~6K

总可训练参数: ~25M  (约占 Pi0 总参数的 1%)
```

### 6.2 优化器配置

```python
# SAMoEPi0Config 专属优化器
AdamW(
    lr=1e-4,                # 比全量微调高 10x（只训练小模块）
    betas=(0.9, 0.95),
    weight_decay=0.01,
    grad_clip_norm=1.0,
)
```

### 6.3 训练命令

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=sa_moe_pi0 \
  --policy.path=./ai-models/lerobot/pi0_libero_base \
  --dataset.repo_id=cjx-cell/ur3_pick_place \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --batch_size=4 \
  --steps=30000 \
  --output_dir=./outputs/train/ur3_sa_moe \
  --save_freq=5000 \
  --log_freq=50
```

训练前的验证：

```bash
# 1. 导入测试
python -c "
from lerobot.policies.sa_moe_pi0 import SAMoEPi0Config, SAMoEPi0Policy
cfg = SAMoEPi0Config()
print('Config:', cfg.type)
print('Features:', list(cfg.input_features.keys()))
"

# 2. 工厂加载测试
python -c "
from lerobot.policies.factory import get_policy_class, make_policy_config
cls = get_policy_class('sa_moe_pi0')
print('Policy class:', cls)
"

# 3. 前向形状测试
python -c "
import torch
from lerobot.policies.sa_moe_pi0 import SAMoEPi0Config, SAMoEPi0Policy
cfg = SAMoEPi0Config(device='cpu', dtype='float32')
cfg.input_features['observation.images.camera0'] = \
    type(cfg.input_features['observation.state'])(
        type=type(cfg.input_features['observation.state']).type, shape=(3,480,640))
cfg.input_features['observation.images.camera1'] = \
    type(cfg.input_features['observation.state'])(
        type=type(cfg.input_features['observation.state']).type, shape=(3,480,640))
cfg.validate_features()

policy = SAMoEPi0Policy(cfg)
batch = {
    'observation.images.camera0': torch.randn(2,3,224,224),
    'observation.images.camera1': torch.randn(2,3,224,224),
    'observation.state': torch.randn(2,7),
    'observation.force': torch.randn(2,6),
    'observation.language.tokens': torch.randint(0,256,(2,48)),
    'observation.language.attention_mask': torch.ones(2,48),
    'action': torch.randn(2,50,7),
}
loss, loss_dict = policy.forward(batch)
print('Loss:', loss.item())
print('Dict:', {k: v for k,v in loss_dict.items() if isinstance(v, (int,float))})

# 4. 验证可训练参数
trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total = sum(p.numel() for p in policy.parameters())
print(f'Trainable: {trainable:,} / {total:,} = {100*trainable/total:.1f}%')
"
```

---

## 7. 阶段标签生成（待实现）

当前 `stage_loss = 0`（无监督）。后续可基于物理启发式规则自动生成伪标签：

```python
def generate_stage_labels(gripper_pos, force, threshold_force=5.0):
    """
    gripper_pos: [B] 夹爪位置 (0=闭合, 1=张开)
    force:       [B, 6] F/T 传感器
    """
    force_mag = torch.norm(force[:, :3], dim=-1)  # 力幅值
    torque_mag = torch.norm(force[:, 3:], dim=-1) # 力矩幅值

    labels = torch.zeros(B, dtype=torch.long)

    # 接近: 夹爪张开 + 无力
    approach_mask = (gripper_pos > 0.3) & (force_mag < threshold_force)
    labels[approach_mask] = 0

    # 对准: 夹爪张开 + 微弱力（靠近目标表面）
    align_mask = (gripper_pos > 0.3) & (force_mag >= threshold_force * 0.3) & \
                 (force_mag < threshold_force)
    labels[align_mask] = 1

    # 插入: 夹爪闭合 + 中力
    insert_mask = (gripper_pos < 0.05) & (force_mag >= threshold_force) & \
                  (torque_mag < threshold_force * 2)
    labels[insert_mask] = 2

    # 拧紧: 夹爪闭合 + 大力 + 大扭矩
    tighten_mask = (gripper_pos < 0.05) & (torque_mag >= threshold_force * 2)
    labels[tighten_mask] = 3

    return labels
```

---

## 8. 文件对照表

| 计划中的文件 | 实际文件 | 状态 |
|------------|---------|------|
| `sa_moe_pi0/__init__.py` | `lerobot/policies/sa_moe_pi0/__init__.py` | ✅ 已创建 |
| `sa_moe_pi0/configuration_sa_moe_pi0.py` | `lerobot/policies/sa_moe_pi0/configuration_sa_moe_pi0.py` | ✅ 已创建 |
| `sa_moe_pi0/modeling_sa_moe_pi0.py` | `lerobot/policies/sa_moe_pi0/modeling_sa_moe_pi0.py` | ✅ 已创建 |
| `sa_moe_pi0/processor_sa_moe_pi0.py` | `lerobot/policies/sa_moe_pi0/processor_sa_moe_pi0.py` | ✅ 已创建 |
| `pi0/sa_moe_modules.py` (共用) | `sa_moe_pi0/sa_moe_modules.py` | ✅ 已升级（独立副本） |
| `factory.py` (3 处修改) | `lerobot/policies/factory.py` | ✅ 已注册 |
| `policies/__init__.py` | `lerobot/policies/__init__.py` | ✅ 已导出 |
| — | `pi0/sa_moe_modules.py` (旧) | 📦 保留参考 |
| — | `pi0/sa_moe_pi0.py` (旧) | 📦 保留参考 |
| — | `pi0/sa_moe_config.yaml` (旧) | 📦 保留参考 |

---

## 9. 当前限制 & 后续工作

| 限制 | 说明 | 计划 |
|------|------|------|
| 无阶段监督 | `stage_loss = 0`，StageGate 无直接训练信号 | 实现物理启发式伪标签生成器 |
| 无 GT 力标签 | `force_loss = 0`，force_out_proj 不受监督 | 从数据中提取或加入力控目标 |
| 未端到端测试 | forward() 形状正确但未跑完整训练 | 在 A100 上验证完整训练+推理 |
| 与 UR3 任务的适配 | SA-MOE 面向装配，当前任务是 pick-place | pick-place 可能不需要力觉，先验证架构可行性 |
| 模型加载兼容性 | SAMoEPi0Config 需 pi0_libero_base 的 safetensors 作为起点 | 用 `from_pretrained` 加载 Pi0 权重，SA-MOE 模块随机初始化 |
