# SA-MOE (Stage-Aware Mixture of Experts) 方案文档

> 版本：v4 — Transformer Encoder 中间融合层 | 状态：代码已实现，待训练验证

## 1. 方案定位

面向**在轨装配/在轨服务**（物理本质 = 轴孔装配 peg-in-hole），在 Pi0 VLA 模型基础上
集成力觉模态 + 阶段感知稀疏混合专家，实现「非接触视觉主导、接触力觉主导」的自适应
多模态决策。

SA-MOE 作为 **VLM 感知与动作专家之间的中间融合层**，只处理三种感知模态：

```
                             VLM (感知，冻结)               SA-MOE (融合，可训练)         动作专家 (执行，冻结)

   图像 ──► SigLIP ────► image_patches ──► PaliGemma KV ─────────────────────────┐
          │                                                                      │
          └──► mean pool ► Zv ──────────────────────┐                             │
   文本 ──► Embed ──► lang_tokens ──► PaliGemma KV ─┤                             │
          │                                         │                             │
          └──► mean pool ► Zt ────────────────────┐ │                             │
   力   ──► ForceEncoder ─► Zf ─────────────────┐ │ │                             │
                                                  │ │ │                             │
                                         ┌────────▼─▼─▼──────────┐                  │
                                         │  Transformer Encoder   │                  │
                                         │  (2层自注意力)          │                  │
                                         │       ↓               │                  │
                                         │  StageGate → stage    │                  │
                                         │  Expert[stage]         │                  │
                                         │  ModalGate → αv, αf   │                  │
                                         │  Fusion               │                  │
                                         └───────────┬───────────┘                  │
                                                     │ G_sa_moe                    │
   状态 ──► state_proj ─────────────────────────────┐ │                             │
   噪声 ──► action_in_proj ─────────────────────────┤ │                             │
                                                     │ │                             │
                                            ┌────────▼─▼──────────┐                  │
                                            │  Gemma Expert 300M  │                  │
                                            │  + G_sa_moe 残差     │                  │
                                            │  cross-attn to KV ──┘                  │
                                            └─────────┬──────────┘
                                                      ▼
                                                    动作 v_t
```

## 2. 模块清单

### 感知层（冻结，Pi0 预训练）

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| SigLIP | 图像 [B,3,224,224]×2 | image_patches [B,512,2048] | 视觉特征提取 |
| PaliGemma Embed | 文本 token [B,48] | lang_tokens [B,48,2048] | 语言理解 |
| state_proj | 状态 [B,7]→pad[32] | [B,1024] | 状态编码（直接进后缀） |

### 投影层（可训练）

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| vis_proj | Zv_raw [B,2048] | Zv [B,1024] | 视觉特征降维 |
| text_to_D | Zt_raw [B,2048] | Zt [B,1024] | 语言特征降维 |
| ForceEncoder | 力/力矩 [B,6] | Zf [B,1024] | 力觉编码 |

### SA-MOE 核心融合层（全部可训练，~28M）

| 模块 | 位置 | 作用 |
|------|------|------|
| **MultimodalTransformerEncoder** | 最前 | 2层自注意力，让 Zv/Zf/Zt 三个感知 token 互相交互。Zv 获得力觉上下文，Zf 获得视觉上下文 |
| **StageGate** | Transformer 之后 | 基于 refined Zv'+Zf' 判断当前装配阶段（接近/对准/插入/拧紧），输出 stage_probs + expert_idx |
| **ExpertLibrary** | StageGate 之后 | 4 个阶段专属 MLP 专家。同一输入 `[Zv',Zf',Zt']`，不同专家提取不同特征——expert[0]专注空间接近度，expert[3]专注力矩柔顺 |
| **ModalGate** | Expert 之后 | 基于 stage_probs + Z_expert 输出 αv, αf。决定当前信视觉还是力觉。含 stage_prior_bias 物理先验 |
| **ModalityWeightedFusion** | 最末 | **显式执行** αv·Zv + βf·Zf，concat 后投影。α/β 功能性参与特征计算 |

### 注入 + 输出

| 模块 | 作用 |
|------|------|
| sa_moe_proj | concat[Z_fused, Z_expert] → G_sa_moe [B,1024] → expand [B,50,1024] |
| 注入方式 | **suffix 残差加法**：suffix_out[:, -50:] + G_sa_moe → action_out_proj |
| force_out_proj | Z_expert → 期望力/力矩 [B,6] |

## 3. 数据流（训练）

```
1. 标准 Pi0 前缀（不动）
   [img_patches | lang_tokens] → PaliGemma 2B → KV cache

2. 特征提取
   Zv = vis_proj(mean(SigLIP_patches))  [B,1024]  ← 冻结
   Zf = ForceEncoder(force [B,6])        [B,1024]  ← 可训练
   Zt = text_to_D(mean(lang_emb))        [B,1024]  ← 冻结

3. Transformer Encoder
   tokens = stack([Zv, Zf, Zt], dim=1)   [B,3,1024]
   tokens = TransformerEncoder(tokens)   [B,3,1024]
   Zv', Zf', Zt' = tokens[:,0], tokens[:,1], tokens[:,2]

4. StageGate + Expert
   stage_probs, expert_idx = StageGate(Zv', Zf')
   Z_expert = Expert[expert_idx](concat[Zv', Zf', Zt'])  [B,1024]

5. ModalGate + Fusion
   α, β = ModalGate(stage_probs, Z_expert, expert_idx)
   Z_fused = Fusion(Zv', Zf', α, β)      [B,1024]

6. 组装 + 注入
   G_sa_moe = sa_moe_proj(concat[Z_fused, Z_expert])  [B,1024]
   G_sa_moe = expand → [B,50,1024]

7. Pi0 后缀 + 残差
   suffix_out = JointTransformer(Prefix_KV, [state|noise×50])
   action_tokens = suffix_out[:, -50:] + G_sa_moe
   v_t = action_out_proj(action_tokens)   [B,50,7]
```

## 4. 推理数据流

```
1. Pi0 prefix 编码 → KV cache [一次性]
2. 样本噪声 x_1 = N(0,I) [B,50,32]
3. 提取 SA-MOE: tokens → Transformer → Gate → Expert → Fusion → G_sa_moe [B,50,1024]
4. for step in 0..num_steps-1:
     suffix_out = Gemma Expert(suffix, cross_attn_to_KV)
     v_t = action_out_proj(suffix_out[:, -50:] + G_sa_moe)
     x_t = x_t + dt * v_t
5. 返回 actions, force_pred, αv, βf, stage_probs
```

## 5. 损失函数

| 损失 | 公式 | 权重 | 说明 |
|------|------|------|------|
| L_action | MSE(v_t, u_t) | 1.0 | Pi0 流匹配主损失 |
| L_stage | CrossEntropy(stage_probs, pseudo_labels) | 0.2（5K步后激活） | 阶段分类监督 |
| L_balance | -entropy + underuse_penalty | 0.01 | 防止专家坍塌 |
| L_alpha | MSE(αv, target_α[expert_idx]) | 0.2 | 引导模态权重符合物理直觉 |

损失权重调度：warmup(0-5K) → main(5K-20K) → finetune(20K+)

## 6. 参数统计

```
总参数:     ~2.36B
冻结:       ~2.33B (Pi0 backbone)
可训练:     ~28M    (1.0%)

  ForceEncoder                ~2.8M
  Transformer Encoder (2层)  ~12.6M  ← v4 新增
  StageGate                   ~0.5M
  ExpertLibrary (4个, 3*D)   ~12.6M
  ModalGate                   ~0.3M
  ModalityWeightedFusion      ~2.1M
  sa_moe_proj                 ~2.1M
  vis/text projections        ~4.2M
```

## 7. 文件结构

```
lerobot/src/lerobot/policies/sa_moe_pi0/
├── __init__.py                   # 导出 SAMoEPi0Config, SAMoEPi0Policy
├── configuration_sa_moe_pi0.py   # 配置类（继承 PI0Config）
├── sa_moe_modules.py             # 所有模块定义
│   ├── ForceEncoder              # 力/力矩编码器
│   ├── MultimodalTransformerEncoder  # v4: 自注意力感知融合
│   ├── StageGate                 # 阶段分类 + Top-1 路由
│   ├── AtomicSkillExpert         # 单个阶段专家 (3层MLP+残差)
│   ├── AtomicSkillExpertLibrary  # 4专家 Top-1 调度
│   ├── ModalGate                 # 模态权重 αv/αf
│   ├── ModalityWeightedFusion    # 显式 αv·Zv + βf·Zf 融合
│   └── generate_stage_labels()   # 阶段伪标签生成
├── modeling_sa_moe_pi0.py        # SAMoEPi0Model + SAMoEPi0Policy
└── processor_sa_moe_pi0.py       # 预/后处理器
```

## 8. 与 ForceVLA 对比

| | ForceVLA (NeurIPS 2025) | SA-MOE v4 |
|---|---|---|
| 力编码 | Linear [6→2048] | ForceEncoder MLP [6→1024] |
| 模态交互 | Transformer Encoder (MHA) | **Transformer Encoder (MHA)** |
| 专家路由 | MoE (4专家, per-token 隐式路由) | **StageGate (4专家, 阶段显式路由)** |
| 模态权重 | ❌ 无 | **αv/αf 可监控** |
| 阶段先验 | ❌ 无 | **stage_prior_bias** |
| 注入点 | Suffix 残差 | Suffix 残差 |
| 动作生成 | Pi0 flow matching | Pi0 flow matching |
| 可解释性 | ★ | ★★★ |
| 物理归纳偏置 | ★★ | ★★★ |

## 9. 各模块在在轨装配场景中的角色

| 阶段 | StageGate | Expert | αv | 行为 |
|------|-----------|--------|-----|------|
| 接近 | `[0.9, 0.1, 0, 0]` | expert[0] | ~0.9 | 视觉引导靠近目标，力传感器无读数 |
| 对准 | `[0.2, 0.7, 0.1, 0]` | expert[1] | ~0.7 | 微调位姿，微弱力反馈 |
| 插入 | `[0, 0.1, 0.8, 0.1]` | expert[2] | ~0.2 | 力主导柔顺控制 |
| 拧紧 | `[0, 0, 0.1, 0.9]` | expert[3] | ~0.1 | 完全依赖力觉，大扭矩锁定 |
