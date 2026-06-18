# SA-MOE (Stage-Aware Mixture of Experts) 方案文档

> 版本：v6 — 最终架构 | E_VL 输入 + 条件 token 注入

## 1. 数据流

```
══════════════════════════════════════════════════════════════════
                    SA-MOE v6 — 完整数据流
══════════════════════════════════════════════════════════════════

  图像+文本 → PaliGemma 18层 → E_VL [B, 560, 2048]
                                       │
                               ┌───────┴────────┐
                               │                │
                               ▼                ▼
                          KV cache        SA-MOE(E_VL, force)
                       (主通路,完全不改)         │
                               │         ┌─────┴──────┐
                               │         │ Transformer │  2层自注意力
                               │         │ StageGate   │  阶段分类
                               │         │ Expert      │  阶段特化
                               │         │ ModalGate   │  模态权重
                               │         │ Fusion      │  显式加权
                               │         └─────┬──────┘
                               │               │
                               │        G_sa_moe [B, 1, 1024]
                               │               │
                    state ──────┤               │
                    noise ──────┤               │
                               │               │
                               ▼               ▼
                    ┌──────────────────────────────────┐
                    │  Suffix = [G_sa_moe  ← 条件token │
                    │           | state_token           │
                    │           | actions×50            │
                    │           | time_token]           │
                    └──────────────┬───────────────────┘
                                   │
                           Gemma Expert 300M
                           (每层 cross-attn → KV cache)
                           (每层 self-attn → G_sa_moe)
                                   │
                           action_out_proj → v_t
```

## 2. 模块清单

### SA-MOE 核心（全部可训练，~55M）

| # | 模块 | 维度 | 作用 |
|---|------|------|------|
| 1 | ForceEncoder | [B,6]→[B,1024] | 力/力矩编码 |
| 2 | force_to_vlm | [B,1024]→[B,2048] | 力 token 投影到 VLM 维度 |
| 3 | **Transformer Encoder** | [B, N+1, 2048] | 2层自注意力，E_VL + force token 交互 |
| 4 | vlm_to_sa_moe | [B,2048]→[B,1024] | 池化后投影回 SA-MOE 维度 |
| 5 | StageGate | Zv+Zf→[B,4] | 阶段分类（接近/对准/插入/拧紧） |
| 6 | ExpertLibrary | [B,3072]→[B,1024] | 4 个阶段专属 MLP 专家 |
| 7 | ModalGate | [B,4+1024]→[B,2] | 模态权重 αv, βf |
| 8 | ModalityWeightedFusion | [B,2048]→[B,1024] | 显式 αv·Zv + βf·Zf |
| 9 | sa_moe_proj | [B,2048]→[B,1024] | G_sa_moe 条件 token |

### Pi0 组件（冻结）

| 组件 | 作用 |
|------|------|
| SigLIP | 图像→patch 嵌入 |
| PaliGemma 2B | 18层处理图像+语言 → E_VL + KV cache |
| state_proj | 状态 [7]→[1024] |
| Gemma Expert 300M | 18层 self-attn + cross-attn → 去噪 |
| action_out_proj | [1024]→[7] 关节速度场 |

## 3. 关键设计决策

| 决策 | 原因 |
|------|------|
| SA-MOE 输入 E_VL 而非原始 patches | PaliGemma 已处理 18 层，不重复学视觉 |
| E_VL → KV cache（主通路）\| 池化 → SA-MOE（条件通路） | KV cache 保持 PaliGemma 预训练语义完全不变；SA-MOE 只池化聚合，不修改原始 token |
| G_sa_moe 作为 suffix 条件 token | 阶段+模态信息贯穿 Gemma Expert 全部 18 层，深层引导而非末端修正 |
| 1 个全局条件 token 而非 H_action 个逐步 token | 力觉+阶段是全局状态，应由解码器自行学习如何调制各步动作 |
| 状态不进 SA-MOE | 状态是执行信息，不是感知信息；Pi0 原本就直连动作专家 |
| E_VL + SA-MOE 一次性计算 | 推理时只算一次，不每去噪步重算 |

## 4. 与 ForceVLA 的本质差异

ForceVLA 源码分析确认了以下核心区别：

| | ForceVLA | SA-MOE v6 |
|---|---|---|
| E_VL 来源 | 联合前向 `prefix_out`（已和 suffix 交叉过） | 独立 prefix-only 前向（纯感知，未混入动作） |
| 力觉注入深度 | **末端修正**：残差加在 `action_out_proj` 之前 | **全层引导**：条件 token 贯穿 Gemma Expert 18 层 |
| 对 E_VL 的干预 | LIMoE 二次编码全部 VL token，**修改预训练特征** | 只池化聚合，**不修改原始 token 序列** |
| 融合输出 | `H_action` 个逐步 token（每步一个） | **1 个全局条件 token** |
| 融合逻辑 | per-token MoE 隐式路由，黑盒 | **StageGate + ModalGate 显式输出**，白盒可解释 |
| 约束能力 | 无 | **stage_prior_bias + alpha_aux_loss** 可人工干预 |

ForceVLA 是学术验证（力模态+MoE 有效），SA-MOE v6 是工程落地（更深注入+显式可控+保留预训练泛化）。

## 5. 参数统计

```
Pi0 backbone (冻结):  ~2.30B
SA-MOE (可训练):     ~55M    (2.3%)

  ForceEncoder                ~2.8M
  force_to_vlm                ~2.1M
  Transformer Encoder (2层)  ~33.6M
  vlm_to_sa_moe               ~2.1M
  StageGate                   ~0.5M
  ExpertLibrary (4个, 3*D)   ~12.6M
  ModalGate                   ~0.3M
  ModalityWeightedFusion      ~2.1M
  sa_moe_proj                 ~2.1M
```

## 6. 文件结构

```
lerobot/src/lerobot/policies/sa_moe_pi0/
├── __init__.py
├── configuration_sa_moe_pi0.py
├── sa_moe_modules.py
│   ├── ForceEncoder
│   ├── MultimodalTransformerEncoder
│   ├── StageGate
│   ├── AtomicSkillExpert / ExpertLibrary
│   ├── ModalGate
│   ├── ModalityWeightedFusion
│   └── compute_load_balancing_loss / compute_alpha_aux_loss / generate_stage_labels
├── modeling_sa_moe_pi0.py
│   ├── SAMoEPi0Model(PI0Pytorch)
│   │   ├── _get_vlm_output()       # PaliGemma prefix-forward → E_VL + KV
│   │   ├── _forward_sa_moe()       # SA-MOE 融合（Transformer→Gate→Expert→Fusion）
│   │   ├── denoise_step()          # 覆写：G_sa_moe 作为 suffix 条件 token
│   │   ├── forward()               # 训练：prefix → SA-MOE → suffix+condition
│   │   └── sample_actions()        # 推理：同上
│   └── SAMoEPi0Policy(PI0Policy)
└── processor_sa_moe_pi0.py
```
