#!/usr/bin/env python3
"""
修复 V4 checkpoint 中动作归一化统计量：将绝对动作统计量替换为相对动作统计量。

问题：
  Preprocessor 管道是 raw → relative → normalize → model
  但 Normalizer 使用的 stats 来自 dataset.meta.stats（绝对动作），
  导致 normalized_delta ≈ (delta - 1.5) / 0.5 ≈ -3.0，模型工作在极端数值范围。

修复：
  从原始数据计算 action_rel = action - state 的 mean/std，
  对 excluded joints（gripper）保留绝对动作统计量，
  写入 checkpoint 的 normalizer/unnormalizer safetensors 文件。

用法 (pi0-env):
  conda activate pi0-env
  python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_fix_relative_action_stats.py \
      --input ai-models/ur3_pick_place_raw \
      --checkpoint outputs/train/ur3_pi0_v4_expert \
      --exclude-joints gripper
"""

import argparse, os, sys
import numpy as np
from pathlib import Path

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
STATE_DIM = 7


def find_episodes(input_dir):
    """扫描所有 success episode 的 data.npz 文件"""
    episodes = []
    for d in sorted(os.listdir(input_dir)):
        ep_dir = os.path.join(input_dir, d)
        if "_episode_" in d and os.path.isdir(ep_dir) and "_failed" not in d:
            npz_path = os.path.join(ep_dir, "data.npz")
            if os.path.exists(npz_path):
                episodes.append(npz_path)
    return episodes


def compute_relative_action_stats(input_dir, exclude_joints):
    """
    从原始数据计算相对动作 (action - state) 的统计量。

    对 excluded_joints，保留原始绝对动作统计量。
    对其他关节，计算 action_rel = action - state 的 mean/std。
    """
    episodes = find_episodes(input_dir)
    if not episodes:
        raise FileNotFoundError(f"在 {input_dir} 中未找到任何 episode 数据")

    print(f"计算相对动作统计量 ({len(episodes)} episodes)...")

    all_states = []
    all_actions_abs = []

    for ep_path in episodes:
        data = np.load(ep_path)
        # action 在原始数据中等于 next_state（绝对关节角）
        states = data["state"]   # shape: (N, 7)
        actions = data["action"] # shape: (N, 7), action ≈ next_state
        all_states.append(states)
        all_actions_abs.append(actions)
        data.close()

    all_states = np.concatenate(all_states, axis=0)
    all_actions_abs = np.concatenate(all_actions_abs, axis=0)

    # 相对动作: action_rel = action_abs - state
    action_rel = all_actions_abs - all_states
    n_frames = len(action_rel)

    print(f"  总帧数: {n_frames}")
    print(f"  State 范围: [{all_states.min():.4f}, {all_states.max():.4f}]")
    print(f"  Action abs 范围: [{all_actions_abs.min():.4f}, {all_actions_abs.max():.4f}]")

    # 确定哪些关节使用相对统计量，哪些使用绝对统计量
    if exclude_joints is None:
        exclude_joints = []

    exclude_indices = []
    for joint_name in exclude_joints:
        if joint_name in ALL_JOINTS:
            exclude_indices.append(ALL_JOINTS.index(joint_name))
        elif joint_name == "gripper":
            # "gripper" 是 relative_exclude_joints 中常用的简写
            exclude_indices = [6]  # gripper 是第 7 个关节
            break

    print(f"  排除关节 (保持绝对): {[ALL_JOINTS[i] for i in exclude_indices]}")

    # 构建新的 action stats
    # 对于非排除关节: 使用相对动作统计量 (mean ≈ 0, std ≈ 0.01-0.05)
    # 对于排除关节: 使用绝对动作统计量 (保持原样)
    new_action_mean = np.zeros(STATE_DIM, dtype=np.float32)
    new_action_std = np.zeros(STATE_DIM, dtype=np.float32)
    new_action_min = np.zeros(STATE_DIM, dtype=np.float32)
    new_action_max = np.zeros(STATE_DIM, dtype=np.float32)

    for i in range(STATE_DIM):
        if i in exclude_indices:
            # 使用绝对动作统计量
            new_action_mean[i] = all_actions_abs[:, i].mean()
            new_action_std[i] = all_actions_abs[:, i].std()
            new_action_min[i] = all_actions_abs[:, i].min()
            new_action_max[i] = all_actions_abs[:, i].max()
        else:
            # 使用相对动作统计量
            new_action_mean[i] = action_rel[:, i].mean()
            new_action_std[i] = action_rel[:, i].std()
            new_action_min[i] = action_rel[:, i].min()
            new_action_max[i] = action_rel[:, i].max()

    # 打印新旧对比
    print(f"\n{'关节':<16} {'旧 mean (abs)':>12} {'新 mean':>12} {'旧 std (abs)':>12} {'新 std':>12}")
    print("-" * 64)
    for i, name in enumerate(ALL_JOINTS):
        old_mean = all_actions_abs[:, i].mean()
        old_std = all_actions_abs[:, i].std()
        marker = " [abs]" if i in exclude_indices else " [rel]"
        print(f"{name+marker:<16} {old_mean:>12.6f} {new_action_mean[i]:>12.6f} "
              f"{old_std:>12.6f} {new_action_std[i]:>12.6f}")

    return {
        "mean": new_action_mean,
        "std": new_action_std,
        "min": new_action_min,
        "max": new_action_max,
        "count": np.array([n_frames], dtype=np.float32),
        # 分位数基于相对动作（或绝对，取决于关节）
        "q01": _compute_quantiles(action_rel, all_actions_abs, exclude_indices, 0.01),
        "q10": _compute_quantiles(action_rel, all_actions_abs, exclude_indices, 0.10),
        "q50": _compute_quantiles(action_rel, all_actions_abs, exclude_indices, 0.50),
        "q90": _compute_quantiles(action_rel, all_actions_abs, exclude_indices, 0.90),
        "q99": _compute_quantiles(action_rel, all_actions_abs, exclude_indices, 0.99),
    }


def _compute_quantiles(action_rel, actions_abs, exclude_indices, q):
    """计算分位数，对排除关节使用绝对动作"""
    result = np.zeros(STATE_DIM, dtype=np.float32)
    for i in range(STATE_DIM):
        if i in exclude_indices:
            result[i] = np.quantile(actions_abs[:, i], q)
        else:
            result[i] = np.quantile(action_rel[:, i], q)
    return result


def fix_checkpoint(checkpoint_dir, new_action_stats, dry_run=False):
    """
    更新 checkpoint 中 normalizer 和 unnormalizer 的 action stats。

    修改的文件:
      - policy_preprocessor_step_6_normalizer_processor.safetensors
      - policy_postprocessor_step_0_unnormalizer_processor.safetensors
    """
    from safetensors.torch import load_file, save_file
    import torch

    ckpt_path = Path(checkpoint_dir)

    # 查找所有 pretrained_model 目录
    pretrained_dirs = list(ckpt_path.rglob("pretrained_model"))
    if not pretrained_dirs:
        # 检查是否本身就是 pretrained_model
        if (ckpt_path / "model.safetensors").exists():
            pretrained_dirs = [ckpt_path]
        else:
            print(f"错误: 在 {checkpoint_dir} 中未找到 pretrained_model 目录")
            return

    print(f"\n修复 {len(pretrained_dirs)} 个 checkpoint...")

    for ptd in sorted(pretrained_dirs):
        # 修复 preprocessor normalizer
        pre_files = list(ptd.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
        for pre_file in pre_files:
            stats = load_file(str(pre_file))
            old_mean = stats["action.mean"].numpy().copy()
            old_std = stats["action.std"].numpy().copy()

            if dry_run:
                print(f"\n  [DRY RUN] {pre_file.relative_to(ckpt_path)}")
                print(f"    action.mean: {np.array2string(old_mean, precision=4)}")
                print(f"    →            {np.array2string(new_action_stats['mean'], precision=6)}")
            else:
                stats["action.mean"] = torch.from_numpy(new_action_stats["mean"])
                stats["action.std"] = torch.from_numpy(new_action_stats["std"])
                stats["action.min"] = torch.from_numpy(new_action_stats["min"])
                stats["action.max"] = torch.from_numpy(new_action_stats["max"])
                stats["action.count"] = torch.from_numpy(new_action_stats["count"])
                for q_key in ["q01", "q10", "q50", "q90", "q99"]:
                    stats[f"action.{q_key}"] = torch.from_numpy(new_action_stats[q_key])
                save_file(stats, str(pre_file))
                print(f"\n  ✓ {pre_file.relative_to(ckpt_path)}")
                print(f"    action.mean: {np.array2string(old_mean, precision=4)} → "
                      f"{np.array2string(new_action_stats['mean'], precision=6)}")

        # 修复 postprocessor unnormalizer
        post_files = list(ptd.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
        for post_file in post_files:
            stats = load_file(str(post_file))
            old_mean = stats["action.mean"].numpy().copy()

            if dry_run:
                print(f"\n  [DRY RUN] {post_file.relative_to(ckpt_path)}")
                print(f"    action.mean: {np.array2string(old_mean, precision=4)}")
                print(f"    →            {np.array2string(new_action_stats['mean'], precision=6)}")
            else:
                stats["action.mean"] = torch.from_numpy(new_action_stats["mean"])
                stats["action.std"] = torch.from_numpy(new_action_stats["std"])
                stats["action.min"] = torch.from_numpy(new_action_stats["min"])
                stats["action.max"] = torch.from_numpy(new_action_stats["max"])
                stats["action.count"] = torch.from_numpy(new_action_stats["count"])
                for q_key in ["q01", "q10", "q50", "q90", "q99"]:
                    stats[f"action.{q_key}"] = torch.from_numpy(new_action_stats[q_key])
                save_file(stats, str(post_file))
                print(f"  ✓ {post_file.relative_to(ckpt_path)}")
                print(f"    action.mean: {np.array2string(old_mean, precision=4)} → "
                      f"{np.array2string(new_action_stats['mean'], precision=6)}")


def main():
    parser = argparse.ArgumentParser(
        description="修复 Pi0 checkpoint 中动作归一化统计量：绝对→相对")
    parser.add_argument("--input", type=str, required=True,
                        help="原始 npz 数据集目录 (如 ai-models/ur3_pick_place_raw)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint 目录 (如 outputs/train/ur3_pi0_v4_expert)")
    parser.add_argument("--exclude-joints", type=str, nargs="*",
                        default=["gripper"],
                        help="保持绝对动作统计量的关节名 (默认: gripper)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览修改，不实际写入")
    args = parser.parse_args()

    # 验证路径
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        print(f"错误: Checkpoint 目录不存在: {args.checkpoint}")
        sys.exit(1)

    print("=" * 64)
    print("Pi0 相对动作统计量修复")
    print("=" * 64)
    print(f"数据源:   {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"排除关节: {args.exclude_joints}")
    print(f"模式:     {'DRY RUN (预览)' if args.dry_run else '写入修改'}")

    # 1. 计算正确的相对动作统计量
    new_stats = compute_relative_action_stats(args.input, args.exclude_joints)

    # 2. 更新 checkpoint
    fix_checkpoint(checkpoint_dir, new_stats, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n✅ 完成！所有 checkpoint 的动作统计量已更新为相对动作统计量。")
        print(f"   推理脚本现在可以正确反归一化模型输出的相对 delta。")
    else:
        print(f"\n[DRY RUN] 以上是将要写入的修改。去掉 --dry-run 以实际执行。")


if __name__ == "__main__":
    main()
