#!/usr/bin/env python3
"""
将 ur3_collect_dataset.py 生成的 npz 文件转换为 LeRobot v3.0 格式。

用法 (pi0-env):
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 ur3_convert_to_lerobot.py --input ~/ur3_dataset --repo_id my_ur3_data
"""

import argparse, os, json, time
import numpy as np
from pathlib import Path

# LeRobot 需要在 pi0-env 中
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
except ImportError as e:
    print(f"ERROR: LeRobot 未安装 ({e})")
    print("请在 pi0-env 中运行: source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env")
    sys.exit(1)


def find_episodes(input_dir):
    """扫描所有 episode_XXXX/data.npz 文件"""
    episodes = []
    for d in sorted(os.listdir(input_dir)):
        ep_dir = os.path.join(input_dir, d)
        if d.startswith("episode_") and os.path.isdir(ep_dir):
            npz_path = os.path.join(ep_dir, "data.npz")
            if os.path.exists(npz_path):
                episodes.append(npz_path)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="npz → LeRobot 格式转换")
    parser.add_argument("--input", type=str, required=True, help="npz 数据集目录")
    parser.add_argument("--repo_id", type=str, default="cjx-cell/ur3_pick_place",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="本地输出目录 (默认同时保存到 ~/ur3_ft300_ws/ai-models/ur3_pick_place_lerobot)")
    parser.add_argument("--push_to_hub", action="store_true", help="推送到 HuggingFace Hub")
    args = parser.parse_args()

    input_dir = Path(args.input)
    episodes = find_episodes(input_dir)
    print(f"找到 {len(episodes)} 个 episodes")

    if not episodes:
        print("没有数据！先运行 ur3_collect_dataset.py")
        return

    # 读取第一个 episode 获取特征信息
    first = np.load(episodes[0])
    state_shape = first["state"].shape[1]   # (N_steps, state_dim)
    action_shape = first["action"].shape[1]
    cam0_shape = first["camera0"].shape[1:]  # (224, 224, 3)
    cam1_shape = first["camera1"].shape[1:]

    print(f"State dim: {state_shape}, Action dim: {action_shape}")
    print(f"Camera0: {cam0_shape}, Camera1: {cam1_shape}")

    # 生成关节名称
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "robotiq_85_left_knuckle_joint",
    ]

    # 创建 LeRobot 数据集
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": joint_names if action_shape == 6 else [f"joint_{i}" for i in range(action_shape)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_shape,),
            "names": joint_names if state_shape == 6 else [f"joint_{i}" for i in range(state_shape)],
        },
        "observation.images.camera0": {
            "dtype": "video",
            "shape": tuple(cam0_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": tuple(cam1_shape),
            "names": ["height", "width", "channels"],
        },
    }

    print(f"创建 LeRobot 数据集: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="ur3",
        use_videos=True,
    )

    # 逐 episode 添加帧
    total_frames = 0
    t0 = time.time()
    for ep_idx, npz_path in enumerate(episodes):
        data = np.load(npz_path)
        states = data["state"]
        actions = data["action"]
        cam0 = data["camera0"]
        cam1 = data["camera1"]

        n_frames = len(states)
        task_text = str(data.get("task", "pick up the red cube and place it into the bowl"))

        for i in range(n_frames):
            frame = {
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "observation.images.camera0": (cam0[i] * 255).astype(np.uint8),
                "observation.images.camera1": (cam1[i] * 255).astype(np.uint8),
                "task": task_text,
            }
            dataset.add_frame(frame)
            total_frames += 1

        dataset.save_episode()
        elapsed = time.time() - t0
        print(f"  Episode {ep_idx+1}/{len(episodes)}: {n_frames} 帧 "
              f"({elapsed:.1f}s, {total_frames} 总计)")

    # 完成
    dataset.finalize()
    print(f"\n数据集已创建: {total_frames} 帧, {len(episodes)} episodes, "
          f"耗时 {time.time()-t0:.1f}s")

    stats = getattr(dataset, "stats", None)
    if not stats:
        try:
            stats = dataset.meta.stats
        except Exception:
            stats = None
    if stats:
        print(f"\n归一化统计:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}: mean={v.get('mean', 'N/A')}, std={v.get('std', 'N/A')}")

    output_path = dataset.root
    print(f"\n数据集路径: {output_path}")

    # 复制到本地目录
    import shutil
    local_dir = args.output_dir or os.path.expanduser(
        "~/ur3_ft300_ws/ai-models/ur3_pick_place_lerobot")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    shutil.copytree(output_path, local_dir)
    print(f"已复制到: {local_dir}")

    if args.push_to_hub:
        print("推送到 HuggingFace Hub...")
        dataset.push_to_hub()
        print(f"已推送: https://huggingface.co/datasets/{args.repo_id}")

    print("\n下一步：微调 Pi0")
    print(f"  python -m lerobot.scripts.lerobot_train \\")
    print(f"    --policy.path=lerobot/pi0 \\")
    print(f"    --dataset.repo_id={args.repo_id}")


if __name__ == "__main__":
    main()
