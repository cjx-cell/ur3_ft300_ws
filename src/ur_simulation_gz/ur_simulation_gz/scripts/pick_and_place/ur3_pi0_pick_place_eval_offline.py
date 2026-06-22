#!/usr/bin/env python3
"""
Evaluate PI0 V2 full fine-tune model on pick-and-place dataset.

Usage:
    conda activate pi0-env
    python eval_pi0_v2_full.py

Computes:
  1. Per-joint MAE (rad) on 100 random frames
  2. "Predict zero delta" baseline MAE (action = current state)
  3. "Predict mean action" baseline MAE
  4. Vision encoder loading diagnostics
"""

import json
import sys
import time
import torch
import numpy as np

from pathlib import Path
from safetensors.torch import load_file

# ── Paths ──
MODEL_DIR = Path("/home/ubuntu/ur3_ft300_ws/ai-models/pi0/pi0_full_v2/checkpoints/084000/pretrained_model")
DATASET_ROOT = Path("/home/ubuntu/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_lerobot")
TOKENIZER_PATH = Path("/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer")
TASK = "pick up the red cube and place it into the bowl\n"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]

N_FRAMES = 100
SEED = 42

# ── Helper ──
def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def format_arr(arr, fmt=".4f"):
    """Format a numpy array as a compact string."""
    return "[" + ", ".join(f"{v:{fmt}}" for v in arr) + "]"


def main():
    print_section("PI0 V2 Full Fine-Tune Evaluation")
    print(f"Model : {MODEL_DIR}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Frames : {N_FRAMES}  |  Seed: {SEED}")
    print(f"Task   : {TASK.strip()}")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Load normalization stats from preprocessor/postprocessor
    # ─────────────────────────────────────────────────────────────
    print_section("Step 1: Loading normalization stats")

    preprocessor_safetensors = MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    postprocessor_safetensors = MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    if not preprocessor_safetensors.exists():
        print(f"ERROR: Preprocessor safetensors not found at {preprocessor_safetensors}")
        sys.exit(1)
    if not postprocessor_safetensors.exists():
        print(f"ERROR: Postprocessor safetensors not found at {postprocessor_safetensors}")
        sys.exit(1)

    pre_stats = load_file(str(preprocessor_safetensors))
    post_stats = load_file(str(postprocessor_safetensors))

    state_mean = pre_stats["observation.state.mean"].numpy().astype(np.float32)
    state_std  = pre_stats["observation.state.std"].numpy().astype(np.float32)
    action_mean = post_stats["action.mean"].numpy().astype(np.float32)
    action_std  = post_stats["action.std"].numpy().astype(np.float32)

    eps = 1e-8  # from preprocessor config

    print(f"  state_mean : {format_arr(state_mean)}")
    print(f"  state_std  : {format_arr(state_std)}")
    print(f"  action_mean: {format_arr(action_mean)}")
    print(f"  action_std : {format_arr(action_std)}")

    # Warn about near-zero std (wrist_2_joint especially)
    for i, name in enumerate(JOINT_NAMES):
        if state_std[i] < 0.01:
            print(f"  ⚠  {name}: state_std={state_std[i]:.6f} (very small — may cause instability)")
        if action_std[i] < 0.01:
            print(f"  ⚠  {name}: action_std={action_std[i]:.6f} (very small — may cause instability)")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Load model
    # ─────────────────────────────────────────────────────────────
    print_section("Step 2: Loading PI0 model")

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from transformers import AutoTokenizer

    # Load config
    with open(MODEL_DIR / "config.json") as f:
        raw_config = json.load(f)

    print(f"  Config type: {raw_config.get('type', 'N/A')}")
    print(f"  n_obs_steps: {raw_config.get('n_obs_steps', 'N/A')}")
    print(f"  chunk_size : {raw_config.get('chunk_size', 'N/A')}")
    print(f"  dtype      : {raw_config.get('dtype', 'N/A')}")
    print(f"  pretrained : {raw_config.get('pretrained_path', 'N/A')}")
    print(f"  freeze_vision_encoder: {raw_config.get('freeze_vision_encoder', 'N/A')}")

    input_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in raw_config["input_features"].items()
    }
    output_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in raw_config["output_features"].items()
    }

    config = PI0Config(
        device="cpu",
        dtype="bfloat16",
        empty_cameras=0,
        input_features=input_features,
        output_features=output_features,
    )

    # Set device to "meta" for memory-efficient loading
    config.device = "meta"

    # Load state dict
    sd = load_file(str(MODEL_DIR / "model.safetensors"), device="cpu")
    for k in list(sd.keys()):
        if sd[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
            sd[k] = sd[k].to(dtype=torch.bfloat16)

    print(f"  State dict keys: {len(sd)}")

    # Count vision-related keys
    vision_keys = [k for k in sd.keys() if "vision" in k.lower() or "siglip" in k.lower() or "paligemma" in k.lower()]
    print(f"  Vision-related keys: {len(vision_keys)}")
    if vision_keys:
        # Show a few examples
        for k in vision_keys[:5]:
            print(f"    - {k}: shape={list(sd[k].shape)}")
    else:
        print(f"  ⚠  WARNING: No vision-related keys found in state dict!")
        print(f"  First 10 keys:")
        for k in list(sd.keys())[:10]:
            print(f"    - {k}: shape={list(sd[k].shape)}")

    # Load model with init_empty_weights (memory efficient)
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            policy = PI0Policy(config)
        policy.load_state_dict(sd, strict=False, assign=True)
        policy = policy.to_empty(device="cuda")
        print(f"  Model loaded (init_empty_weights → to_empty to cuda)")
    except ImportError:
        print(f"  accelerate not available, loading on CPU then moving to CUDA...")
        config.device = "cpu"
        policy = PI0Policy(config)
        policy.load_state_dict(sd, strict=False)
        policy = policy.to("cuda")
        print(f"  Model loaded (cpu → cuda)")

    policy.eval()
    torch.set_grad_enabled(False)
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory allocated: {gpu_mem:.2f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    print(f"  Tokenizer loaded from {TOKENIZER_PATH}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2b: Vision encoder diagnostics
    # ─────────────────────────────────────────────────────────────
    print_section("Step 2b: Vision encoder diagnostics")

    # Check if the model has vision tower parameters
    vision_param_count = 0
    total_param_count = 0
    for name, param in policy.named_parameters():
        total_param_count += param.numel()
        if any(x in name.lower() for x in ["vision", "siglip", "embedding"]):
            vision_param_count += param.numel()

    print(f"  Total parameters : {total_param_count:,}")
    print(f"  Vision-related params: {vision_param_count:,} ({100*vision_param_count/max(1,total_param_count):.1f}%)")

    # Check if vision params have gradients or are frozen
    vision_trainable = 0
    for name, param in policy.named_parameters():
        if any(x in name.lower() for x in ["vision", "siglip"]):
            if param.requires_grad:
                vision_trainable += param.numel()
    print(f"  Vision trainable params: {vision_trainable:,}")

    # Show specific vision layer info
    if vision_keys:
        sample_vision_key = vision_keys[0]
        # Find corresponding param in model
        model_key = sample_vision_key
        if not model_key.startswith("model."):
            model_key = f"model.{sample_vision_key}"
        found = False
        for name, _ in policy.named_parameters():
            if model_key in name or sample_vision_key in name:
                print(f"  ✓ Found vision param in model: {name}")
                found = True
                break
        if not found:
            print(f"  ⚠  WARNING: Vision key '{sample_vision_key}' not found in model parameters!")
            print(f"  This may indicate the vision encoder did NOT load properly.")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Load dataset
    # ─────────────────────────────────────────────────────────────
    print_section("Step 3: Loading dataset")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id="local/ur3_pick_place",
        root=str(DATASET_ROOT),
        video_backend="pyav",
    )
    print(f"  Frames   : {len(ds)}")
    print(f"  Episodes : {ds.meta.total_episodes}")

    # Verify sample shapes
    sample = ds[0]
    state_shape = tuple(sample["observation.state"].shape)
    action_shape = tuple(sample["action"].shape)
    cam0_shape = tuple(sample["observation.images.camera0"].shape)
    cam1_shape = tuple(sample["observation.images.camera1"].shape)
    print(f"  state shape : {state_shape}")
    print(f"  action shape: {action_shape}")
    print(f"  cam0 shape  : {cam0_shape}")
    print(f"  cam1 shape  : {cam1_shape}")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Evaluate on 100 random frames
    # ─────────────────────────────────────────────────────────────
    print_section("Step 4: Model evaluation on 100 random frames")

    rng = np.random.RandomState(SEED)
    n_total = len(ds)
    indices = sorted(rng.choice(n_total, size=N_FRAMES, replace=False).tolist())

    # Pre-tokenize the task string
    tokens = tokenizer(
        TASK,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=48,
    )
    lang_tokens = tokens["input_ids"]  # shape (1, 48)
    lang_mask = tokens["attention_mask"].bool()  # shape (1, 48)

    all_errors = []       # list of per-joint abs errors for each frame
    all_gt = []           # list of ground truth actions
    all_pred = []         # list of predictions
    all_states = []       # list of observation states

    print(f"  Evaluating {len(indices)} frames...")
    t_start = time.time()

    for i, idx in enumerate(indices):
        frame = ds[idx]

        # Ground truth
        state_raw = frame["observation.state"].numpy().astype(np.float32)
        gt_action = frame["action"].numpy().astype(np.float32)

        # Normalize state
        state_norm = (state_raw - state_mean) / (state_std + eps)
        state_t = torch.from_numpy(state_norm).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

        # Images (dataset already returns [C, H, W])
        cam0 = frame["observation.images.camera0"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        cam1 = frame["observation.images.camera1"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

        # Build batch
        batch = {
            "observation.state": state_t,
            "observation.images.camera0": cam0,
            "observation.images.camera1": cam1,
            OBS_LANGUAGE_TOKENS: lang_tokens.to("cuda"),
            OBS_LANGUAGE_ATTENTION_MASK: lang_mask.to("cuda"),
        }

        # Run inference
        policy.reset()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                action_chunk = policy.predict_action_chunk(batch)
                pred_norm = action_chunk[:, 0, :]  # first predicted action

        pred_np = pred_norm.cpu().float().numpy().flatten()

        # Unnormalize prediction
        pred_unnorm = pred_np * action_std + action_mean

        # Compute per-joint absolute error
        abs_err = np.abs(pred_unnorm - gt_action)

        all_errors.append(abs_err)
        all_gt.append(gt_action)
        all_pred.append(pred_unnorm)
        all_states.append(state_raw)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"    {i+1:3d}/{N_FRAMES}  |  avg MAE so far: {np.mean([e.mean() for e in all_errors]):.4f} rad  |  {elapsed:.1f}s")

    elapsed = time.time() - t_start
    print(f"  Done. {N_FRAMES} frames in {elapsed:.1f}s ({elapsed/N_FRAMES:.2f}s/frame)")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Per-joint MAE
    # ─────────────────────────────────────────────────────────────
    print_section("Step 5: Per-joint MAE results (model predictions)")

    per_joint_mae = np.array(all_errors)  # shape: (N_FRAMES, 7)

    print(f"\n  {'Joint':<20s} {'Mean MAE':>10s} {'Median MAE':>10s} {'Max MAE':>10s} {'Std':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for j in range(7):
        j_err = per_joint_mae[:, j]
        print(f"  {JOINT_NAMES[j]:<20s} {j_err.mean():>10.4f} {np.median(j_err):>10.4f} "
              f"{j_err.max():>10.4f} {j_err.std():>10.4f}")

    overall_mae = per_joint_mae.mean()
    overall_median = np.median(per_joint_mae)
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'OVERALL':<20s} {overall_mae:>10.4f} {overall_median:>10.4f} "
          f"{per_joint_mae.max():>10.4f} {per_joint_mae.std():>10.4f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: "Predict zero delta" baseline (action = current state)
    # ─────────────────────────────────────────────────────────────
    print_section("Step 6: Zero-delta baseline (action = current state)")

    all_states_arr = np.array(all_states)  # (N_FRAMES, 7)
    all_gt_arr = np.array(all_gt)          # (N_FRAMES, 7)

    zero_delta_errors = np.abs(all_states_arr - all_gt_arr)  # (N_FRAMES, 7)

    print(f"\n  {'Joint':<20s} {'Mean MAE':>10s} {'Median MAE':>10s} {'Max MAE':>10s} {'Std':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for j in range(7):
        j_err = zero_delta_errors[:, j]
        print(f"  {JOINT_NAMES[j]:<20s} {j_err.mean():>10.4f} {np.median(j_err):>10.4f} "
              f"{j_err.max():>10.4f} {j_err.std():>10.4f}")

    zd_overall_mae = zero_delta_errors.mean()
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'OVERALL':<20s} {zd_overall_mae:>10.4f} {np.median(zero_delta_errors):>10.4f} "
          f"{zero_delta_errors.max():>10.4f} {zero_delta_errors.std():>10.4f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: "Predict mean" baseline (action = dataset mean action)
    # ─────────────────────────────────────────────────────────────
    print_section("Step 7: Predict-mean baseline (action = dataset mean)")

    mean_action = action_mean  # already loaded from stats
    mean_errors = np.abs(mean_action - all_gt_arr)

    print(f"\n  {'Joint':<20s} {'Mean MAE':>10s} {'Median MAE':>10s} {'Max MAE':>10s} {'Std':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for j in range(7):
        j_err = mean_errors[:, j]
        print(f"  {JOINT_NAMES[j]:<20s} {j_err.mean():>10.4f} {np.median(j_err):>10.4f} "
              f"{j_err.max():>10.4f} {j_err.std():>10.4f}")

    mean_overall_mae = mean_errors.mean()
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'OVERALL':<20s} {mean_overall_mae:>10.4f} {np.median(mean_errors):>10.4f} "
          f"{mean_errors.max():>10.4f} {mean_errors.std():>10.4f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 8: Summary / Comparison
    # ─────────────────────────────────────────────────────────────
    print_section("Step 8: Summary comparison")

    print(f"\n  {'Metric':<30s} {'Overall MAE (rad)':>18s}")
    print(f"  {'-'*30} {'-'*18}")
    print(f"  {'Model prediction':<30s} {overall_mae:>18.4f}")
    print(f"  {'Zero-delta baseline':<30s} {zd_overall_mae:>18.4f}")
    print(f"  {'Predict-mean baseline':<30s} {mean_overall_mae:>18.4f}")

    improvement_over_zd = (1 - overall_mae / zd_overall_mae) * 100 if zd_overall_mae > 0 else 0
    improvement_over_mean = (1 - overall_mae / mean_overall_mae) * 100 if mean_overall_mae > 0 else 0
    print(f"\n  Model vs zero-delta  : {improvement_over_zd:+.1f}% ({'better' if overall_mae < zd_overall_mae else 'worse'})")
    print(f"  Model vs predict-mean: {improvement_over_mean:+.1f}% ({'better' if overall_mae < mean_overall_mae else 'worse'})")

    # ─────────────────────────────────────────────────────────────
    # STEP 9: Summary verdict
    # ─────────────────────────────────────────────────────────────
    print_section("Step 9: Verdict")

    print(f"\n  Model path      : {MODEL_DIR}")
    print(f"  Dataset          : {ds.meta.total_episodes} episodes, {len(ds)} frames")
    print(f"  Eval frames      : {N_FRAMES}")
    print(f"  Overall MAE      : {overall_mae:.4f} rad")
    print(f"  Loss=0.006 check : {'⚠ MATCH: MAE is large despite low training loss' if overall_mae > 0.1 else 'Loss and MAE are consistent'}")

    if overall_mae < 0.05:
        print(f"\n  +++ Model is excellent (MAE < 0.05 rad)")
    elif overall_mae < 0.10:
        print(f"\n  ++  Model is good (MAE < 0.10 rad)")
    elif overall_mae < 0.20:
        print(f"\n  +   Model is moderate (MAE < 0.20 rad)")
    elif overall_mae < 0.35:
        print(f"\n  ~   Model is borderline (MAE ~ {overall_mae:.2f} rad)")
    else:
        print(f"\n  --- Model is poor (MAE > 0.35 rad)")

    # Also report per-joint verdict
    print(f"\n  Per-joint breakdown:")
    for j in range(7):
        j_mae = per_joint_mae[:, j].mean()
        zd_mae = zero_delta_errors[:, j].mean()
        mean_mae = mean_errors[:, j].mean()
        beats_zd = "✓" if j_mae < zd_mae else "✗"
        beats_mean = "✓" if j_mae < mean_mae else "✗"
        print(f"    {JOINT_NAMES[j]:<20s} model={j_mae:.4f}  zero_delta={zd_mae:.4f}  "
              f"pred_mean={mean_mae:.4f}  >zd:{beats_zd}  >mean:{beats_mean}")

    print(f"\n  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print_section("Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
