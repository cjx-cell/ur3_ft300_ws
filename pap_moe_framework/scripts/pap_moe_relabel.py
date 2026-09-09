#!/usr/bin/env python3
"""
PAP-MoE offline label (re-)generation script.
Reads raw .npz trajectories, applies the v2 physics-aware label function,
and writes updated labels back in-place (or to a new output dir).

Usage:
  # Analyze all trajectories (no modification)
  python3 pap_moe_relabel.py --analyze

  # Relabel all trajectories in-place
  python3 pap_moe_relabel.py --relabel

  # Relabel and create backup
  python3 pap_moe_relabel.py --relabel --backup

  # Analyze + relabel a single episode
  python3 pap_moe_relabel.py --episode 0002 --relabel
"""

import argparse, os, sys, shutil
import numpy as np
from pathlib import Path

RAW_DIR = Path.home() / "ur3_ft300_ws" / "pap_moe_framework" / "datasets" / "raw"

# ══════════════════════════════════════════════════════════════════════════════
# v2.2 Physics-aware label function (extracted from recording script)
# ══════════════════════════════════════════════════════════════════════════════

PEG_EXTEND    = 0.240
RING_TOP_Z    = 0.885
RING_BOTTOM_Z = 0.835

F_NORMAL_RIGID_REF = 3.0
TAU_VAR_RIGID_REF  = 0.01
TAU_FLEX_REF       = 1.5
SIGMOID_STEEPNESS  = 2.0
T_VAR_WINDOW       = 5


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    z = steepness * (x - midpoint)
    if z > 50.0:
        return 1.0
    if z < -50.0:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-z)))


def _rolling_stat(values: list, window: int):
    if len(values) < max(2, window):
        return 0.0, 0.0
    arr = np.array(values[-window:], dtype=np.float64)
    return float(np.mean(arr)), float(np.var(arr))


def calibrate_force_with_payload(force: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    """
    Calibrates force using empty gripper offset at start, and automatically
    detects the 'approach' phase start to capture and compensate the grasped peg's gravity
    after the initial gripper clamping impact spike dies down.
    """
    n_frames = len(force)
    empty_offset = force[0].copy()
    
    approach_indices = []
    if tasks is not None:
        task_strs = [str(t) for t in tasks]
        for idx, task in enumerate(task_strs):
            if "approach" in task:
                approach_indices.append(idx)
                
    if len(approach_indices) > 0:
        start_idx = approach_indices[0]
        # Skip the first 10 frames of approach to avoid the transient gripper clamping impact spike (~130N)
        stable_start = min(start_idx + 10, n_frames - 5)
        stable_end = min(stable_start + 15, n_frames)
        payload_offset = np.mean(force[stable_start:stable_end], axis=0)
        
        force_calibrated = np.zeros_like(force)
        force_calibrated[:start_idx] = force[:start_idx] - empty_offset
        force_calibrated[start_idx:] = force[start_idx:] - payload_offset
        return force_calibrated
    else:
        return force - empty_offset


def compute_labels(
    force: np.ndarray,        # (N, 6)
    tool0_z: np.ndarray,      # (N,)
    img_means: np.ndarray = None,  # (N,) — per-frame camera brightness [0,1]
    gripper_joint: np.ndarray = None, # (N,) — per-frame gripper joint value
    joint_states: np.ndarray = None, # (N, 6) — per-frame joint values
    tasks: np.ndarray = None, # (N,) — per-frame subtask strings
) -> np.ndarray:              # (N, 4)
    """
    Offline label generation. Uses full-trajectory context for rolling statistics.
    Returns (N, 4) soft probability vectors [E1, E2, E3, E4].
    """
    N = len(force)
    stages = np.zeros((N, 4), dtype=np.float32)

    if joint_states is not None and len(joint_states) > 1:
        q_dot = np.gradient(joint_states, axis=0) * 10.0  # 10Hz sampling
        q_dot_norm = np.linalg.norm(q_dot, axis=1)
    else:
        q_dot_norm = np.zeros(N)

    tau_history: list[float] = []
    fz_history: list[float] = []
    _contact_memory = 0.0

    # Baseline image brightness: median of all frames (exclude extremes)
    # Used for relative glare detection — "significantly brighter than normal"
    if img_means is not None and len(img_means) > 10:
        _img_baseline = float(np.median(img_means))
    else:
        _img_baseline = 0.5

    # Calculate force derivative (df/dt) for contact impulse spike detection
    df_dt = np.gradient(force, axis=0) if len(force) > 1 else np.zeros_like(force)

    for i in range(N):
        Fx, Fy, Fz = float(force[i, 0]), float(force[i, 1]), float(force[i, 2])
        Tx, Ty, Tz = float(force[i, 3]), float(force[i, 4]), float(force[i, 5])

        f_mag   = float(np.sqrt(Fx**2 + Fy**2 + Fz**2))
        tau_mag = float(np.sqrt(Tx**2 + Ty**2 + Tz**2))

        # ── Optimization 3: Force Derivative Impulse Spike (dF/dt) ───
        # Detects millisecond-level contact impact spikes even if force magnitude is small
        dFx, dFy, dFz = float(df_dt[i, 0]), float(df_dt[i, 1]), float(df_dt[i, 2])
        df_mag = float(np.sqrt(dFx**2 + dFy**2 + dFz**2))
        SIGMA_DF = 4.0  # N/step derivative threshold
        impulse_spike = float(1.0 - np.exp(- (df_mag ** 2) / (2 * (SIGMA_DF ** 2))))

        peg_tip_z = float(tool0_z[i] - PEG_EXTEND) if tool0_z[i] > 0 else RING_TOP_Z + 0.050

        # Rolling buffers
        tau_history.append(tau_mag)
        fz_history.append(Fz)
        if len(tau_history) > T_VAR_WINDOW * 4:
            tau_history.pop(0)
            fz_history.pop(0)

        tau_mean, tau_var = _rolling_stat(tau_history, T_VAR_WINDOW)
        fz_mean,  _       = _rolling_stat(fz_history,  T_VAR_WINDOW)

        # ── E2: Optical Blind Zone ─────────────────────────────────────
        if img_means is not None:
            b = float(img_means[i])
            if b < 0.02:                      # dropout → fully black
                visual_loss = 1.0
            elif b > 0.50 and b > _img_baseline * 1.40:  # glare: 40%+ brighter than baseline
                visual_loss = float(np.clip((b / _img_baseline - 1.40) / 1.10, 0.0, 1.0))
            else:
                visual_loss = 0.0             # normal image → no blind
        else:
            visual_loss = 0.0

        # ── Optimization 2: Noise-Normalized Mahalanobis Distance ─────
        # Calculates 6D noise-normalized Mahalanobis distance for universal task adaptation
        SIGMA_F_VEC = np.array([8.0, 8.0, 8.0], dtype=np.float64)
        SIGMA_T_VEC = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        mahalanobis_sq = float(np.sum((np.array([Fx, Fy, Fz]) / SIGMA_F_VEC)**2) + 
                              np.sum((np.array([Tx, Ty, Tz]) / SIGMA_T_VEC)**2))
        
        f_free_score = float(np.exp(- 0.5 * mahalanobis_sq))
        
        # Free motion score (vision active + no contact force)
        w_free = float((1.0 - visual_loss) * f_free_score)
        
        # Optical blind zone score (vision lost in free space)
        w_blind = float(visual_loss * f_free_score)

        # ── Contact Score ──────────────────────────────────────────────
        # Contact score depends PURELY on FT300 force sensors, NOT suppressed by visual loss!
        raw_contact_score = float(max(1.0 - f_free_score, impulse_spike))
        contact_score = float(raw_contact_score)
        contact_score = max(0.0, min(1.0, contact_score))

        # ── E3: Rigid Micro-Constraint ─────────────────────────────────
        # Stick-slip signal based on high-frequency torque variance
        stick_slip_signal = _sigmoid(tau_var, TAU_VAR_RIGID_REF, SIGMOID_STEEPNESS / 0.005)
        w_rigid = float(contact_score * stick_slip_signal)

        # ── Contact Memory Hysteresis ──────────────────────────────────
        # Prevents E1 from spiking during brief relief-lifts in spiral search.
        # Fast decay when truly free (forces/torques within background noise).
        # Slow decay near ring (relief lift during search).
        if w_rigid > 0.3:
            _contact_memory = min(1.0, _contact_memory + 0.20)
        elif f_mag < 2.0 and tau_mag < 0.4:
            _contact_memory *= 0.85  # fast decay: truly free transport
        else:
            _contact_memory *= 0.95  # slow decay: relief lift near ring
        _contact_memory = max(0.0, min(1.0, _contact_memory))

        # ── E4: Flexible / Fragile Contact ─────────────────────────────
        # Complementary to E3: smooth contact without high-frequency vibration
        w_flexible = float(contact_score * (1.0 - stick_slip_signal))

        # Gripper-based state gate on E4 (E4 active only when holding the peg)
        # robotiq_85_left_knuckle_joint > 0.3 means gripper is closed and holding
        gripper_joint_val = float(gripper_joint[i]) if gripper_joint is not None else 0.5
        if gripper_joint_val < 0.3:
            w_flexible = 0.0

        # Suppress contact experts (E3, E4) at high velocities to prevent inertial activation
        vel_suppression = float(np.exp(-q_dot_norm[i] / 0.04))
        w_rigid = w_rigid * vel_suppression
        w_flexible = w_flexible * vel_suppression

        # ── Assemble & normalize ───────────────────────────────────────
        w_vector = np.array([w_free, w_blind, w_rigid, w_flexible], dtype=np.float32)
        sum_w = float(w_vector.sum())
        if sum_w > 1e-8:
            w_vector = w_vector / sum_w
        else:
            # Fallback logic: if under contact force, default to E4 (compliance); else E1 (free space)
            if f_mag > 4.0 or tau_mag > 0.8:
                w_vector[3] = 1.0
            else:
                w_vector[0] = 1.0

        # Post-normalization hysteresis: E1→E3 transfer during contact memory
        if _contact_memory > 0.3 and w_vector[0] > 0.25:
            transfer = float(w_vector[0] * min(0.95, _contact_memory * 1.10))
            w_vector[0] -= transfer
            w_vector[2] += transfer

        stages[i] = w_vector

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = ["E1:Free/Load", "E2:Blind", "E3:Rigid", "E4:Flexible"]


def analyze_episode(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    force = d["force"]
    n = len(force)
    f_mag = np.linalg.norm(force[:, :3], axis=1)
    tau_mag = np.linalg.norm(force[:, 3:], axis=1)

    # Load labels — check v2 cache first, then compute, then fall back to in-npz
    stage_v2_path = str(npz_path).replace(".npz", "_stage_v2.npy")
    has_tool0 = "tool0_z" in d.files

    # Always load tool0_z for phase analysis (separate from label loading)
    tool0_z = d["tool0_z"] if has_tool0 else None

    if os.path.exists(stage_v2_path):
        stages = np.load(stage_v2_path)
        label_source = "v2 (cached _stage_v2.npy)"
    elif has_tool0:
        cam0 = d.get("camera0", None)
        img_means = cam0.mean(axis=(1,2,3)) if cam0 is not None else None
        state = d.get("state", None)
        gripper_joint = state[:, 6] if state is not None and state.shape[1] > 6 else None
        joint_states = state[:, :6] if state is not None else None
        tasks = d.get("task", None)
        force_calibrated = calibrate_force_with_payload(force, tasks)
        stages = compute_labels(force_calibrated, tool0_z, img_means, gripper_joint, joint_states=joint_states, tasks=tasks)
        label_source = "v2 (computed on-the-fly)"
    else:
        stages = d["stage"]
        if stages.ndim == 2 and stages.shape[1] == 4:
            label_source = "v1 (from recording, 4D)"
        else:
            label_source = "legacy (unknown format)"

    stage_mean = stages.mean(axis=0)
    dom = np.argmax(stages, axis=1)
    dom_counts = [(dom == i).sum() for i in range(4)]

    # Segment analysis
    transitions = int((dom[:-1] != dom[1:]).sum())
    peg_tip_z = tool0_z - PEG_EXTEND if tool0_z is not None else None

    # Phase analysis (if tool0_z available)
    phases = {}
    if has_tool0:
        # Transport: peg above ring
        transport_mask = peg_tip_z > RING_TOP_Z
        # Approach: peg within 2cm above ring
        approach_mask = (peg_tip_z <= RING_TOP_Z) & (peg_tip_z > RING_TOP_Z - 0.020)
        # Contact/search: peg within ring zone
        contact_mask = (peg_tip_z <= RING_TOP_Z) & (peg_tip_z >= RING_BOTTOM_Z)
        # Deep: peg below ring
        deep_mask = peg_tip_z < RING_BOTTOM_Z

        for name, mask in [("Transport", transport_mask), ("Approach", approach_mask),
                           ("Contact/Search", contact_mask), ("Deep/Below ring", deep_mask)]:
            if mask.sum() > 0:
                phases[name] = {
                    "frames": int(mask.sum()),
                    "pct": float(mask.sum() / n * 100),
                    "avg_stage": stages[mask].mean(axis=0),
                    "dom": [(stages[mask].argmax(axis=1) == i).sum() for i in range(4)],
                }

    return {
        "n": n, "f_mag": f_mag, "tau_mag": tau_mag,
        "stage_mean": stage_mean, "dom_counts": dom_counts,
        "transitions": transitions, "label_source": label_source,
        "has_tool0": has_tool0, "phases": phases,
        "tool0_z": tool0_z, "peg_tip_z": peg_tip_z,
    }


def print_analysis(npz_path: str, info: dict):
    ep_name = Path(npz_path).parent.name
    n = info["n"]
    fm, tm = info["f_mag"], info["tau_mag"]
    sm = info["stage_mean"]

    print(f"\n{'='*70}")
    print(f"  {ep_name}  ({n} frames)  [{info['label_source']}]")
    print(f"{'='*70}")
    print(f"  |F|:  min={fm.min():6.1f}  max={fm.max():6.1f}  mean={fm.mean():6.1f} N")
    print(f"  |T|:  min={tm.min():5.2f}  max={tm.max():5.2f}  mean={tm.mean():5.3f} Nm")
    print(f"  Stage avg:  E1={sm[0]:.3f}  E2={sm[1]:.3f}  E3={sm[2]:.3f}  E4={sm[3]:.3f}")
    print(f"  Dominant:   E1={info['dom_counts'][0]:>4}  E2={info['dom_counts'][1]:>4}  "
          f"E3={info['dom_counts'][2]:>4}  E4={info['dom_counts'][3]:>4}")
    print(f"  Transitions: {info['transitions']} (every {n/max(1,info['transitions']):.1f} frames)")

    # Phase breakdown
    if info["phases"]:
        print(f"\n  {'Phase':<18} {'Frames':>6} {'%':>5}  {'E1':>6} {'E2':>6} {'E3':>6} {'E4':>6}")
        print(f"  {'─'*18} {'─'*6} {'─'*5}  {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for name, p in info["phases"].items():
            s = p["avg_stage"]
            print(f"  {name:<18} {p['frames']:>6} {p['pct']:>4.0f}%  "
                  f"{s[0]:>6.3f} {s[1]:>6.3f} {s[2]:>6.3f} {s[3]:>6.3f}")

    # Warnings
    if info["transitions"] > n * 0.15:
        print(f"  ⚠ High label churn: {info['transitions']} transitions in {n} frames")
    if not info["has_tool0"]:
        print(f"  ⚠ No tool0_z in data — cannot compute v2 labels!")
    if sm[1] > 0.10:
        print(f"  ⚠ E2 > 10% avg — verify visual degradation in source images")


def relabel_episode(npz_path: str, backup: bool = False) -> bool:
    d = np.load(npz_path, allow_pickle=True)
    if "tool0_z" not in d.files:
        print(f"  ✗ Cannot relabel: no tool0_z in {npz_path}")
        return False

    force = d["force"]
    tool0_z = d["tool0_z"]
    cam0 = d.get("camera0", None)
    img_means = cam0.mean(axis=(1,2,3)) if cam0 is not None else None
    state = d.get("state", None)
    gripper_joint = state[:, 6] if state is not None and state.shape[1] > 6 else None

    joint_states = state[:, :6] if state is not None else None
    tasks = d.get("task", None)
    force_calibrated = calibrate_force_with_payload(force, tasks)
    stages_new = compute_labels(force_calibrated, tool0_z, img_means, gripper_joint, joint_states=joint_states, tasks=tasks)

    if backup:
        backup_path = npz_path + ".v1_backup"
        shutil.copy2(npz_path, backup_path)
        print(f"  Backup: {backup_path}")

    # Save only the stage array (avoids re-compressing 100+ MB of images)
    stage_path = str(npz_path).replace(".npz", "_stage_v2.npy")
    np.save(stage_path, stages_new)
    print(f"  ✓ Relabeled: {stage_path}  ({len(stages_new)} frames, {os.path.getsize(stage_path)/1024:.0f} KB)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PAP-MoE offline label (re-)generation")
    parser.add_argument("--analyze", action="store_true", help="Analyze label distributions")
    parser.add_argument("--relabel", action="store_true", help="Regenerate labels with v2 function")
    parser.add_argument("--backup", action="store_true", help="Create .v1_backup before relabel")
    parser.add_argument("--episode", type=str, default=None, help="Episode substring to match")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found"); sys.exit(1)

    # Find episodes
    npz_files = []
    for ep_dir in sorted(raw_dir.glob("*_episode_*")):
        npz = ep_dir / "data.npz"
        if npz.exists():
            if args.episode is None or args.episode in ep_dir.name:
                npz_files.append(npz)

    if not npz_files:
        print("No episodes found."); return

    print(f"Found {len(npz_files)} episode(s)")

    if args.relabel:
        ok = 0
        for npz_path in npz_files:
            if relabel_episode(str(npz_path), args.backup):
                ok += 1
        print(f"\nRelabeled {ok}/{len(npz_files)} episodes")

    if args.analyze or not args.relabel:
        for npz_path in npz_files:
            info = analyze_episode(str(npz_path))
            print_analysis(str(npz_path), info)

    # Summary if multiple
    if len(npz_files) > 1 and args.analyze:
        print(f"\n{'='*70}")
        print(f"  SUMMARY ({len(npz_files)} episodes)")
        print(f"{'='*70}")
        all_stages = []
        for npz_path in npz_files:
            info = analyze_episode(str(npz_path))
            all_stages.append(info["stage_mean"])
        avg = np.mean(all_stages, axis=0)
        print(f"  Mean across episodes: E1={avg[0]:.3f} E2={avg[1]:.3f} E3={avg[2]:.3f} E4={avg[3]:.3f}")


if __name__ == "__main__":
    main()
