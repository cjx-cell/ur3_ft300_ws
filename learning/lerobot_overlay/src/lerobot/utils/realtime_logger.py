"""
Real-time Logger & Plotter for PAP-MoE Stage 2 Training.
Tracks:
- Total Loss & Action Loss (Flow Matching MSE)
- PhysicsGate Stage Cross-Entropy Loss & Teacher Forcing Decay (tf_prob)
- Per-Expert Activation Weights (Pred E1..E4 vs GT E1..E4)
- Learning Rate Schedule
Outputs clear log prints and updates real-time dashboard PNG image every N steps.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.labelcolor': '#c9d1d9', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#30363d',
    'text.color': '#c9d1d9', 'axes.titlecolor': '#e6edf3',
})

EXPERT_NAMES = ["E1 Free Motion", "E2 Optical Blind", "E3 Rigid Contact", "E4 Compliant"]
EXPERT_COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff']


class RealtimePAPMoELogger:
    def __init__(self, output_dir: str, save_png_path: str = None, artifact_png_path: str = None):
        self.output_dir = os.path.abspath(output_dir)
        self.save_png_path = os.path.abspath(save_png_path) if save_png_path else os.path.join(self.output_dir, "pap_moe_stage2_realtime_dashboard.png")
        self.artifact_png_path = os.path.abspath(artifact_png_path) if artifact_png_path else None

        self.history = {
            "steps": [],
            "loss_total": [],
            "loss_action": [],
            "loss_stage": [],
            "loss_factor_bcm": [],
            "loss_route_sequence_bcm": [],
            "tf_prob": [],
            "lr": [],
            "pred_E1": [], "pred_E2": [], "pred_E3": [], "pred_E4": [],
            "gt_E1": [], "gt_E2": [], "gt_E3": [], "gt_E4": [],
            "norm_E1": [], "norm_E2": [], "norm_E3": [], "norm_E4": [],
            "loss_expert_representation": [],
            "loss_visual_memory_distillation": [],
            "loss_expert_representation_E1": [], "loss_expert_representation_E2": [],
            "loss_expert_representation_E3": [], "loss_expert_representation_E4": [],
            "optimize_action": [], "optimize_stage": [],
        }

    def update(self, step: int, output_dict: dict, lr: float):
        self.history["steps"].append(step)
        self.history["loss_total"].append(output_dict.get("loss", 0.0))
        # Standard Pi0/Pi0.5 reports its action objective as ``loss`` while
        # PAP-MoE reports the same component as ``loss_action``.
        self.history["loss_action"].append(
            output_dict.get("loss_action", output_dict.get("loss", 0.0))
        )
        self.history["loss_stage"].append(output_dict.get("loss_stage", 0.0))
        self.history["loss_factor_bcm"].append(output_dict.get("loss_factor_bcm", 0.0))
        self.history["loss_route_sequence_bcm"].append(
            output_dict.get("loss_route_sequence_bcm", 0.0)
        )
        self.history["tf_prob"].append(output_dict.get("tf_prob", 1.0))
        self.history["lr"].append(lr)

        for e in range(1, 5):
            self.history[f"pred_E{e}"].append(output_dict.get(f"pred_E{e}", 0.0))
            self.history[f"gt_E{e}"].append(output_dict.get(f"gt_E{e}", 0.0))
            self.history[f"norm_E{e}"].append(output_dict.get(f"norm_E{e}", 0.0))
        for name in (
            "loss_expert_representation",
            "loss_visual_memory_distillation",
            "loss_expert_representation_E1",
            "loss_expert_representation_E2",
            "loss_expert_representation_E3",
            "loss_expert_representation_E4",
            "optimize_action",
            "optimize_stage",
        ):
            self.history[name].append(output_dict.get(name, 0.0))

    def log_and_plot(self, step: int, total_steps: int):
        if len(self.history["steps"]) == 0:
            return

        # ── 1. Text Log Output ────────────────────────────────────────────────
        last_total = self.history["loss_total"][-1]
        last_action = self.history["loss_action"][-1]
        last_stage = self.history["loss_stage"][-1]
        last_factor = self.history["loss_factor_bcm"][-1]
        last_route_sequence = self.history["loss_route_sequence_bcm"][-1]
        last_tf = self.history["tf_prob"][-1]
        last_lr = self.history["lr"][-1]
        action_role = "optimized" if self.history["optimize_action"][-1] else "diagnostic"
        stage_role = "optimized" if self.history["optimize_stage"][-1] else "diagnostic"

        p1, g1 = self.history["pred_E1"][-1], self.history["gt_E1"][-1]
        p2, g2 = self.history["pred_E2"][-1], self.history["gt_E2"][-1]
        p3, g3 = self.history["pred_E3"][-1], self.history["gt_E3"][-1]
        p4, g4 = self.history["pred_E4"][-1], self.history["gt_E4"][-1]
        norms = [self.history[f"norm_E{e}"][-1] for e in range(1, 5)]
        representation_losses = [
            self.history[f"loss_expert_representation_E{e}"][-1]
            for e in range(1, 5)
        ]
        representation_loss = self.history["loss_expert_representation"][-1]
        memory_loss = self.history["loss_visual_memory_distillation"][-1]

        log_str = (
            f"\n{'='*75}\n"
            f"[PAP-MoE Training Step {step}/{total_steps}]\n"
            f"  Optimized Loss: {last_total:.6f}  |  Action Loss ({action_role}): {last_action:.6f}\n"
            f"  PhysicsGate Loss ({stage_role}): {last_stage:.4f}"
            f"  |  b/c/m: {last_factor:.4f}"
            f"  |  Future b/c/m: {last_route_sequence:.4f}"
            f"  |  TF Prob: {last_tf:.3f}"
            f"  |  LR: {last_lr:.2e}\n"
            f"  Expert Activations (Predicted vs Ground-Truth):\n"
            f"    • E1 (Free Motion)  : Pred = {p1:.3f}  |  GT = {g1:.3f}  (Diff: {p1-g1:+.3f})\n"
            f"    • E2 (Optic Blind)  : Pred = {p2:.3f}  |  GT = {g2:.3f}  (Diff: {p2-g2:+.3f})\n"
            f"    • E3 (Rigid Contact): Pred = {p3:.3f}  |  GT = {g3:.3f}  (Diff: {p3-g3:+.3f})\n"
            f"    • E4 (Compliant)    : Pred = {p4:.3f}  |  GT = {g4:.3f}  (Diff: {p4-g4:+.3f})\n"
            f"  Expert Output Norms: E1={norms[0]:.3f} E2={norms[1]:.3f} "
            f"E3={norms[2]:.3f} E4={norms[3]:.3f}\n"
            f"  Representation Loss: total={representation_loss:.4f} "
            f"E1={representation_losses[0]:.4f} E2={representation_losses[1]:.4f} "
            f"E3={representation_losses[2]:.4f} E4={representation_losses[3]:.4f}\n"
            f"  E2 Visual-Memory Distillation: {memory_loss:.4f}\n"
            f"{'='*75}"
        )
        logging.info(log_str)

        # ── 2. Real-time Plotting ──────────────────────────────────────────────
        try:
            self._render_dashboard(step, total_steps)
        except Exception as err:
            logging.warning(f"Failed to render realtime dashboard: {err}")

    def _render_dashboard(self, step: int, total_steps: int):
        steps_all = np.array(self.history["steps"])
        if len(steps_all) == 0:
            return

        # 采样规则：严格每50步提取一个数据点 (50, 100, 150...), 并保留第1步与最新当前步
        mask = (steps_all % 50 == 0) | (steps_all == steps_all[-1]) | (steps_all == 1)
        steps = steps_all[mask]

        fig = plt.figure(figsize=(18, 12), facecolor='#0f1117')
        fig.suptitle(f'PAP-MoE Stage 2 Real-Time Training Dashboard (50-Step Sampling, Step {step} / {total_steps})',
                     fontsize=16, fontweight='bold', color='#e6edf3', y=0.97)

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                               left=0.07, right=0.95, top=0.91, bottom=0.08)

        # ── Subplot 1: Action Loss (Flow Matching MSE) ─────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        act_loss = np.array(self.history["loss_action"])[mask]
        ax1.plot(steps, act_loss, color='#58a6ff', lw=2.2, marker='o', markersize=3.5, label='Action Loss MSE (50-Step Interval)')
        ax1.set_title('Action Flow Matching MSE Loss (Sampling Every 50 Steps)', fontsize=11, pad=6)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend(fontsize=8.5, framealpha=0.3, loc='upper right')

        # ── Subplot 2: Stage Cross Entropy & Teacher Forcing ───────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        stg_loss = np.array(self.history["loss_stage"])[mask]
        tf_prob = np.array(self.history["tf_prob"])[mask]
        ax2.plot(steps, stg_loss, color='#f78166', lw=2.2, marker='o', markersize=3.5, label='PhysicsGate Stage Loss (CE)')
        ax2.set_ylabel('Stage CE Loss', color='#f78166')
        ax2.tick_params(axis='y', labelcolor='#f78166')

        ax2_tf = ax2.twinx()
        ax2_tf.plot(steps, tf_prob, color='#f0e68c', lw=1.8, ls='--', label='Teacher Forcing Prob')
        ax2_tf.set_ylabel('TF Prob (1.0 -> 0.0)', color='#f0e68c')
        ax2_tf.tick_params(axis='y', labelcolor='#f0e68c')
        ax2_tf.set_ylim(-0.05, 1.05)
        ax2.set_title('PhysicsGate Loss & Teacher Forcing Decay (Sampling Every 50 Steps)', fontsize=11, pad=6)

        # ── Subplot 3: Expert Activation Weights vs GT Labels ──────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        for e_idx in range(1, 5):
            c = EXPERT_COLORS[e_idx-1]
            p_val = np.array(self.history[f"pred_E{e_idx}"])[mask]
            g_val = np.array(self.history[f"gt_E{e_idx}"])[mask]

            ax3.plot(steps, p_val, color=c, lw=2.2, marker='o', markersize=3.5, label=f'Pred {EXPERT_NAMES[e_idx-1]}')
            ax3.plot(steps, g_val, color=c, lw=1.5, ls='--', alpha=0.7, label=f'GT {EXPERT_NAMES[e_idx-1]}')

        ax3.set_ylim(-0.05, 1.05)
        ax3.set_title('Expert Activations (Sampling Every 50 Steps: Solid=Pred, Dashed=GT)', fontsize=11, pad=6)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Activation Weight')
        ax3.legend(fontsize=7.5, ncol=2, framealpha=0.3, loc='upper right')

        # ── Subplot 4: Total Loss & Learning Rate Schedule ─────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        total_loss = np.array(self.history["loss_total"])[mask]
        lr_val = np.array(self.history["lr"])[mask]
        ax4.plot(steps, total_loss, color='#3fb950', lw=2.0, marker='o', markersize=3.5, label='Optimized Total Loss')
        ax4.set_ylabel('Total Loss', color='#3fb950')
        ax4.tick_params(axis='y', labelcolor='#3fb950')

        ax4_lr = ax4.twinx()
        ax4_lr.plot(steps, lr_val, color='#d2a8ff', lw=1.8, ls='-.', label='Learning Rate')
        ax4_lr.set_ylabel('Learning Rate', color='#d2a8ff')
        ax4_lr.tick_params(axis='y', labelcolor='#d2a8ff')
        ax4_lr.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax4.set_title('Optimized Loss & Learning Rate Schedule (Sampling Every 50 Steps)', fontsize=11, pad=6)

        os.makedirs(os.path.dirname(self.save_png_path), exist_ok=True)
        plt.savefig(self.save_png_path, dpi=120, bbox_inches='tight', facecolor='#0f1117')
        if self.artifact_png_path:
            os.makedirs(os.path.dirname(self.artifact_png_path), exist_ok=True)
            os.system(f"cp {self.save_png_path} {self.artifact_png_path} 2>/dev/null")
        plt.close(fig)
