"""Pure Pi0.5 / predicted-PhysicsGate candidate, fixed 50/10 RTC contract.

No ROS control, recovery, route overrides, action scaling or controller changes.
Inputs are immutable saved observation traces, not unchecked live /tmp files.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from action_contract import clip_physical_gripper

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0, '/home/ubuntu/lerobot/src')
sys.path.insert(0, str(ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'))


class ResidentBackend:
    def __init__(self, kind, checkpoint, device='cuda'):
        import lerobot.policies.pi05.processor_pi05  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
        from ur3_peg_in_hole_inference_common import postprocess_action_chunk

        if kind not in ('pi05', 'pap_moe'):
            raise ValueError(kind)
        self.kind, self.checkpoint = kind, str(Path(checkpoint).resolve(strict=True))
        self.device = torch.device(device)
        torch.set_grad_enabled(False)
        cls = PAPMoEPolicy if kind == 'pap_moe' else PI05Policy
        self.policy = cls.from_pretrained(self.checkpoint, strict=True)
        self.policy.to(self.device).eval()
        if self.policy.config.chunk_size != 50:
            raise ValueError('Only the frozen 50-step contract is supported')
        self.original_config = copy.deepcopy(self.policy.config)
        self.processor_factory = make_pre_post_processors
        self.postprocess_chunk = postprocess_action_chunk
        self.leftover = None
        self.pre = self.post = None

    def raw(self, observation):
        if self.kind == 'pap_moe':
            from ur3_pap_moe_peg_in_hole_inference import _raw_observation
            return _raw_observation(observation)
        from ur3_baseline_peg_in_hole_inference import _raw_observation, TASK
        return _raw_observation(observation['state'], observation['camera0'],
                                observation['camera1'], None, TASK)

    def batch(self, observation):
        value = self.pre(self.raw(observation))
        if self.kind == 'pap_moe':
            value['expert_mask'] = torch.ones(4, device=self.device)
        return value

    def reset_episode(self, seed):
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if type(seed) is not int or seed < 0:
            raise ValueError('Seed must be a non-negative integer')
        self.policy.config = copy.deepcopy(self.original_config)
        # The model and policy must reference the same restored configuration.
        self.policy.model.config = self.policy.config
        self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10,
            max_guidance_weight=10., prefix_attention_schedule=RTCAttentionSchedule.EXP)
        self.policy.reset()
        self.policy.init_rtc_processor()
        self.policy.eval()
        self.leftover = None
        self.pre, self.post = self.processor_factory(
            self.policy.config, pretrained_path=self.checkpoint)
        state = np.array([0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0], np.float32)
        history = state.copy()
        history[6] = .1  # Preserve CURRENT cold-start warmup, not physical initial grip.
        warm = dict(state=state, camera0=np.zeros((224, 224, 3), np.float32),
                    camera1=np.zeros((224, 224, 3), np.float32),
                    force=np.zeros(6, np.float32), force_fast=np.zeros((64, 6), np.float32),
                    force_slow=np.zeros((50, 6), np.float32),
                    state_history=np.tile(history, (10, 1)),
                    visual_quality=np.array([1, 0, 0, 0], np.float32), metadata={})
        batch = self.batch(warm)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.postprocess_chunk(self.policy.predict_action_chunk(batch), self.post)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        # Cold main() seeds again AFTER warmup but retains its visual memory.
        torch.manual_seed(seed)

    def predict(self, trace_path):
        with np.load(trace_path, allow_pickle=False) as trace:
            observation = {key.removeprefix('observation/'): trace[key].copy()
                           for key in trace.files if key.startswith('observation/')}
            batch = self.batch(observation)
            # Independent check against the original cold-process preprocessing.
            for key, value in batch.items():
                saved_key = 'processed/' + key
                if saved_key in trace and isinstance(value, torch.Tensor):
                    actual = value.detach().cpu().numpy()
                    if not np.array_equal(actual, trace[saved_key]):
                        raise RuntimeError(f'Preprocessing differs from cold trace: {key}')
        with torch.no_grad():
            normalized = self.policy.predict_action_chunk(batch,
                prev_chunk_left_over=self.leftover, inference_delay=0, execution_horizon=10,
                rtc_action_mask=torch.tensor([1.] * 6 + [0.], device=self.device))
            physical = self.postprocess_chunk(normalized, self.post)[0].float().cpu().numpy()
        if physical.shape != (50, 7) or not np.isfinite(physical).all():
            raise RuntimeError('Invalid action shape or non-finite action')
        # Preserve the two original scripts' different full-chunk conventions.
        clip_physical_gripper(physical, self.kind)
        self.leftover = normalized[:, 10:].clone()
        result = dict(normalized_action=normalized.float().cpu().numpy(),
                      physical_action=physical, execution_action=physical[:10].copy())
        if self.kind == 'pap_moe':
            result['route_sequence'] = self.policy.last_routing_probs[0].float().cpu().numpy()
        return result
