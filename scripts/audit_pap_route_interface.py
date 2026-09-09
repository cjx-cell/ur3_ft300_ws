"""Read-only route-label preprocessing, timing and boundary audit."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from replay_first_action_trace import ROOT, np, torch, _raw_observation, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pap_moe.pap_moe_modules import FactorizedPhysicsGate


def main():
    out = ROOT/'artifacts/pap_early_mapping_interface_20260907_v1'
    checkpoint = ROOT/'outputs/train/pap_corrected_expert_action_joint_20260907_000738/checkpoints/003000/pretrained_model'
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    pre, _ = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    data_root = ROOT/'pap_moe_framework/datasets/lerobot_v3_workspace50_v10_full_clean_global_stats_v1'
    table = pq.read_table(sorted((data_root/'data').rglob('*.parquet')),
                          columns=['index','episode_index','observation.physics_gate_target']).to_pydict()
    order = np.argsort(table['index'])
    routes = np.asarray(table['observation.physics_gate_target'],np.float32)[order]
    labels = torch.from_numpy(routes[:50].copy())
    obs_path = ROOT/'artifacts/gazebo_pap_moe_workspace50_20260907_174617_ep0001_seed0/diagnostic/chunk_00001.npz'
    with np.load(obs_path) as d:
        raw = {k.removeprefix('observation/'): d[k].copy() for k in d.files
               if k.startswith('observation/') and '/' not in k.removeprefix('observation/')}
    batch = _raw_observation(raw)
    # Training DataLoader has already collated route sequences to [B,T,4].
    batch['observation.physics_gate_target'] = labels.unsqueeze(0)
    processed = pre(batch)['observation.physics_gate_target'].cpu()
    assert processed.shape == (1,50,4), processed.shape
    assert torch.equal(processed[0],labels)
    normalized = processed.relu() / (processed.relu().sum(-1,keepdim=True)+1e-8)
    factors = FactorizedPhysicsGate.expert_probs_to_factors(torch.from_numpy(routes), stable=True)
    reconstructed = FactorizedPhysicsGate.factors_to_expert_probs(factors).numpy()
    info=json.loads((data_root/'meta/info.json').read_text())
    result=dict(frames=len(routes),fps=info['fps'],
                route_delta_indices=config.feature_delta_indices['observation.physics_gate_target'],
                action_delta_indices=config.action_delta_indices,
                route_shape=list(processed.shape),processor_exact_identity=True,
                label_renormalization_max_abs_diff=float((normalized-processed).abs().max()),
                factor_roundtrip_max_abs_diff=float(np.abs(routes-reconstructed).max()),
                route_zero_fraction=(routes==0).mean(0).tolist(),route_mean=routes.mean(0).tolist(),
                all_e2_clean_labels_zero=bool((routes[:,1]==0).all()),
                training_route_jitter=config.physical_route_jitter_std,
                training_condition_dropout=config.physical_condition_dropout_probability,
                visual_degradation_probability=config.visual_degradation_training_probability,
                note='Clean labels; training additionally synthesizes degraded-view E2 targets. '
                     'Same frame indices do not assert physical correctness of the route labels.')
    assert result['route_delta_indices'] == result['action_delta_indices'] == list(range(50))
    (out/'route_interface_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
