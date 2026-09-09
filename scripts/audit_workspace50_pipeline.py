"""Read-only audit of raw/converted data and frozen processor contracts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
DATA = ROOT/'pap_moe_framework/datasets'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    full = DATA/'lerobot_v3_workspace50_v10_full_clean'
    table = pq.read_table(sorted((full/'data').rglob('*.parquet'))).to_pydict()
    arrays = {k: np.asarray(v) for k, v in table.items()}
    order = np.argsort(arrays['index'])
    arrays = {k:v[order] for k,v in arrays.items()}
    report = dict(frames=len(order), features={}, episodes=[], statistics={}, checkpoints={}, checks={})
    for key, arr in arrays.items():
        report['features'][key] = dict(shape=list(arr.shape), finite=bool(np.isfinite(arr).all()))
    files = sorted((DATA/'workspace_50_v10_canonical').glob('*_success/data.npz'))
    pairs = {'state':'observation.state', 'action':'action', 'force':'observation.force',
             'force_fast':'observation.force_fast', 'force_slow':'observation.force_slow',
             'state_history':'observation.state_history', 'visual_quality':'observation.visual_quality',
             'stage':'observation.physics_gate_target'}
    for episode, file in enumerate(files):
        mask = arrays['episode_index'] == episode
        with np.load(file, allow_pickle=True) as raw:
            dt = np.diff(raw['timestamp'])
            rec = dict(episode=episode+1, frames=int(mask.sum()),
                       xy=[float(raw[k]) for k in ['peg_x','peg_y','hole_x','hole_y']],
                       group_label=int(raw['position_group_id']), style=int(raw['trajectory_style_id']),
                       dt_min=float(dt.min()), dt_max=float(dt.max()),
                       policy_hz=float(raw['policy_hz']),
                       raw_vs_parquet={k:bool(np.array_equal(raw[k],arrays[v][mask])) for k,v in pairs.items()},
                       valid_counts={k:[int(raw[k].min()),int(raw[k].max())] for k in ['force_fast_valid','force_slow_valid','state_history_valid']},
                       gripper_action_range=[float(raw['action'][:,6].min()),float(raw['action'][:,6].max())],
                       gripper_state_range=[float(raw['state'][:,6].min()),float(raw['state'][:,6].max())],
                       history_latest_state_maxabs=float(np.max(np.abs(raw['state_history'][:,-1,:]-raw['state']))),
                       force_reference=str(raw['force_reference_mode']),
                       force_filter=str(raw['force_filter_mode']),
                       gripper_contract=str(raw['gripper_command_contract']))
            report['episodes'].append(rec)
        print(f'raw/parquet {episode+1}/50', flush=True)
    groups = {}
    for row in report['episodes']:
        key = ','.join(map(str,np.round(row['xy'],6)))
        groups.setdefault(key,[]).append(row['style'])
    report['checks'].update(actual_xy_groups=groups,
        reported_group_ids=sorted({r['group_label'] for r in report['episodes']}),
        five_positions_ten_unique_styles=len(groups)==5 and all(sorted(v)==list(range(10)) for v in groups.values()),
        routes_nonnegative=bool((arrays['observation.physics_gate_target']>=0).all()),
        routes_sum_maxabs=float(np.abs(arrays['observation.physics_gate_target'].sum(-1)-1).max()),
        route_mean=arrays['observation.physics_gate_target'].mean(0).tolist())
    for name in ['lerobot_v3_workspace50_v10_full_clean','lerobot_v3_workspace50_v10_full_clean_global_stats_v1',
                 'lerobot_v3_workspace50_v10_baseline_global_stats_v1']:
        stats = json.loads((DATA/name/'meta/stats.json').read_text())
        info = json.loads((DATA/name/'meta/info.json').read_text())
        entry = dict(fps=info['fps'], total_frames=info['total_frames'], total_episodes=info['total_episodes'], features={})
        for key in ['action','observation.state','observation.state_history','observation.force','observation.force_fast','observation.force_slow']:
            if key not in stats: continue
            x=arrays[key]
            # Statistics aggregate history/time axes, retaining the last sensor dimension.
            x=x.reshape(-1,x.shape[-1]).astype(np.float64)
            calc=dict(mean=x.mean(0),std=x.std(0),min=x.min(0),max=x.max(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0))
            entry['features'][key]={metric:dict(saved=stats[key][metric],recomputed=value.tolist(),
                maxabs=float(np.max(np.abs(value-np.array(stats[key][metric]))))) for metric,value in calc.items() if metric in stats[key]}
        report['statistics'][name]=entry
    model_paths = {
      'pi05':'pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model',
      'pap_source':'pap_lr_low_verified_10k_20260905/checkpoints/010000/pretrained_model',
      'pap_C42':'pap_e1_late_sum_s42_20260906/checkpoints/010000/pretrained_model',
      'pap_C43':'pap_e1_late_sum_s43_20260906/checkpoints/010000/pretrained_model'}
    for name,relative in model_paths.items():
        model=ROOT/'outputs/train'/relative
        cfg=json.loads((model/'config.json').read_text())
        record=dict(path=str(model),config={k:cfg.get(k) for k in ['chunk_size','n_action_steps','num_inference_steps','normalization_mapping','use_relative_actions','physics_gate_architecture','action_step_routing','visual_memory_history_indices','mask_invalid_prefix_tokens','mask_invalid_history_cameras']},
                    processors={},roundtrip={})
        normalizers={}
        for p in model.glob('*processor*.safetensors'):
            with safe_open(p,framework='np') as sf:
                vals={k:sf.get_tensor(k) for k in sf.keys()}
            record['processors'][p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finite=all(np.isfinite(v).all() for v in vals.values()))
            for feature in ['action','observation.state']:
                if f'{feature}.q01' not in vals:continue
                lo,hi=vals[f'{feature}.q01'],vals[f'{feature}.q99']
                normalizers[(('post' if 'postprocessor' in p.name else 'pre'),feature)]=(lo,hi)
        for feature in ['action','observation.state']:
            lo,hi=normalizers[('pre',feature)]; plo,phi=normalizers[('post',feature)]
            x=arrays[feature]
            # Algebraic check of saved mappings, not a replacement for executing processors.
            denom=np.where(hi-lo==0,1e-8,hi-lo)
            post_denom=np.where(phi-plo==0,1e-8,phi-plo)
            y=(x-lo)/denom*2-1
            restored=(y+1)/2*post_denom+plo
            record['roundtrip'][feature]=dict(pre_post_equal=bool(np.array_equal(lo,plo) and np.array_equal(hi,phi)),
                maxabs=float(np.abs(restored-x).max()),normalized_outside_1_fraction=(np.abs(y)>1).mean(0).tolist(),
                quantile_span=(hi-lo).tolist())
        report['checkpoints'][name]=record
    report['limits']=['No live Gazebo or new training run; synchronization/physical success not verified.',
                      'History-axis statistics flattening must be interpreted against actual normalizer broadcasting.',
                      'Raw/parquet comparison covers numeric modalities, not all video pixels.',
                      'Algebraic roundtrip can pass with wrong statistics; global recomputation is a separate check.']
    (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('audit saved', args.output/'audit.json', flush=True)


if __name__ == '__main__':
    main()
