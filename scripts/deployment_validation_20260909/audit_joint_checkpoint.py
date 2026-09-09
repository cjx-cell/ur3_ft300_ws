"""Inspect saved weight changes and all persisted normalization tensors on CPU."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
SOURCE=ROOT/'outputs/train/pap_corrected_gate_calibration_20260907_104300/checkpoints/015000/pretrained_model'


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    selected=['model.action_conditioner.cross_attn.out_proj.weight',
        'model.expert_library.free_load.load.1.weight','model.expert_library.visual_blind.current.1.weight',
        'model.expert_library.rigid_micro.dynamic_stats.1.weight','model.expert_library.compliant.force_trends.1.weight',
        'model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.base.weight',
        'model.paligemma_with_expert.gemma_expert.model.layers.17.mlp.down_proj.base.weight']
    changes={};frozen=[]
    with safe_open(SOURCE/'model.safetensors',framework='pt',device='cpu') as a,safe_open(args.checkpoint/'model.safetensors',framework='pt',device='cpu') as b:
        for name in selected:
            x=a.get_tensor(name);y=b.get_tensor(name);delta=(y.float()-x.float()).abs()
            assert torch.isfinite(y).all() and delta.max()>0,name
            changes[name]=dict(dtype=str(y.dtype),max_change=float(delta.max()),changed_fraction=float((delta>0).float().mean()))
        for name in a.keys():
            if name.startswith(('model.physics_gate.','model.force_encoder.','model.proprio_encoder.','model.visual_quality_encoder.')):
                assert torch.equal(a.get_tensor(name),b.get_tensor(name)),name
                frozen.append(name)
        # Representative VLM tensors only; not a hash of the entire frozen VLM.
        vlm=[n for n in a.keys() if n.startswith('model.paligemma_with_expert.paligemma.model.language_model.layers.0.') and n.endswith('norm.weight')]
        for name in vlm:assert torch.equal(a.get_tensor(name),b.get_tensor(name)),name
    stats=[]; scalar_metadata_reshapes=[]
    for path in SOURCE.glob('*processor*.safetensors'):
        other=args.checkpoint/path.name
        with safe_open(path,framework='pt',device='cpu') as a,safe_open(other,framework='pt',device='cpu') as b:
            assert set(a.keys())==set(b.keys())
            for key in a.keys():
                x=a.get_tensor(key);y=b.get_tensor(key)
                if not torch.equal(x,y):
                    # Reload/save canonicalizes metadata scalars [1] -> [].
                    # Never permit this exception for effective state/action/FT
                    # normalization vectors, nor for any changed numeric value.
                    metadata=(key.endswith('.count') or key.rsplit('.',1)[0] in
                              {'episode_index','frame_index','index','task_index','timestamp'})
                    assert metadata and x.numel()==y.numel()==1 and torch.equal(x.reshape(1),y.reshape(1)),key
                    scalar_metadata_reshapes.append(dict(file=path.name,key=key,source_shape=list(x.shape),target_shape=list(y.shape),value=float(x.reshape(1)[0])))
                stats.append(path.name+'/'+key)
    result=dict(source=str(SOURCE),checkpoint=str(args.checkpoint),changed_trainable_samples=changes,
        exact_frozen_gate_encoder_tensors=frozen,exact_vlm_sample_tensors=vlm,numerically_equal_processor_tensors=stats,
        scalar_metadata_reshapes=scalar_metadata_reshapes,
        limitation='Weight changes plus optimizer checks confirm actual updates, not that every auxiliary expert learned useful task information.')
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(changed_samples=len(changes),frozen_gate_encoder_tensors=len(frozen),processor_tensors=len(stats),passed=True)))


if __name__=='__main__':main()
