"""Probe the actual captured conditioner attention, without running the robot."""
import json
import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root=Path('/home/ubuntu/ur3_ft300_ws')
    out=root/'artifacts/pap_early_mapping_interface_20260907_v1'
    stage=out/'step003000'
    data=np.load(stage/'condition_attention_input.npz')
    q,k,v=[torch.from_numpy(data[name]) for name in ['query','keys','values']]
    mask=torch.from_numpy(data['mask'])
    attention=torch.nn.MultiheadAttention(q.shape[-1],8,batch_first=True).eval()
    prefix='model.action_conditioner.cross_attn.'
    checkpoint=root/'outputs/train/pap_corrected_expert_action_joint_20260907_000738/checkpoints/003000/pretrained_model/model.safetensors'
    with safe_open(checkpoint,framework='pt',device='cpu') as sf:
        state={name.removeprefix(prefix):sf.get_tensor(name) for name in sf.keys() if name.startswith(prefix)}
    attention.load_state_dict(state,strict=True)
    routes=np.load(stage/'actions.npz')['common_route_route'][0]
    outputs={}; results={}
    for name,epsilon in [('original',None),('e2_zero',0.),('e2_1e-8',1e-8),('e2_1e-6',1e-6)]:
        key=k.clone();value=v.clone();m=mask.clone()
        if epsilon is not None:
            ratio=torch.tensor(epsilon/routes[:,1]).reshape(-1,1)
            key[:,1] *= ratio;value[:,1] *= ratio;m[:,1] = epsilon == 0.
        delta,weights=attention(q,key,value,key_padding_mask=m,need_weights=True,average_attn_weights=False)
        outputs[name]=delta
        results[name]=dict(attention_mean10=weights[:10].mean(dim=(0,1,2)).tolist(),
                           delta10_norm_mean=float(delta[:10].norm(dim=-1).mean()))
    for name in outputs:
        results[name]['delta10_mean_l2_vs_e2_zero']=float((outputs[name][:10]-outputs['e2_zero'][:10]).norm(dim=-1).mean())
    result=dict(checkpoint=str(checkpoint),source=str(stage/'condition_attention_input.npz'),
                original_route_mean10=routes[:10].mean(0).tolist(),results=results,
                note='Actual 3k first denoising-step conditioner inputs. CPU float32 attention reconstruction; '
                     'only E2 weighted K/V and zero mask changed, no route renormalization. '
                     'Attention-output feature difference, not action error or closed-loop result.')
    (out/'attention_boundary_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
