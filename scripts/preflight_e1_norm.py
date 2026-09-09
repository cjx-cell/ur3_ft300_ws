#!/usr/bin/env python3
"""Real-batch gradient and initial complete-action check without weight updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch, _route_chunks
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(exist_ok=False)
    policy=PAPMoEPolicy.from_pretrained(a.checkpoint)
    c=policy.config
    assert policy.model.expert_library.free_load.out.mode == c.e1_fusion_mode
    assert policy.model.expert_library.free_load.out[0].mode==c.e1_fusion_normalization
    assert c.mask_invalid_prefix_tokens and c.mask_invalid_history_cameras
    pre,post=make_pre_post_processors(c,pretrained_path=a.checkpoint)
    data=_load_episode_once(a.episode); index=0
    batch=pre(_raw_batch(data,np.array([index]),50,wrist_dropout=False,continuous_gripper=True,
        visual_history_indices=tuple(c.visual_memory_history_indices)))
    images,masks,history,padding=policy._preprocess_pap_images(batch)
    device=next(policy.parameters()).device
    route=torch.from_numpy(_route_chunks(data['stage'],np.array([index]),50,
        wrist_dropout=False,all_camera_dropout=False)).to(device)
    common=dict(images=images,img_masks=masks,tokens=batch['observation.language.tokens'],
        masks=batch['observation.language.attention_mask'],visual_history=history,
        visual_history_padding=padding,stage_override=route)
    for key in ['state','state_history','force','force_fast','force_slow','visual_quality']:
        common[key]=batch['observation.'+key]
    noise=torch.randn(1,50,c.max_action_dim,generator=torch.Generator().manual_seed(1000)).to(device)
    if c.rtc_config:c.rtc_config.enabled=False
    policy.eval()
    with torch.inference_mode():
        output=policy.model.sample_actions(**common,noise=noise.clone())
        actions=post(output['actions'][...,:7]).float().cpu().numpy()
        assert np.isfinite(actions).all()
        np.save(a.output/'initial_actions.npy',actions)
    policy.train(); torch.manual_seed(42)
    output=policy.model.forward(**common,actions=policy.prepare_action(batch),noise=noise,
        time=torch.full((1,),.5,device=device),stage_labels_soft=route[:,0],
        route_sequence_labels_soft=route,tf_prob=1.)
    loss=output['action_loss'][...,:7].mean()
    if 'expert_representation_loss' in output:
        loss=loss+c.expert_representation_loss_weight*output['expert_representation_loss']
    assert torch.isfinite(loss)
    loss.backward()
    norms={}
    modules={name:module for name,module in policy.model.expert_library.free_load.named_children()}
    modules['action_out_proj']=policy.model.action_out_proj
    for name,module in modules.items():
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        norms[name]=sum(g.float().square().sum().item() for g in grads)**.5
        assert norms[name]>0,name
    result=dict(mode=c.e1_fusion_normalization,fusion_mode=c.e1_fusion_mode,loss=float(loss.detach()),gradient_norms=norms,
        initial_actions='initial_actions.npy',optimizer_updates=0,
        limitations='one real initial-state batch; auxiliary/action backward, not full-distribution or closed-loop verification')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
