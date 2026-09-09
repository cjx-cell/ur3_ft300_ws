#!/usr/bin/env python3
"""Checkpoint-local gradient and shared fusion-bias probes; no optimizer updates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/lerobot/src')
from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch, _route_chunks
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401


def grad_stats(a, b):
    aa = sum((g.float().square().sum() for g in a if g is not None), torch.tensor(0., device='cuda'))
    bb = sum((g.float().square().sum() for g in b if g is not None), torch.tensor(0., device='cuda'))
    dot = sum(((x.float()*y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None), torch.tensor(0., device='cuda'))
    return dict(action_norm=aa.sqrt().item(), weighted_aux_norm=bb.sqrt().item(),
                cosine=(dot/(aa*bb).sqrt()).item() if aa > 0 and bb > 0 else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode-npz', action='append', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'records.jsonl').exists():
        raise FileExistsError(args.output)
    policy = PAPMoEPolicy.from_pretrained(args.checkpoint)
    cfg = policy.config
    cfg.physical_condition_dropout_probability = 0
    cfg.physical_route_jitter_std = 0
    if cfg.rtc_config:
        cfg.rtc_config.enabled = False
    model = policy.model
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    groups = {name:list(module.parameters()) for name,module in model.expert_library.free_load.named_children()}
    groups['fusion_bias'] = [model.action_conditioner.cross_attn.in_proj_bias, model.action_conditioner.cross_attn.out_proj.bias]
    groups['scales'] = [model.expert_conditioning_scales]
    params = [p for group in groups.values() for p in group]
    for param in params:
        param.requires_grad_(True)
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    device = next(policy.parameters()).device
    modes = ['full', 'zero_attention_output_bias', 'zero_attention_value_bias',
             'zero_attention_biases', 'zero_E1_scale', 'drop_E1']
    records = []
    (args.output/'contract.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,
        episodes=args.episode_npz, seeds=[1000,1001003], anchors_per_episode=3,
        gradients='clean deterministic flow t=.5; all stochastic dropout/jitter off; weighted auxiliary is full representation loss * configured .1',
        parameter_selection='E1 branches, shared attention biases and scales only require_grad for memory; other parameter gradients not measured; no optimizer',
        masks=dict(prefix=cfg.mask_invalid_prefix_tokens,history=cfg.mask_invalid_history_cameras),
        limitations='Training observations, not closed-loop. Attention biases shared across experts, not E1-specific.', modes=modes),indent=2))
    with (args.output/'records.jsonl').open('w') as log:
        for ep, path in enumerate(args.episode_npz):
            data = _load_episode_once(Path(path))
            indices = np.unique(np.linspace(0,max(0,len(data['action'])-cfg.chunk_size),3).astype(int))
            for index in indices:
                policy.reset()
                batch = pre(_raw_batch(data,np.array([index]),cfg.chunk_size,wrist_dropout=False,
                    continuous_gripper=True, visual_history_indices=tuple(cfg.visual_memory_history_indices)))
                images, masks, history, padding = policy._preprocess_pap_images(batch)
                route = torch.from_numpy(_route_chunks(data['stage'],np.array([index]),cfg.chunk_size,
                    wrist_dropout=False,all_camera_dropout=False)).to(device)
                common = dict(images=images,img_masks=masks,tokens=batch['observation.language.tokens'],
                    masks=batch['observation.language.attention_mask'],visual_history=history,
                    visual_history_padding=padding,stage_override=route)
                for key in ['force','force_fast','force_slow','state','state_history','visual_quality']:
                    common[key]=batch['observation.'+key]
                target = data['action'][index:index+cfg.chunk_size].astype(np.float32)
                for seed in [1000,1001003]:
                    generator=torch.Generator().manual_seed(seed+ep*10007+int(index))
                    noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=generator).to(device)
                    # Train graph with deterministic modules: checkpointing on, no stochastic interventions.
                    model.train()
                    for module in model.modules():
                        if isinstance(module,torch.nn.Dropout):
                            module.eval()
                    output=model.forward(**common,actions=policy.prepare_action(batch),noise=noise,
                        time=torch.full((1,),.5,device=device),stage_labels_soft=route[:,0],
                        route_sequence_labels_soft=route,tf_prob=1.)
                    loss=output['action_loss'][...,:7].mean()
                    aux=output['expert_representation_loss']*cfg.expert_representation_loss_weight
                    ga=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                    gb=torch.autograd.grad(aux,params,allow_unused=True)
                    stats={}
                    offset=0
                    for name, group in groups.items():
                        stats[name]=grad_stats(ga[offset:offset+len(group)],gb[offset:offset+len(group)])
                        offset+=len(group)
                    row=dict(kind='gradient',episode=ep,frame=int(index),seed=seed,
                        action_loss=loss.item(),weighted_aux_loss=aux.item(),groups=stats,
                        E1_current_weight=route[0,0,0].item())
                    records.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                    del ga,gb,output,loss,aux
                    model.eval()
                    attn=model.action_conditioner.cross_attn
                    with torch.no_grad():
                        saved=[attn.in_proj_bias.clone(),attn.out_proj.bias.clone(),model.expert_conditioning_scales.clone()]
                        for mode in modes:
                            try:
                                if mode=='zero_attention_output_bias':
                                    attn.out_proj.bias.zero_()
                                elif mode=='zero_attention_value_bias':
                                    attn.in_proj_bias[2*attn.embed_dim:].zero_()
                                elif mode=='zero_attention_biases':
                                    attn.in_proj_bias.zero_(); attn.out_proj.bias.zero_()
                                elif mode=='zero_E1_scale':
                                    model.expert_conditioning_scales[0]=0
                                mask=torch.tensor([0.,1.,1.,1.] if mode=='drop_E1' else [1.,1.,1.,1.],device=device)
                                output=model.sample_actions(**common,noise=noise,expert_mask=mask)
                                prediction=post(output['actions'][...,:7]).float().cpu().numpy()[0]
                                if not np.isfinite(prediction).all():
                                    raise RuntimeError('Nonfinite actions')
                                row=dict(kind='action',episode=ep,frame=int(index),seed=seed,mode=mode)
                                for h in [10,50]:
                                    err=(prediction[:h]-target[:h])**2
                                    row[f'arm_mse_{h}']=float(err[:,:6].mean())
                                    row[f'gripper_mse_{h}']=float(err[:,6].mean())
                                records.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                            finally:
                                attn.in_proj_bias.copy_(saved[0]); attn.out_proj.bias.copy_(saved[1])
                                model.expert_conditioning_scales.copy_(saved[2])
                    print(f'episode={ep+1} frame={index} seed={seed} done',flush=True)
            del data
    summary={mode:{key:float(np.mean([r[key] for r in records if r['kind']=='action' and r['mode']==mode]))
        for key in ['arm_mse_10','arm_mse_50','gripper_mse_10','gripper_mse_50']} for mode in modes}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
