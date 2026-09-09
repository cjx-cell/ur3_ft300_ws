"""Frozen, CPU-only replay of saved branch features; no policy updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from audit_e1_information_fusion import probes
from lerobot.policies.pap_moe.pap_moe_modules import ActionTokenConditioner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    results = {}
    for name in ['A_s42', 'B_s42', 'A_s43', 'B_s43']:
        checkpoint = Path(json.loads((args.root / f'{name}.json').read_text())['output_dir']) / 'checkpoints/010000/pretrained_model'
        cfg = json.loads((checkpoint / 'config.json').read_text())
        with np.load(args.root / f'eval_{name}/features.npz') as z:
            arrays = {k: z[k] for k in z.files}
        with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as f:
            weights = {k: f.get_tensor(k).float() for k in f.keys() if
                       'free_load.out.' in k or 'action_conditioner.' in k or k == 'model.expert_conditioning_scales'}
        prefix = 'model.expert_library.free_load.out.'
        def linear(x, layer):
            return F.linear(x, weights[prefix + layer + '.weight'], weights[prefix + layer + '.bias'])
        with torch.inference_mode():
            concat = torch.from_numpy(np.concatenate([arrays['feature_' + b] for b in ['visual', 'motion', 'load']], -1)).float()
            if cfg['e1_fusion_normalization'] == 'joint':
                norm = F.layer_norm(concat, (768,), weights[prefix+'0.weight'], weights[prefix+'0.bias'])
            else:
                norm = F.layer_norm(concat.reshape(-1, 3, 256), (256,)).reshape(-1,768)
                norm = norm * weights[prefix+'0.weight'] + weights[prefix+'0.bias']
            compressed = linear(norm, '1')
            compression_weight = weights[prefix+'1.weight']
            retained = (compressed-weights[prefix+'1.bias']) @ torch.linalg.pinv(compression_weight).T
            discarded = norm-retained
            activated = F.gelu(compressed)
            out = linear(activated, '3')
            expansion_weight = weights[prefix+'3.weight']
            singular = torch.linalg.svdvals(expansion_weight)
            recovered = (out-weights[prefix+'3.bias']) @ torch.linalg.pinv(expansion_weight).T
            captured = torch.from_numpy(arrays['feature_out']).float()
            replay_error = float((out-captured).abs().max())
            # Validate cached feature replay before drawing any layer comparison.
            assert torch.allclose(out, captured, atol=2e-4, rtol=2e-4), replay_error
            interface = ActionTokenConditioner(2048,1024)
            interface.load_state_dict({k.removeprefix('model.action_conditioner.'):v for k,v in weights.items() if 'model.action_conditioner.' in k})
            interface.eval()
            projected = interface.condition_proj(out)
            scale = weights['model.expert_conditioning_scales'].tanh()
            scale[0] *= cfg['nominal_expert_conditioning_multiplier']
            tokens = torch.zeros(len(out),4,2048)
            tokens[:,0] = out
            route = torch.zeros(len(out),1,4)
            route[:,:,0] = 1
            q0 = torch.zeros(len(out),1,1024)
            gen = torch.Generator().manual_seed(72)
            q1 = torch.randn(q0.shape,generator=gen)
            raw = interface(q0,tokens,route,scale,None)-q0
            raw1 = interface(q1,tokens,route,scale,None)-q1
            clipped = interface(q0,tokens,route,scale,1.0)-q0
            # Decompose the single-active-key attention value path into content + shared biases.
            attn = interface.cross_attn
            vw = attn.in_proj_weight[2048:]
            vb = attn.in_proj_bias[2048:]
            content = F.linear(F.linear(projected*scale[0],vw),attn.out_proj.weight)
            bias = F.linear(vb,attn.out_proj.weight,attn.out_proj.bias)
            assert torch.allclose(raw[:,0],content+bias,atol=2e-5,rtol=2e-5)
            stat = dict(replay_max_abs_error=replay_error,
                pure_E1_query_change_max_abs=float((raw-raw1).abs().max()),
                pure_E1_clip_fraction=float((raw.norm(dim=-1)>1).float().mean()),
                raw_norm_median=float(raw.norm(dim=-1).median()),
                content_norm_median=float(content.norm(dim=-1).median()),
                shared_bias_norm=float(bias.norm()), scale=float(scale[0]))
            stat.update(expansion_rank=int(torch.linalg.matrix_rank(expansion_weight)),
                        expansion_condition_number=float(singular[0]/singular[-1]),
                        expansion_inverse_max_error=float((recovered-activated).abs().max()),
                        discarded_compression_max_abs=float(F.linear(discarded,compression_weight).abs().max()))
            # These interface probes use E1-only routes, not actual mixed contact routes.
            extra = dict(concat=concat, normalized=norm, compressed=compressed, activated=activated,
                         replay_out=out, interface_projected=projected,
                         interface_raw=raw[:,0], interface_clipped=clipped[:,0],
                         recovered_activated=recovered, compression_discarded=discarded,
                         compression_retained=retained)
            for key,value in extra.items(): arrays['feature_'+key] = value.numpy()
        print(name,json.dumps(stat),flush=True)
        output = probes(arrays)
        (args.output/f'{name}_probes.json').write_text(json.dumps(output))
        summary=[]
        for target in sorted({x['target'] for x in output}):
            for split in ['position','held_styles']:
                for feature in sorted({x['feature'] for x in output}):
                    for alpha in [.001,.01,.1]:
                        rows=[x for x in output if x['target']==target and x['feature']==feature and x['alpha']==alpha and x['fold'].startswith(split)]
                        mse=float(np.mean([x['pooled_mse'] for x in rows]))
                        null=float(np.mean([np.mean(x['null_mse']) for x in rows]))
                        summary.append(dict(target=target,split=split,feature=feature,alpha=alpha,ratio=mse/null))
        (args.output/f'{name}_summary.json').write_text(json.dumps(summary,indent=2))
        results[name]=stat
        (args.output/'replay_checks.json').write_text(json.dumps(results,indent=2))
    (args.output/'contract.json').write_text(json.dumps(dict(policy_updates=0,
        source=str(args.root),cases=list(results),samples_per_model=300,
        intervention='E1-only route, actual conditioner; query zero/random; not a full policy rollout',
        probe='same fixed folds and three ridge alphas; policy saw all50; no deployment generalization claim'),indent=2))


if __name__ == '__main__':
    main()
