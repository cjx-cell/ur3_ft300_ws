"""Fail-fast parameter/optimizer checks for the isolated interface A/B."""
import hashlib
import json
from pathlib import Path

from lerobot.scripts import lerobot_train as trainer

factory=trainer.make_optimizer_and_scheduler


def checked(cfg,policy):
    source=Path(__import__('lerobot.policies.pap_moe.modeling_pap_moe',fromlist=['x']).__file__).resolve()
    assert 'deployment_validation_20260909_interface_ab/code' in str(source),source
    assert policy.config.train_expert_action_joint
    assert not policy.config.train_physicsgate_action_joint
    assert policy.model.action_conditioner.route_attention_prior==cfg.policy.action_conditioning_route_prior
    assert not any(p.requires_grad for p in policy.model.physics_gate.parameters())
    assert not any(p.requires_grad for p in policy.model.force_encoder.parameters())
    assert not any(p.requires_grad for p in policy.model.paligemma_with_expert.paligemma.parameters())
    assert all(p.requires_grad for p in policy.model.expert_library.parameters())
    assert all(p.requires_grad for p in policy.model.action_conditioner.parameters())
    assert all(p.requires_grad for p in policy.model.paligemma_with_expert.gemma_expert.model.parameters())
    opt,sched=factory(cfg,policy)
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))
    assert set(ids)=={id(p) for p in policy.parameters() if p.requires_grad}
    assert len(opt.param_groups)==2
    lrs=[g.get('initial_lr',g['lr']) for g in opt.param_groups]
    assert lrs==[2.5e-5,2.5e-5],lrs
    names=[n for n,p in policy.named_parameters() if p.requires_grad]
    result=dict(route_prior=cfg.policy.action_conditioning_route_prior,source=str(source),
        trainable_numel=sum(p.numel() for p in policy.parameters() if p.requires_grad),
        group_numel=[sum(p.numel() for p in g['params']) for g in opt.param_groups],
        peak_lrs=lrs,trainable_names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        frozen=['VLM','PhysicsGate','force_encoder'],trainable_names=names)
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'verified_interface_contract.json').write_text(json.dumps(result,indent=2))
    print('INTERFACE_TRAINING_CONTRACT_PASS',json.dumps({k:v for k,v in result.items() if k!='trainable_names'}),flush=True)
    return opt,sched


if __name__=='__main__':
    trainer.make_optimizer_and_scheduler=checked
    trainer.main()
