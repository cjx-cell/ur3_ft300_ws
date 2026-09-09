"""Unit regression before any joint-training experiment; no production import replacement."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0,str(ROOT/'scripts'))
from probe_route_continuity_flow import continuous_route_prior


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.ActionTokenConditioner


def main():
    old_cls=load('old_modules',Path('/home/ubuntu/lerobot/src/lerobot/policies/pap_moe/pap_moe_modules.py'))
    new_cls=load('new_modules',ROOT/'artifacts/deployment_validation_20260909_interface_ab/code/lerobot/policies/pap_moe/pap_moe_modules.py')
    torch.set_num_threads(2);torch.manual_seed(709)
    old=old_cls(32,32,nhead=4,zero_init_output=False)
    legacy=new_cls(32,32,nhead=4,zero_init_output=False)
    candidate=new_cls(32,32,nhead=4,zero_init_output=False,route_attention_prior='log_probability')
    legacy.load_state_dict(old.state_dict(),strict=True);candidate.load_state_dict(old.state_dict(),strict=True)
    a=torch.randn(2,50,32);e=torch.randn(2,4,32)
    records=[]
    for stepwise in [False,True]:
        shape=(2,50,4) if stepwise else (2,4)
        route=torch.rand(shape);route[...,1]=0.;route/=route.sum(-1,keepdim=True)
        for mode in ['soft','onehot','zero','epsilon']:
            w=route.clone()
            if mode=='onehot': w.zero_();w[...,0]=1.
            if mode=='zero':w.zero_()
            if mode=='epsilon':w[...,1]=1e-8;w/=w.sum(-1,keepdim=True)
            reference=old(a,e,w)
            copied=legacy(a,e,w)
            assert torch.equal(reference,copied),(stepwise,mode,'legacy changed')
            changed=candidate(a,e,w)
            with continuous_route_prior(old): golden=old(a,e,w)
            maxdiff=float((changed-golden).abs().max())
            assert maxdiff<1e-6,(stepwise,mode,maxdiff)
            if mode in ('onehot','zero'): assert torch.allclose(reference,changed,atol=1e-6)
            wg=w.detach().clone().requires_grad_(True)
            aa=a.clone().requires_grad_(True);ee=e.clone().requires_grad_(True)
            candidate.zero_grad(set_to_none=True)
            candidate(aa,ee,wg).square().mean().backward()
            for x in [aa.grad,ee.grad,wg.grad,*[p.grad for p in candidate.parameters()]]:
                assert x is not None and torch.isfinite(x).all(),(stepwise,mode,'bad gradient')
            records.append(dict(stepwise=stepwise,route=mode,legacy_exact=True,hook_equivalence_max=maxdiff,finite_gradients=True))
    path=ROOT/'artifacts/deployment_validation_20260909_interface_ab/interface_unit_tests.json'
    with path.open('x') as f:json.dump(records,f,indent=2)
    print(json.dumps(dict(passed=len(records),new_parameters=0,legacy_default_exact=True)),flush=True)


if __name__=='__main__':main()
