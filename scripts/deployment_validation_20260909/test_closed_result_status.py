"""Regression: valid task failures must be scored, not retried or discarded."""
import json
from run_interface_closed import check_result_status, read_trial, ROOT, AB


def main():
    cases = [(0,True,True,[],True),(6,False,True,[],True),
             (6,False,False,['bad_input'],False),(1,False,True,[],False),
             (0,False,True,[],False),(6,True,True,[],False),
             (0,True,True,['conflict'],False)]
    for code,success,valid,reason,expected in cases:
        raw=dict(success=success,evaluation_valid=valid,infrastructure_invalid_reasons=reason,
                 outcome='success' if success else 'timeout')
        try:
            check_result_status(code,raw)
            actual=True
        except AssertionError:
            actual=False
        assert actual==expected,(code,raw)
    checkpoint=ROOT/'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model'
    real=[]
    for eid in (1,11):
        row=read_trial(AB/f'closed_5pos_seed0/pi05_reference_ep{eid:04d}_seed0',
                       'pi05_reference',checkpoint,eid,'interface_ab_fresh_gripper_read_v1')
        real.append(dict(episode=eid,outcome=row['outcome'],exit_code=row['runner_exit_code'],fingerprints_passed=True))
    result=dict(passed=True,synthetic_cases=len(cases),real_trials=real,
                 retry_count=0,changed_policy_or_controller_contract=False)
    with (AB/'closed_result_status_tests.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
