"""Per-Gazebo-episode adapter. Original loaders + original paired-v1 replies.

Only this lightweight client is killed on each Gazebo teardown. It does not
load weights or send unpaired/legacy actions. Candidate engineering use only.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np

PEG = Path('/home/ubuntu/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole')
sys.path.insert(0, str(PEG))
import policy_action_exchange as exchange
from observation_snapshot import read_stable


def rpc(stream, request):
    stream.write(json.dumps(request) + '\n')
    stream.flush()
    response = json.loads(stream.readline())
    if 'error' in response:
        raise RuntimeError(response)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi05', 'pap_moe'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    if not exchange.enabled():
        raise RuntimeError('Resident bridge requires paired-v1; no legacy fallback')
    if args.kind == 'pap_moe':
        from ur3_pap_moe_peg_in_hole_inference import _load_observation as load
    else:
        from ur3_baseline_peg_in_hole_inference import _load_observation

        def load():
            state, image0, image1, _, _ = _load_observation(False, 'continuous_radians_0_0.8')
            return dict(state=state, camera0=image0, camera1=image1)

    output = Path(os.environ['POLICY_PAIRED_ACTION_FILE']).parent
    inputs = output / 'resident_inputs'
    inputs.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(60)
        connection.connect(args.socket)
        with connection.makefile('rw') as stream:
            hello = json.loads(stream.readline())
            if hello.get('kind') != args.kind or hello.get('checkpoint') != str(args.checkpoint.resolve()):
                raise RuntimeError('Resident model identity mismatch')
            started = time.monotonic()
            token = rpc(stream, dict(op='begin', seed=args.seed))['token']
            connection.settimeout(3)
            metadata = dict(policy_type=args.kind, checkpoint=str(args.checkpoint.resolve()),
                predicted_action_steps=50, executed_action_steps=10, action_dt_s=.1,
                inference_seed=args.seed, state_gripper_mode='continuous_radians_0_0.8',
                gripper_action_mode='continuous_radians_0_0.8', rtc_enabled=True,
                rtc_execution_horizon=10, rtc_inference_delay_steps=0,
                rtc_max_guidance_weight=10., rtc_prefix_attention_schedule='EXP',
                rtc_action_dimensions='arm_only', resident_episode_token=token,
                reset_seconds=time.monotonic() - started,
                evaluation_kind=os.environ.get('WORKSPACE50_EVALUATION_KIND', 'engineering_resident_canary'))
            (output / 'resident_ready.json').write_text(json.dumps(metadata, indent=2))
            ready = Path('/tmp/ur3_inference_ready.txt')
            temp = ready.with_suffix('.resident_tmp')
            temp.write_text(json.dumps(metadata) + '\n')
            temp.replace(ready)
            previous_id = None
            sequence = 0
            def report_read(event):
                event.update(sequence=sequence, wall_time=time.time())
                with (output / 'resident_observation_reads.jsonl').open('a') as log:
                    log.write(json.dumps(event) + '\n')

            while True:
                start = time.monotonic()
                snapshot = read_stable(load,
                    lambda: exchange.observation_id('/tmp/ur3_joint_state.txt'),
                    previous_id, report=report_read)
                if snapshot is None:
                    time.sleep(.001)
                    continue
                request_id, observation = snapshot
                trace = inputs / f'chunk_{sequence:05d}.npz'
                values = {f'observation/{key}': value for key, value in observation.items()
                          if isinstance(value, np.ndarray)}
                with trace.open('xb') as file:
                    np.savez(file, **values)
                response = rpc(stream, dict(op='infer', token=token, sequence=sequence, trace=str(trace)))
                if response.get('token') != token or response.get('sequence') != sequence:
                    raise RuntimeError('Stale or out-of-order resident response')
                with np.load(response['result'], allow_pickle=False) as result:
                    actions = result['execution_action'].copy()
                if actions.shape != (10, 7):
                    raise RuntimeError('Wrong execution prefix shape')
                if exchange.observation_id('/tmp/ur3_joint_state.txt') != request_id:
                    raise RuntimeError('Request superseded during inference; refusing stale reply')
                exchange.publish_reply(request_id, actions)
                with (output / 'resident_requests.jsonl').open('a') as log:
                    log.write(json.dumps(dict(token=token, sequence=sequence,
                        request_id=request_id, input=str(trace), result=response['result'],
                        latency_ms=(time.monotonic() - start) * 1000)) + '\n')
                print(f'Resident chunk {sequence}: {(time.monotonic()-start)*1000:.1f} ms', flush=True)
                previous_id = request_id
                sequence += 1


if __name__ == '__main__':
    main()
