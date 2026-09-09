"""Engineering-only temporal replay; never a model or recovery policy score."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import policy_action_exchange as exchange


def action_window(actions, frame, size=10):
    if frame >= len(actions):
        raise ValueError('Demonstration exhausted; do not invent a recovery action')
    chunk = actions[frame:frame + size].copy()
    if len(chunk) < size:
        chunk = np.concatenate((chunk, np.repeat(chunk[-1:], size - len(chunk), axis=0)))
    return chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)  # launcher compatibility, never loaded
    parser.add_argument('--episode', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.episode, allow_pickle=False) as data:
        actions = data['action'].astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('Expected finite continuous-radian actions [T,7]')
    if (actions[:, 6] < 0).any() or (actions[:, 6] > .8).any():
        raise ValueError('Not the continuous 0..0.8 rad dataset contract')
    if not exchange.enabled():
        raise ValueError('Demonstration diagnosis requires paired action exchange')
    ready = Path('/tmp/ur3_inference_ready.txt')
    tmp = ready.with_suffix('.tmp')
    tmp.write_text(json.dumps({'mode': 'engineering_demonstration_replay', 'episode': str(args.episode),
                               'gripper_action_mode': 'continuous_radians', 'predicted_action_steps': 50,
                               'executed_action_steps': 10, 'action_dt_s': .1}))
    tmp.replace(ready)
    last_request, frame = None, 0
    while True:
        try:
            request = exchange.observation_id('/tmp/ur3_joint_state.txt')
        except FileNotFoundError:
            time.sleep(.01)
            continue
        if request == last_request:
            time.sleep(.001)
            continue
        if frame >= len(actions):
            print('Demonstration exhausted, no task success inferred.', flush=True)
            finished = Path(os.environ['POLICY_PAIRED_ACTION_FILE'] + '.finished')
            temporary = finished.with_suffix('.tmp')
            temporary.write_text(request)
            temporary.replace(finished)
            return
        chunk = action_window(actions, frame)
        exchange.publish_reply(request, chunk)
        print(f'DEMONSTRATION frames={frame}:{min(frame+10,len(actions))}, gripper={chunk[:,6].tolist()}', flush=True)
        frame += 10
        last_request = request


if __name__ == '__main__':
    main()
