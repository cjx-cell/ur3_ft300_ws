"""Read-only audit of all seven failures; writes derived evidence to a new directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.batch / 'results.json').read_text())
    failures = [r for r in rows if r['outcome'] == 'unknown_failure']
    assert len(failures) == 7
    records = []
    for row in failures:
        artifact = Path(row['artifact_dir'])
        source = artifact / 'diagnostic/controller.jsonl'
        events = [json.loads(line) for line in source.read_text().splitlines()]
        dispatch = [e for e in events if e['kind'] == 'dispatch'][-1]
        chunk = dispatch['chunk_id']
        feedback = [e for e in events if e['kind'] == 'feedback' and e['chunk_id'] == chunk]
        assert feedback and len(feedback[-1]['joints']) == 7
        req = np.asarray(dispatch['requested'])
        sent = np.asarray(dispatch['controller'])
        measured = np.asarray([e['actual'] for e in feedback])
        desired = np.asarray([e['desired'] for e in feedback])
        time = np.asarray([e['trajectory_time'] for e in feedback])
        tail = time >= time[-1] - 1.
        requests = [json.loads(s) for s in (artifact / 'resident_requests.jsonl').read_text().splitlines()]
        request = requests[chunk]
        assert request['sequence'] == chunk
        with np.load(request['result'], allow_pickle=False) as saved:
            model = saved['execution_action'].copy()
        assert model.shape == req.shape
        delta = np.abs(model - req).max()
        assert delta < 1e-7, delta
        name = f"{row['eval_label']}_ep{row['episode']:04d}_seed{row['seed']}"
        record = dict(name=name, artifact=str(artifact), chunk=chunk,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            worker_result=request['result'], model_to_requested_max=float(delta),
            requested_to_sent_gripper_max=float(abs(req[:, 6]-sent[:, 6]).max()),
            requested_to_sent_arm_max=float(abs(req[:, :6]-sent[:, :6]).max()),
            start_measured_gripper=dispatch['state'][6], first_target_gripper=float(req[0,6]),
            last_target_gripper=float(req[-1,6]), last_sent_gripper=float(sent[-1,6]),
            final_measured_gripper=float(measured[-1,6]),
            measured_gripper_range=[float(measured[:,6].min()),float(measured[:,6].max())],
            final_gripper_error=float(desired[-1,6]-measured[-1,6]),
            final_arm_max_error=float(abs(desired[-1,:6]-measured[-1,:6]).max()),
            final_second_gripper_range=float(np.ptp(measured[tail,6])),
            trajectory_time_end=float(time[-1]), gripper_peg_distance_at_dispatch=dispatch['gripper_distance'],
            interpretation='Target reached controller unchanged; endpoint tracking failure. Contact cause not proven by position feedback alone.')
        records.append(record)
        with (args.output / f'{name}.csv').open('x') as out:
            writer = csv.writer(out)
            writer.writerow(['trajectory_time_s', 'desired_gripper_rad', 'actual_gripper_rad', 'arm_max_error_rad'])
            writer.writerows(zip(time, desired[:,6], measured[:,6], abs(desired[:,:6]-measured[:,:6]).max(1)))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        command_time = (np.arange(len(req))+1)*dispatch['action_dt']
        ax.plot(command_time, req[:,6], 'o--', label='model requested')
        ax.plot(command_time, sent[:,6], 'x', label='sent to controller')
        ax.plot(time, desired[:,6], label='controller desired')
        ax.plot(time, measured[:,6], label='measured')
        ax.axvline(1., color='gray', linestyle=':', label='nominal goal end')
        ax.set(xlabel='trajectory time (simulation seconds)', ylabel='gripper joint (rad)', title=name)
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(args.output / f'{name}.svg'); plt.close(fig)
        print(json.dumps(record), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
