#!/usr/bin/env python3
"""Isolated static-pose Gazebo acceptance test, not dynamic drop testing."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'))
from fixture_geometry_audit import parse_model_poses, score_seating


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    models = ROOT / 'src/ur_simulation_gz/ur_simulation_gz/models'
    cases = dict(seated=(0, .110, 0, True, True), partial=(0, .113, 0, True, False),
                 hovering=(0, .2, 0, False, False), too_deep=(0, .09, 0, False, False),
                 sideways=(0, .11, math.pi/2, False, False), inverted=(0, .11, math.pi, False, False),
                 off_axis=(.008, .11, 0, False, False), below_floor=(0, -.1, 0, False, False))
    model_xml = []
    for index, (name, (dx, z, pitch, _, _)) in enumerate(cases.items()):
        for kind, x, height, angle in [('peg', index+dx, z, pitch), ('hole', index, 0, 0)]:
            mesh = models / f'pap_moe_real_{kind}/meshes/{"peg" if kind == "peg" else "hole"}.stl'
            model_xml.append(f'<model name="{name}_{kind}"><static>true</static><pose>{x} 0 {height} 0 {angle} 0</pose>'
                             f'<link name="link"><visual name="visual"><geometry><mesh><uri>file://{mesh}</uri>'
                             '</mesh></geometry></visual></link></model>')
    world = '<sdf version="1.6"><world name="geometry_audit"><physics name="default" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor></physics>'
    world += '<plugin filename="ignition-gazebo-physics-system" name="gz::sim::systems::Physics"/>'
    world += '<plugin filename="ignition-gazebo-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>'
    world += ''.join(model_xml) + '</world></sdf>'
    path = args.output / 'static_cases.sdf'
    path.write_text(world)
    env = dict(os.environ, IGN_PARTITION=f'geometry_audit_{os.getpid()}')
    env['GZ_PARTITION'] = env['IGN_PARTITION']
    with (args.output / 'gazebo.log').open('x') as log:
        server = subprocess.Popen(['ign', 'gazebo', '-s', '-r', str(path)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+30
            names = tuple(f'{name}_{kind}' for name in cases for kind in ('peg', 'hole'))
            poses = {}
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    reply = subprocess.run(['ign', 'topic', '-t', '/world/geometry_audit/pose/info', '-e', '-n', '1'],
                                           env=env, capture_output=True, text=True, timeout=3, check=True)
                    poses = parse_model_poses(reply.stdout, names)
                    if set(poses) == set(names):
                        (args.output / 'pose_message.txt').write_text(reply.stdout)
                        break
                except subprocess.SubprocessError:
                    pass
            if set(poses) != set(names):
                raise RuntimeError('Incomplete Gazebo pose stream')
            results = []
            for name, (_, _, _, inserted, seated) in cases.items():
                score = score_seating(poses[name+'_peg'], poses[name+'_hole'])
                passed = score['candidate_inserted'] == inserted and score['candidate_fully_seated'] == seated
                results.append(dict(case=name, passed=passed, score=score))
            report = dict(kind='static_pose_transport_acceptance_not_dynamic_physics', cases=results,
                          passed=all(row['passed'] for row in results))
            (args.output / 'result.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
            if not report['passed']:
                raise RuntimeError('Geometry acceptance failed')
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == '__main__':
    main()
