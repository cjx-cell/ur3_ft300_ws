"""Read-only live verification of the plugin actually mapped in ROS domain 84."""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
AB = ROOT/'artifacts/deployment_validation_20260909_interface_ab'
EXPECTED = ROOT/'artifacts/deployment_validation_20260909_gripper_candidate/install/gz_ros2_control/lib/libgz_hardware_plugins.so'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-dir',type=Path,default=AB/'closed_5pos_seed0')
    args=parser.parse_args()
    found = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            env = path.joinpath('environ').read_bytes().split(b'\0')
            if b'ROS_DOMAIN_ID=84' not in env: continue
            maps = [line for line in path.joinpath('maps').read_text().splitlines() if 'libgz_hardware_plugins.so' in line]
            if not maps: continue
            libraries = sorted({line.split()[-1] for line in maps})
            assert libraries == [str(EXPECTED)], (path, libraries)
            found.append(dict(pid=int(path.name), partition=[x.decode() for x in env if x.startswith(b'IGN_PARTITION=')],
                              libraries=libraries, command=path.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    assert len(found) == 1, f'Expected exactly one loaded simulator in domain 84, found {found}'
    state = json.loads((args.batch_dir/'status.json').read_text())
    record = dict(time=datetime.now().isoformat(), loaded=found, closed_state=state,
                  sha256=hashlib.sha256(EXPECTED.read_bytes()).hexdigest())
    with (args.batch_dir/'loaded_plugin_runtime.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
