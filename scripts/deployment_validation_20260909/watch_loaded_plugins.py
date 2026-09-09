"""Observe one actual plugin mapping per new closed-loop trial; no ROS writes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/ubuntu/ur3_ft300_ws')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-dir',type=Path,required=True)
    args=parser.parse_args()
    seen=set()
    audit=args.batch_dir/'loaded_plugin_runtime.jsonl'
    if audit.exists():
        for line in audit.read_text().splitlines():
            row=json.loads(line)['closed_state']
            seen.add((row['model'],row['episode']))
    deadline=time.monotonic()+4*3600
    last_heartbeat=0.
    while time.monotonic()<deadline:
        state=json.loads((args.batch_dir/'status.json').read_text())
        if state['status']=='completed':
            print('COMPLETE',len(seen),'trial plugin checks',flush=True)
            return
        if state['status'].startswith('blocked'):
            raise RuntimeError(state)
        if time.monotonic()-last_heartbeat>=60:
            print(datetime.now().isoformat(),'watching',state['status'],len(seen),'checked',flush=True)
            last_heartbeat=time.monotonic()
        key=(state.get('model'),state.get('episode'))
        if state['status']=='running' and key not in seen:
            ready=False
            for p in Path('/proc').glob('[0-9]*'):
                try:
                    if b'ROS_DOMAIN_ID=84' not in (p/'environ').read_bytes().split(b'\0'):continue
                    if 'libgz_hardware_plugins.so' in (p/'maps').read_text():ready=True;break
                except (FileNotFoundError,PermissionError,ProcessLookupError):pass
            if ready:
                subprocess.run(['/usr/bin/python3',ROOT/'scripts/deployment_validation_20260909/audit_loaded_plugin.py',
                                '--batch-dir',args.batch_dir],check=True)
                seen.add(key)
        time.sleep(2)
    raise TimeoutError('plugin observation deadline')


if __name__=='__main__':main()
