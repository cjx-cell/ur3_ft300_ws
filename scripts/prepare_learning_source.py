#!/usr/bin/env python3
"""Prepare a NEW pinned LeRobot source tree; never overwrite an existing environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    target=a.destination.expanduser().resolve()
    if target.exists():raise SystemExit('Refusing existing destination: '+str(target))
    manifest=json.loads((root/'learning/upstream.json').read_text())
    for entry in manifest['files']:
        rel=Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe overlay path')
        source=root/'learning/lerobot_overlay'/rel
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Overlay digest mismatch: '+str(rel))
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',manifest['repository'],str(target)],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',manifest['commit']],check=True)
    for entry in manifest['files']:
        rel=Path(entry['path']);dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'learning/lerobot_overlay'/rel,dest)
    print('Prepared source only:',target)
    print('In your separate ML environment: python -m pip install -e',str(target))


if __name__=='__main__':main()
