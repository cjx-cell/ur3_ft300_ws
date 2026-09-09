"""Freeze the entire candidate code before starting engineering canaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['s4', 'pi05', 's3'], default=['s4', 'pi05', 's3'])
    parser.add_argument('--reuse-cold-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    frozen = args.output / 'code'
    frozen.mkdir()
    hashes = {}
    for path in sorted(source.glob('*.py')):
        target = frozen / path.name
        shutil.copy2(path, target)
        hashes[str(target)] = hashlib.sha256(target.read_bytes()).hexdigest()
        target.chmod(0o444)
    (args.output / 'code_manifest.json').write_text(json.dumps(hashes, indent=2))
    print(f'Frozen candidate: {frozen}', flush=True)
    with (args.output / 'orchestrator.log').open('x') as log:
        command = [sys.executable, str(frozen / 'paired_live.py'), '--output', str(args.output / 'runs'),
                   '--models', *args.models]
        if args.reuse_cold_root:
            command += ['--reuse-cold-root', str(args.reuse_cold_root.resolve())]
        result = subprocess.call(command, stdout=log, stderr=log)
    (args.output / 'completion.json').write_text(json.dumps(dict(exit_code=result)))
    raise SystemExit(result)


if __name__ == '__main__':
    main()
