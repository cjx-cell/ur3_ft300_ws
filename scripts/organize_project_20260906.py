"""Reversible, manifest-driven organization. Never delete model/data contents."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
RECORD = ROOT/'maintenance/20260906'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    RECORD.mkdir(parents=True, exist_ok=True)
    manifest = RECORD/'organization_manifest.json'
    if manifest.exists():
        raise FileExistsError('Existing manifest: do not repeat a completed organization')
    texts, linked = [], []
    for folder in [ROOT/'scripts', ROOT/'pap_moe_framework/scripts', ROOT/'src', ROOT/'ai-models', ROOT/'outputs']:
        for directory, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
            for name in files + dirs:
                p = Path(directory)/name
                if p.is_symlink():
                    linked.append(str(p.resolve()))
                elif p != Path(__file__).resolve() and p.is_file() and p.suffix in ['.sh', '.py', '.json', '.yaml'] and p.stat().st_size < 2_000_000:
                    texts.append(p.read_text(errors='replace'))
    haystack = '\n'.join(texts + linked)
    plans, preserved = [], []
    for p in sorted((ROOT/'artifacts').iterdir()):
        if p.name in ['README.md', '_archive']:
            continue
        referenced = p.name in haystack
        recent = bool(re.search(r'2026090[4-9]', p.name))
        if referenced or recent:
            preserved.append(dict(path=str(p.relative_to(ROOT)), reason='referenced' if referenced else 'recent'))
            continue
        if p.name in ['invalid_training_runs_20260818', 'quarantine_bad_lr_20260827']:
            destination = ROOT/'.cleanup_trash/20260906'/p.name
            reason = 'Previously explicitly rejected training artifacts; recoverable quarantine'
        else:
            destination = ROOT/'artifacts/_archive/through_20260903'/p.name
            reason = 'Historical evidence; not found in scanned executable/config references'
        plans.append(dict(source=str(p.relative_to(ROOT)), destination=str(destination.relative_to(ROOT)), reason=reason))
    for p in sorted((ROOT/'docs').iterdir()):
        if p.name == 'archive':
            continue
        plans.append(dict(source=str(p.relative_to(ROOT)),
                          destination=str(Path('docs/archive/pre_cleanup_20260906')/p.name),
                          reason='Original historical document preserved unchanged'))
    for name in ['PI0_TRAINING_ANALYSIS.md', 'FIXES_2026-06-23.md']:
        if (ROOT/name).exists():
            plans.append(dict(source=name, destination=f'docs/archive/pre_cleanup_20260906/root_notes/{name}',
                              reason='Historical project-root note'))
    for item in plans:
        src, dst = ROOT/item['source'], ROOT/item['destination']
        assert src.exists() or src.is_symlink()
        assert not dst.exists() and not dst.is_symlink(), str(dst)
        assert src.parent in [ROOT, ROOT/'docs', ROOT/'artifacts']
        if src.is_file() and not src.is_symlink() and src.stat().st_size < 2_000_000:
            item['sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
    report = dict(scope='docs, artifacts and two historical root notes; no datasets/checkpoints outside artifacts moved',
                  reversible=True, permanent_deletions=0, preserved=preserved, moves=plans,
                  caveat='Reference scan is conservative but cannot prove absence of dynamically constructed old paths. Use manifest to locate/restore archived paths.')
    if not args.apply:
        (RECORD/'organization_plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        with (RECORD/'completed_moves.jsonl').open('x') as log:
            for item in plans:
                src, dst = ROOT/item['source'], ROOT/item['destination']
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                assert dst.exists() or dst.is_symlink()
                if 'sha256' in item:
                    assert hashlib.sha256(dst.read_bytes()).hexdigest() == item['sha256']
                log.write(json.dumps(item, ensure_ascii=False)+'\n')
                log.flush()
    print(json.dumps(dict(apply=args.apply, move_count=len(plans), preserved_count=len(preserved),
                          quarantine=[p['source'] for p in plans if p['destination'].startswith('.cleanup_trash')]), indent=2))


if __name__ == '__main__':
    main()
