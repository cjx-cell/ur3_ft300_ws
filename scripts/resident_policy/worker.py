"""JSON-lines offline worker. Keeps one checkpoint loaded; no robot connection.

stdout is protocol only; model/library logs go to stderr. Begin returns a fresh
session token. Infer requires that token and the exact next sequence number.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from session import EpisodeSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi05', 'pap_moe'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with redirect_stdout(sys.stderr):
        import numpy as np
        from backend import ResidentBackend
        session = EpisodeSession(ResidentBackend(args.kind, args.checkpoint))
    print(json.dumps({'ready': True, 'mode': 'OFFLINE_CANDIDATE_ONLY'}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                op = request['op']
                if op == 'begin':
                    response = {'token': session.begin(request['seed'])}
                elif op == 'infer':
                    result = session.infer(request['token'], request['sequence'], request['trace'])
                    # Use server-issued token/counter, never an arbitrary output filename.
                    path = args.output / f'{session.token}_{session.sequence - 1:05d}.npz'
                    with path.open('xb') as stream:
                        np.savez(stream, **result)
                    response = {'token': session.token, 'sequence': session.sequence - 1,
                                'result': str(path)}
                elif op == 'end':
                    session.end(request['token'])
                    response = {'ended': True}
                else:
                    raise ValueError('Unknown operation')
            print(json.dumps(response), flush=True)
        except Exception as exc:
            # Fail closed at transport level too: no invisible retries after I/O errors.
            print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)}), flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

