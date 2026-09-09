"""Single-client Unix socket relay to the validated resident worker.

The worker owns one GPU model; disconnect ends its episode, never its process.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def forward(worker, request):
    worker.stdin.write(json.dumps(request) + '\n')
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f'Worker exited: {worker.poll()}')
    response = json.loads(line)
    if 'error' in response:
        raise RuntimeError(response)
    return response


def serve_connection(worker, connection, hello):
    token = None
    with connection:
        stream = connection.makefile('rw')
        try:
            stream.write(json.dumps(hello) + '\n')
            stream.flush()
            for line in stream:
                request = json.loads(line)
                op = request.get('op')
                if op == 'begin':
                    if token is not None:
                        raise RuntimeError('Connection already owns an episode')
                elif token is None or request.get('token') != token:
                    raise RuntimeError('Connection does not own this episode')
                response = forward(worker, request)
                if op == 'begin':
                    token = response['token']
                elif op == 'end':
                    token = None
                stream.write(json.dumps(response) + '\n')
                stream.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Disconnect during Flow: discard reply, release session, retain model.
            pass
        finally:
            try:
                stream.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if token is not None and worker.poll() is None:
                forward(worker, dict(op='end', token=token))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi05', 'pap_moe'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if args.socket.exists():
        raise FileExistsError(args.socket)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    with (args.output / 'worker.log').open('x') as log:
        worker = subprocess.Popen([sys.executable, str(HERE / 'worker.py'), '--kind', args.kind,
            '--checkpoint', str(args.checkpoint), '--output', str(args.output / 'worker')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            ready = json.loads(worker.stdout.readline())
            if not ready.get('ready'):
                raise RuntimeError(ready)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(args.socket))
                os.chmod(args.socket, 0o600)
                server.listen(1)
                (args.output / 'ready.json').write_text(json.dumps(dict(
                    kind=args.kind, checkpoint=str(args.checkpoint), worker_pid=worker.pid,
                    socket=str(args.socket), mode='engineering_resident_candidate')))
                while True:
                    connection, _ = server.accept()
                    serve_connection(worker, connection, dict(kind=args.kind,
                        checkpoint=str(args.checkpoint), ready=True))
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            if args.socket.exists():
                args.socket.unlink()


if __name__ == '__main__':
    main()
