import json
import socket
from threading import Thread
import unittest

from daemon import serve_connection
from session import EpisodeSession
from test_session import FakeBackend


class FakeWorker:
    def __init__(self):
        self.session = EpisodeSession(FakeBackend())
        self.stdin = self.stdout = self
        self.request = ''
        self.reply = ''

    def write(self, text):
        self.request += text

    def flush(self):
        request = json.loads(self.request)
        self.request = ''
        if request['op'] == 'begin':
            response = dict(token=self.session.begin(request['seed']))
        elif request['op'] == 'end':
            self.session.end(request['token'])
            response = dict(ended=True)
        else:
            response = dict(action=self.session.infer(request['token'], request['sequence'], request['trace']))
        self.reply = json.dumps(response) + '\n'

    def readline(self):
        return self.reply

    def poll(self):
        return None


class TransportTests(unittest.TestCase):
    def test_disconnect_resets_session_but_reuses_worker(self):
        worker = FakeWorker()
        tokens = []
        for seed in (0, 1, 0):
            client, server = socket.socketpair()
            client.settimeout(2)
            errors = []

            def serve():
                try:
                    serve_connection(worker, server, dict(ready=True))
                except Exception as exc:
                    errors.append(exc)

            thread = Thread(target=serve, daemon=True)
            thread.start()
            with client, client.makefile('rw') as stream:
                self.assertTrue(json.loads(stream.readline())['ready'])
                stream.write(json.dumps(dict(op='begin', seed=seed)) + '\n')
                stream.flush()
                token = json.loads(stream.readline())['token']
                tokens.append(token)
                stream.write(json.dumps(dict(op='infer', token=token, sequence=0, trace=8)) + '\n')
                stream.flush()
                self.assertEqual(json.loads(stream.readline())['action'], [seed, 8])
                # No end message: simulate episode process teardown.
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIsNone(worker.session.token)
        self.assertEqual(len(set(tokens)), 3)


if __name__ == '__main__':
    unittest.main()
