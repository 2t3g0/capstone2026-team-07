"""Same-user, local Unix control socket for explicit demo mode selection.

The socket only queues bounded requests to the single USB owner. It never
publishes MAVLink, arms, takes off, or interprets a good observation as consent.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import struct
import threading
import time
import uuid

MAX_REQUEST = 4096


def default_socket_path():
    return Path('/run/user') / str(os.getuid()) / 'jolgwa-small-demo' / 'control.sock'


def validate_request(value, session_id):
    if not isinstance(value, dict) or value.get('session_id') != session_id:
        raise ValueError('runtime_session_changed_refresh_status')
    if value.get('kind') == 'observe' and set(value) == {'kind', 'session_id'}:
        return {'kind': 'observe'}
    if (value.get('kind') != 'start' or value.get('observe_passed') is not True
            or set(value) != {'kind', 'session_id', 'observe_passed', 'token'}):
        raise ValueError('explicit_observe_passed_and_start_required')
    token = value.get('token')
    if not isinstance(token, str) or str(uuid.UUID(token)) != token:
        raise ValueError('fresh_uuid_start_token_required')
    return {'kind': 'start', 'token': token, 'observe_passed': True}


class DemoControlServer:
    def __init__(self, worker, socket_path=None):
        self.worker = worker
        self.path = Path(socket_path) if socket_path else default_socket_path()
        self.session_id = str(uuid.uuid4())
        self.server = self.thread = None
        self.socket_inode = None

    def snapshot(self):
        return {'session_id': self.session_id, 'socket_path': str(self.path),
                'status': self.worker.last_demo_status()}

    def dispatch(self, value):
        if value == {'kind': 'status'}:
            return self.snapshot()
        request = validate_request(value, self.session_id)
        now = time.monotonic_ns()
        request.update(submitted_monotonic_ns=now, expires_monotonic_ns=now+200_000_000)
        request_id = self.worker.submit_request(request)
        # ACK means processed by the owner, not just received by this socket.
        deadline = time.monotonic()+.8
        while time.monotonic() < deadline:
            result = self.snapshot()
            last = result['status'].get('last_request') or {}
            if last.get('id', last.get('request_id')) == request_id:
                return result
            time.sleep(.01)
        return {'session_id': self.session_id, 'request_id': request_id,
                'error': 'owner_ack_timeout_status_unknown_request_will_not_wait_for_readiness',
                'status': self.worker.last_demo_status()}

    def start(self):
        if os.name != 'posix' or not hasattr(socket, 'SO_PEERCRED'):
            raise RuntimeError('Linux_same_user_socket_required')
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError('control_directory_must_be_private_owned_0700')
        if self.path.exists() or self.path.is_symlink():
            # Do not delete another process's socket or a stale unresolved link.
            raise FileExistsError('control_socket_exists_refusing_duplicate_runtime')
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(1.)
                _, uid, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != os.getuid():
                    return
                try:
                    raw = self.rfile.readline(MAX_REQUEST+1)
                    if len(raw) > MAX_REQUEST or not raw.endswith(b'\n'):
                        raise ValueError('bounded_json_line_required')
                    result = owner.dispatch(json.loads(raw))
                except (ValueError, TypeError, OSError, RuntimeError) as exc:
                    result = {'error': str(exc), 'session_id': owner.session_id}
                self.wfile.write(json.dumps(result, allow_nan=False).encode()+b'\n')

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            request_queue_size = 2

        self.server = Server(str(self.path), Handler)
        os.chmod(self.path, 0o600)
        self.socket_inode = self.path.lstat().st_ino
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .05}, daemon=True)
        self.thread.start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1.)
            # Only the exact socket inode created by this runtime is removed.
            if self.path.exists():
                info = self.path.lstat()
                if stat.S_ISSOCK(info.st_mode) and info.st_ino == self.socket_inode and info.st_uid == os.getuid():
                    self.path.unlink()


def send_request(path, value):
    encoded = json.dumps(value, allow_nan=False).encode()+b'\n'
    if len(encoded) > MAX_REQUEST:
        raise ValueError('request_too_large')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2.)
        connection.connect(str(path))
        connection.sendall(encoded)
        with connection.makefile('rb') as stream:
            raw = stream.readline(131073)
        if len(raw) > 131072 or not raw.endswith(b'\n'):
            raise ValueError('invalid_control_response')
        return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path)
    parser.add_argument('--mode', choices=('status', 'observe', 'auto-demo'), default='status')
    parser.add_argument('--observe-passed', action='store_true',
                        help='Operator confirms successful observe trial; never inferred by software')
    options = parser.parse_args(argv)
    if options.mode == 'auto-demo' and not options.observe_passed:
        parser.error('--mode auto-demo requires explicit --observe-passed')
    path = options.socket or default_socket_path()
    status = send_request(path, {'kind': 'status'})
    if options.mode == 'status':
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    request = {'kind': 'observe' if options.mode == 'observe' else 'start',
               'session_id': status['session_id']}
    if options.mode == 'auto-demo':
        request.update(observe_passed=True, token=str(uuid.uuid4()))
    result = send_request(path, request)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if 'error' in result:
        return 2
    last = result.get('status', {}).get('last_request') or {}
    return 0 if last.get('result') in ('accepted', 'applied') else 2


if __name__ == '__main__':
    raise SystemExit(main())
