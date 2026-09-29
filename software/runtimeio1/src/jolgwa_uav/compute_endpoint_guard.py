"""Opt-in PC simulation HTTP boundary; physical/legacy transport is unchanged.

Loopback and operator-pinned /proc identities are test isolation, not hostile
process authentication. No proxy, redirect, remote fallback, or lease renewal.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener, urlopen
import uuid

PC_URL = 'http://127.0.0.1:8775'


class NoComputeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('PC compute redirect forbidden')


def require_local_url(request):
    url = request.full_url if hasattr(request, 'full_url') else str(request)
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.netloc != '127.0.0.1:8775'
            or parsed.username is not None or parsed.password is not None or parsed.fragment
            or any(ord(c) < 33 for c in url)):
        raise ValueError('PC compute requires exact loopback127.0.0.1:8775')
    return parsed


def _read_pinned_process(identity):
    root = Path('/proc') / str(identity['pid'])
    def ticks():
        return (root/'stat').read_text().rsplit(')', 1)[1].split()[19]
    before = ticks()
    markers = (root/'environ').read_bytes().split(b'\0')
    required = (b'JOLGWA_SIM_COMPUTE_LOCATION=pc_local',
                ('JOLGWA_SIM_BACKEND_RUN_ID='+identity['run_id']).encode())
    if any(value not in markers for value in required):
        raise ValueError('PC backend ownership marker changed')
    socket = 'socket:['+identity['listener_socket_inode']+']'
    present = False
    for descriptor in (root/'fd').iterdir():
        try:
            present |= os.readlink(descriptor) == socket
        except OSError:
            continue
    if not present or before != ticks():
        raise ValueError('PC backend listener/process changed')
    return dict(identity, start_ticks=before,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        executable=str((root/'exe').resolve()),
        cmdline_sha256=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest())


class PinnedLocalBackend:
    """One immutable process/runtime epoch. Any identity failure retires it."""
    def __init__(self, identity, backend_epoch, *, identity_reader=None):
        if (not isinstance(identity, dict) or type(identity.get('pid')) is not int
                or identity['pid'] <= 0 or identity.get('listener') != '127.0.0.1:8775'
                or str(uuid.UUID(backend_epoch)) != backend_epoch):
            raise ValueError('Valid pinned PC backend evidence required')
        self._identity = copy.deepcopy(identity)
        self._backend_epoch = backend_epoch
        self._reader = identity_reader or _read_pinned_process
        self._lock = threading.Lock()
        self.failure = ''
        self.check()

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def backend_epoch(self):
        return self._backend_epoch

    def check(self, response=None):
        with self._lock:
            if self.failure:
                raise ValueError(self.failure)
            try:
                if self._reader(self.identity) != self._identity:
                    raise ValueError('PC backend pinned process identity changed')
                if response is not None:
                    if (response.headers.get('X-Jolgwa-Backend-Epoch') != self.backend_epoch
                            or response.headers.get('X-Jolgwa-Compute-Location') != 'pc_local'):
                        raise ValueError('PC backend response epoch/location mismatch')
            except Exception as exc:
                self.failure = 'PC backend identity retired: '+str(exc)
                raise ValueError(self.failure) from exc


def compute_guard_from_environment(endpoint, environment=None):
    env = os.environ if environment is None else environment
    location = env.get('JOLGWA_SIM_COMPUTE_LOCATION', 'jetson')
    if location == 'jetson':
        return None
    if location != 'pc_local':
        raise ValueError('Unknown compute location')
    require_local_url(endpoint)
    if env.get('ROS_DOMAIN_ID') != '148' or env.get('ROS_LOCALHOST_ONLY') != '1':
        raise ValueError('PC backend guard requires isolated SITL domain148')
    raw = env.get('JOLGWA_PC_BACKEND_ATTESTATION', '')
    if not raw or len(raw) > 16384:
        raise ValueError('PC backend immutable attestation missing')
    value = json.loads(raw)
    if (value.get('compute_location') != 'pc_local'
            or value.get('backend_url') != PC_URL
            or value.get('jetson_hardware_verified') is not False):
        raise ValueError('PC backend attestation placement mismatch')
    return PinnedLocalBackend(value['local_backend_identity'], value['backend_epoch'])


@contextmanager
def open_compute_request(request, *, timeout, location='jetson', opener=urlopen, guard=None):
    if location == 'jetson':
        with opener(request, timeout=timeout) as response:
            yield response
        return
    if location != 'pc_local':
        raise ValueError('Unknown compute location')
    parsed = require_local_url(request)
    method = request.get_method() if hasattr(request, 'get_method') else 'GET'
    if guard is None and (parsed.path != '/health' or method != 'GET'):
        raise ValueError('PC inference requires pinned backend identity')
    if guard is not None:
        guard.check()
    # Build a private opener; never install or modify global HTTP behavior.
    private = build_opener(ProxyHandler({}), NoComputeRedirect())
    with private.open(request, timeout=timeout) as response:
        if response.geturl() != (request.full_url if hasattr(request, 'full_url') else str(request)):
            raise ValueError('PC compute response URL changed')
        if guard is not None:
            guard.check(response)
        yield response
        if guard is not None:
            guard.check(response)
