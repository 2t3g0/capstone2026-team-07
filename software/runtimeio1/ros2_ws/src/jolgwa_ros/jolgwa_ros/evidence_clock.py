"""Same-host, same-boot monotonic incident leases; never cross-host clock sync.

Used only by the new guarded incident policy. A Linux boot/time-namespace ID
must match at both endpoints. Other platforms or missing identity fail closed.
The deadline belongs to the original observation and is never renewed by a
subscriber, a service request, queue handling, or a repeated ACTIVE event.
"""
from __future__ import annotations

from pathlib import Path
import re
import time
import uuid


_CLOCK_ID = re.compile(r"linux-monotonic:[0-9a-f-]{36}:[1-9][0-9]*\Z")
MAX_INCIDENT_LEASE_NS = 2_000_000_000


def local_evidence_clock_id() -> str:
    try:
        boot = str(uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        namespace = Path("/proc/self/ns/time").stat().st_ino
        if namespace <= 0:
            return ""
        return f"linux-monotonic:{boot}:{namespace}"
    except (OSError, ValueError):
        return ""


def evidence_deadline_error(clock_id, expires_ns, *, local_clock_id=None, now_ns=None):
    """Return an error string, or empty for an unexpired <=2s local lease."""
    expected = local_evidence_clock_id() if local_clock_id is None else local_clock_id
    now = time.monotonic_ns() if now_ns is None else now_ns
    if (not isinstance(expected, str) or not _CLOCK_ID.fullmatch(expected)
            or clock_id != expected):
        return "incident_evidence_clock_mismatch"
    if (type(expires_ns) is not int or type(now) is not int
            or not 0 <= now < 2**64 or not 0 < expires_ns < 2**64):
        return "incident_evidence_deadline_invalid"
    if expires_ns <= now:
        return "incident_evidence_expired"
    if expires_ns-now > MAX_INCIDENT_LEASE_NS:
        return "incident_evidence_lease_too_long"
    return ""
