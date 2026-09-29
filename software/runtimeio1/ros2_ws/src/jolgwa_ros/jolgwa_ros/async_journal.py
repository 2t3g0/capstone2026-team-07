"""Bounded FIFO journal: disk I/O never runs in a flight/inference callback.

write/flush acknowledge queue submission, not durable storage. A journal is
diagnostics, never proof of USB TX, ACK, or fresh FC state. Critical records fail
explicitly on capacity/error; replaceable snapshots may be dropped with counters.
Only the writer thread owns the stream after construction.
"""
from collections import deque
import json
import threading
import time


class AsyncJournal:
    def __init__(self, stream, *, max_bytes=8*1024*1024, max_records=2048,
                 reserve_bytes=256*1024, reserve_records=64, close_timeout_s=3.):
        if not (0 <= reserve_bytes < max_bytes and 0 <= reserve_records < max_records):
            raise ValueError('invalid journal capacity')
        self._stream = stream
        self.name = getattr(stream, 'name', None)
        self._max_bytes, self._max_records = max_bytes, max_records
        self._reserve_bytes, self._reserve_records = reserve_bytes, reserve_records
        self._close_timeout = close_timeout_s
        self._cv = threading.Condition()
        self._queue = deque()
        self._bytes = self._records = self._accepted = self._written = 0
        self._dropped = self._reported_dropped = 0
        self._inflight_at = None
        self._error = ''
        self._closing = False
        self._last_write_s = self._max_write_s = 0.
        self._thread = threading.Thread(target=self._run, name='journal-writer', daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, path):
        return cls(path.open('x', encoding='utf-8', buffering=1))

    def write(self, text):
        self._submit(text, diagnostic=False)
        return len(text)

    def write_diagnostic(self, text):
        return self._submit(text, diagnostic=True)

    def _submit(self, text, diagnostic):
        if not isinstance(text, str):
            raise TypeError('journal records must be text')
        size = len(text.encode('utf-8'))
        with self._cv:
            limit_bytes = self._max_bytes - (self._reserve_bytes if diagnostic else 0)
            limit_records = self._max_records - (self._reserve_records if diagnostic else 0)
            reason = (self._error or ('journal_closed' if self._closing else '') or
                      ('journal_queue_full' if self._bytes+size > limit_bytes or
                       self._records >= limit_records else ''))
            if reason:
                if diagnostic:
                    self._dropped += 1
                    return False
                raise OSError(reason)
            self._queue.append((text, size, time.monotonic()))
            self._bytes += size
            self._records += 1
            self._accepted += 1
            self._cv.notify()
            return True

    def flush(self):
        """Nonblocking wakeup. This deliberately does not assert disk durability."""
        with self._cv:
            if self._error:
                raise OSError(self._error)
            self._cv.notify()

    def status(self):
        with self._cv:
            oldest = self._inflight_at or (self._queue[0][2] if self._queue else None)
            return dict(accepted=self._accepted, written=self._written,
                        pending_records=self._records, pending_bytes=self._bytes,
                        dropped_diagnostics=self._dropped, error=self._error,
                        oldest_pending_s=max(0., time.monotonic()-oldest) if oldest else 0.,
                        max_disk_write_s=self._max_write_s, last_disk_write_s=self._last_write_s)

    def drain(self, timeout_s=3.):
        """Explicit bounded offline/shutdown barrier; never call in flight callbacks."""
        with self._cv:
            done = self._cv.wait_for(lambda: self._records == 0 or self._error, timeout_s)
            if self._error:
                raise OSError(self._error)
            if not done:
                raise TimeoutError('journal_drain_timeout; pending records are not durable')

    def _run(self):
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._queue or self._closing)
                    if not self._queue:
                        break
                    text, size, queued_at = self._queue.popleft()
                    self._inflight_at = queued_at
                    dropped = self._dropped
                started = time.monotonic()
                if dropped != self._reported_dropped:
                    self._stream.write(json.dumps(dict(event='journal_diagnostics_dropped',
                        total=dropped, monotonic_s=started))+'\n')
                    self._reported_dropped = dropped
                self._stream.write(text)
                self._stream.flush()
                elapsed = time.monotonic()-started
                with self._cv:
                    self._last_write_s = elapsed
                    self._max_write_s = max(self._max_write_s, elapsed)
                    self._written += 1
                    self._records -= 1
                    self._bytes -= size
                    self._inflight_at = None
                    self._cv.notify_all()
            self._stream.write(json.dumps(dict(event='journal_writer_summary', **self.status()))+'\n')
            self._stream.flush()
        except Exception as exc:
            with self._cv:
                self._error = 'journal_io_failed:'+type(exc).__name__+':'+str(exc)
                self._cv.notify_all()
        finally:
            try:
                self._stream.close()
            except Exception as exc:
                with self._cv:
                    self._error = self._error or 'journal_close_failed:'+str(exc)

    def close(self):
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join(timeout=self._close_timeout)
        if self._thread.is_alive():
            raise TimeoutError('journal_drain_timeout; pending records are not durable')
        if self._error:
            raise OSError(self._error)
