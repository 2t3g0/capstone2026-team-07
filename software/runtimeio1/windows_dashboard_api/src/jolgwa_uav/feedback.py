from __future__ import annotations

import base64
import platform
import queue
import shutil
import subprocess
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_MAX_CHARS = 240
DEFAULT_DUPLICATE_WINDOW_SECONDS = 2.0
DEFAULT_QUEUE_SIZE = 16
_STOP = object()

_POWERSHELL_SPEAK_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "$text=[Text.Encoding]::UTF8.GetString("
    "[Convert]::FromBase64String($args[0]));"
    "Add-Type -AssemblyName System.Speech;"
    "$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "try{$speaker.Speak($text)}finally{$speaker.Dispose()}"
)


class FeedbackError(RuntimeError):
    """Base error for local speech feedback."""


class SpeechBackendUnavailableError(FeedbackError):
    """Raised when the host has no supported local TTS executable."""


class SpeechFeedbackClosedError(FeedbackError):
    """Raised when speech is requested after the worker has been closed."""


class SpeechRunner(Protocol):
    def run(self, argv: Sequence[str]) -> Any: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechBackend:
    name: str
    executable: str

    def command(self, message: str) -> list[str]:
        """Build a shell-free command for one already-sanitized message."""

        if self.name == "system-speech":
            encoded = base64.b64encode(message.encode("utf-8")).decode("ascii")
            return [
                self.executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _POWERSHELL_SPEAK_SCRIPT,
                encoded,
            ]
        if self.name == "spd-say":
            return [self.executable, "--wait", "--", message]
        if self.name in {"espeak", "espeak-ng"}:
            return [self.executable, "--", message]
        raise ValueError(f"unsupported speech backend: {self.name}")


def detect_speech_backend(
    *,
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> SpeechBackend:
    """Select the preferred local TTS backend for Windows or Ubuntu/Linux."""

    host = (system or platform.system()).casefold()
    if host == "windows":
        for executable in ("powershell.exe", "powershell", "pwsh"):
            resolved = which(executable)
            if resolved:
                return SpeechBackend("system-speech", resolved)
    elif host == "linux":
        for executable in ("spd-say", "espeak", "espeak-ng"):
            resolved = which(executable)
            if resolved:
                return SpeechBackend(executable, resolved)

    raise SpeechBackendUnavailableError(
        "no supported local TTS backend was found "
        "(Windows: PowerShell/System.Speech; Ubuntu: spd-say or espeak)"
    )


def sanitize_feedback_message(message: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Normalize status text, remove controls, and enforce a hard length limit."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if max_chars < 4:
        raise ValueError("max_chars must be at least 4")

    printable = "".join(
        " " if char.isspace() else char
        for char in unicodedata.normalize("NFC", message)
        if not unicodedata.category(char).startswith("C") or char.isspace()
    )
    normalized = " ".join(printable.split())
    if len(normalized) > max_chars:
        normalized = normalized[: max_chars - 3].rstrip() + "..."
    return normalized


class SubprocessSpeechRunner:
    """Run one local TTS process at a time and allow active speech to be stopped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def run(self, argv: Sequence[str]) -> int:
        with self._lock:
            process = subprocess.Popen(
                list(argv),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process = process
        try:
            return process.wait()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


@dataclass(frozen=True, slots=True)
class _SpeechItem:
    message: str
    generation: int


class SpeechFeedback:
    """Asynchronous, bounded, single-worker status speech queue."""

    def __init__(
        self,
        *,
        backend: SpeechBackend | None = None,
        runner: SpeechRunner | Callable[[Sequence[str]], Any] | None = None,
        duplicate_window_seconds: float = DEFAULT_DUPLICATE_WINDOW_SECONDS,
        max_chars: int = DEFAULT_MAX_CHARS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if duplicate_window_seconds < 0:
            raise ValueError("duplicate_window_seconds cannot be negative")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")

        self.backend = backend or detect_speech_backend()
        self._runner = runner or SubprocessSpeechRunner()
        self._duplicate_window = duplicate_window_seconds
        self._max_chars = max_chars
        self._clock = clock
        self._queue: queue.Queue[_SpeechItem | object] = queue.Queue(queue_size)
        self._state_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._generation = 0
        self._recent: dict[str, float] = {}
        self.last_error: Exception | None = None
        self._worker = threading.Thread(
            target=self._work,
            name="jolgwa-tts-feedback",
            daemon=True,
        )
        self._worker.start()

    def speak(self, message: str) -> bool:
        """Queue a message without blocking; return False when it is a duplicate."""

        normalized = sanitize_feedback_message(message, max_chars=self._max_chars)
        if not normalized:
            return False

        now = self._clock()
        with self._state_lock:
            if self._closed:
                raise SpeechFeedbackClosedError("speech feedback is closed")
            last_spoken = self._recent.get(normalized)
            if (
                last_spoken is not None
                and now - last_spoken < self._duplicate_window
            ):
                return False
            self._recent = {
                text: timestamp
                for text, timestamp in self._recent.items()
                if now - timestamp < self._duplicate_window
            }
            self._recent[normalized] = now
            item = _SpeechItem(normalized, self._generation)
            self._idle.clear()

        self._put_latest(item)
        return True

    def stop(self) -> None:
        """Cancel active speech and discard pending messages; future speech is allowed."""

        with self._state_lock:
            if self._closed:
                return
            self._generation += 1
            self._recent.clear()
        self._clear_pending()
        stop = getattr(self._runner, "stop", None)
        if callable(stop):
            stop()
        self._set_idle_if_empty()

    def close(self, *, wait: bool = True, timeout: float | None = 2.0) -> None:
        """Permanently stop feedback and terminate its worker thread."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._recent.clear()
        self._clear_pending()
        stop = getattr(self._runner, "stop", None)
        if callable(stop):
            stop()
        self._put_control(_STOP)
        if wait:
            self._worker.join(timeout)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait until every accepted message has completed or been discarded."""

        return self._idle.wait(timeout)

    def __enter__(self) -> SpeechFeedback:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _SpeechItem)
                with self._state_lock:
                    should_run = not self._closed and item.generation == self._generation
                if should_run:
                    try:
                        self._invoke_runner(self.backend.command(item.message))
                    except Exception as exc:  # Keep later status feedback alive.
                        self.last_error = exc
            finally:
                self._queue.task_done()
                self._set_idle_if_empty()

    def _invoke_runner(self, argv: Sequence[str]) -> Any:
        run = getattr(self._runner, "run", None)
        if callable(run):
            return run(argv)
        if callable(self._runner):
            return self._runner(argv)
        raise TypeError("runner must be callable or provide run(argv)")

    def _put_latest(self, item: _SpeechItem) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    continue

    def _put_control(self, item: object) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    continue

    def _clear_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def _set_idle_if_empty(self) -> None:
        with self._queue.mutex:
            unfinished = self._queue.unfinished_tasks
        if unfinished == 0:
            self._idle.set()
