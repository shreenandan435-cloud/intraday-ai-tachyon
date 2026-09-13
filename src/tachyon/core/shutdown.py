"""Graceful shutdown handler — signal handling and ordered teardown.

Provides:
- SIGTERM/SIGINT handling for systemd/k8s
- Ordered component teardown (Brain → Ingestor → I/O)
- Flush on shutdown (logs, telemetry, trades)
- Timeout enforcement with force-kill escalation
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import os
import signal
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Grace periods (seconds)
SIGTERM_GRACE_SECONDS: float = 30.0
SIGKILL_ESCALATION_SECONDS: float = 10.0
FLUSH_TIMEOUT_SECONDS: float = 5.0

# Exit codes
EXIT_CLEAN: int = 0
EXIT_SIGTERM: int = 143
EXIT_SIGINT: int = 130
EXIT_ERROR: int = 1


# ── Shutdown Coordinator ─────────────────────────────────────────────────────


@dataclass(slots=True)
class ShutdownPhase:
    """A phase in the ordered shutdown sequence."""

    name: str
    coroutine: Callable[[], Any]
    timeout: float
    critical: bool = True  # If False, failure doesn't block later phases


class ShutdownCoordinator:
    """Coordinates ordered shutdown across all system components.

    Phases execute sequentially with individual timeouts.
    A phase failure is logged but doesn't stop subsequent phases.
    """

    def __init__(self, clock: Clock = SYSTEM_CLOCK):
        self._clock = clock
        self._phases: list[ShutdownPhase] = []
        self._shutdown_event = threading.Event()
        self._shutdown_initiated = False
        self._exit_code = EXIT_CLEAN
        self._lock = threading.Lock()

    def register_phase(
        self,
        name: str,
        coroutine: Callable[[], Any],
        timeout: float = 10.0,
        critical: bool = True,
    ) -> None:
        """Register a shutdown phase.

        Phases run in registration order.
        """
        self._phases.append(ShutdownPhase(name, coroutine, timeout, critical))

    def request_shutdown(self, signal_name: str, exit_code: int) -> None:
        """Request shutdown from signal handler (thread-safe)."""
        with self._lock:
            if self._shutdown_initiated:
                _log.warning(
                    "shutdown.already_initiated",
                    signal=signal_name,
                    note="ignoring duplicate shutdown request",
                )
                return

            self._shutdown_initiated = True
            self._exit_code = exit_code
            _log.critical("shutdown.requested", signal=signal_name)
            self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    async def execute_shutdown(self) -> int:
        """Execute all registered shutdown phases in order.

        Returns the exit code to use for process termination.
        """
        _log.info("shutdown.sequence_started", phases=len(self._phases))

        for i, phase in enumerate(self._phases, 1):
            _log.info("shutdown.phase_start", phase=phase.name, index=i, total=len(self._phases))

            try:
                # Run with timeout
                # Safely invoke coroutine if it returns an awaitable
                coro_result = phase.coroutine()
                if coro_result is not None:
                    # If the result is a coroutine or Task, await it with timeout
                    if asyncio.iscoroutine(coro_result) or isinstance(coro_result, asyncio.Task):
                        await asyncio.wait_for(coro_result, timeout=phase.timeout)
                    else:
                        # Non‑awaitable result; log and continue
                        _log.debug("shutdown.phase_nonawaitable", phase=phase.name)
                else:
                    _log.debug("shutdown.phase_none", phase=phase.name)
                _log.info("shutdown.phase_complete", phase=phase.name, duration="<timeout")
            except TimeoutError:
                _log.error(
                    "shutdown.phase_timeout",
                    phase=phase.name,
                    timeout_seconds=phase.timeout,
                )
                if phase.critical:
                    _log.critical("shutdown.critical_phase_failed", phase=phase.name)
            except Exception as exc:  # noqa: BLE001
                _log.error(
                    "shutdown.phase_error",
                    phase=phase.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                if phase.critical:
                    _log.critical("shutdown.critical_phase_failed", phase=phase.name)

        _log.info("shutdown.sequence_completed", exit_code=self._exit_code)
        return self._exit_code


# ── Signal Handler Installation ──────────────────────────────────────────────


def install_signal_handlers(
    coordinator: ShutdownCoordinator,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Install SIGTERM/SIGINT handlers for graceful shutdown.

    On Windows, uses signal.signal (no loop.add_signal_handler).
    On POSIX, uses loop.add_signal_handler for proper async integration.
    """

    def handle_signal(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        exit_code = EXIT_SIGTERM if signum == signal.SIGTERM else EXIT_SIGINT
        coordinator.request_shutdown(signal_name, exit_code)

        # If loop is running, schedule the shutdown coroutine
        if loop and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                # Loop may be closing or not running
                loop.call_soon_threadsafe(asyncio.create_task, coordinator.execute_shutdown())

    # Install handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            if loop:
                loop.add_signal_handler(sig, handle_signal, sig, None)
            else:
                signal.signal(sig, handle_signal)
        except (NotImplementedError, AttributeError, ValueError):
            # Windows fallback or loop not ready
            signal.signal(sig, handle_signal)

    _log.info("shutdown.handlers_installed", signals=["SIGTERM", "SIGINT"])


# ── Graceful Shutdown Context Manager ────────────────────────────────────────


class GracefulShutdown:
    """Context manager for graceful shutdown with automatic signal handling.

    Usage:
        async with GracefulShutdown() as shutdown:
            shutdown.register_component("brain", brain.shutdown)
            shutdown.register_component("telemetry", telemetry.close)
            await main_loop()
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ):
        self._coordinator = ShutdownCoordinator(clock)
        self._loop = loop or asyncio.get_event_loop()
        self._entered = False

    async def __aenter__(self) -> GracefulShutdown:
        install_signal_handlers(self._coordinator, self._loop)
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._entered:
            self._coordinator.request_shutdown("context_exit", EXIT_CLEAN)

    def register_component(
        self,
        name: str,
        coroutine: Callable[[], Any],
        timeout: float = 10.0,
        critical: bool = True,
    ) -> None:
        """Register a component for shutdown."""
        self._coordinator.register_phase(name, coroutine, timeout, critical)

    async def wait_for_shutdown(self) -> int:
        """Wait for shutdown signal and execute teardown.

        Polls the flag in bounded slices instead of one unbounded blocking wait: a single
        ``run_in_executor(None, event.wait)`` parks a pool thread forever, and if this
        coroutine is cancelled that thread can never be interrupted — it leaks and, being
        non-daemon, blocks interpreter exit. A timed wait releases the worker each slice so
        cancellation can take effect between slices. ``threading.Event`` is retained (it is
        thread-safe for signal handlers); only the waiting strategy changes.
        """
        loop = asyncio.get_running_loop()
        while not self._coordinator._shutdown_event.is_set():
            await loop.run_in_executor(None, self._coordinator._shutdown_event.wait, 0.5)
        return await self._coordinator.execute_shutdown()

    def request_shutdown(self, reason: str = "manual", exit_code: int = EXIT_CLEAN) -> None:
        """Manually request shutdown."""
        self._coordinator.request_shutdown(reason, exit_code)


# ── Utility Functions ────────────────────────────────────────────────────────


async def flush_all_logs(_timeout: float = FLUSH_TIMEOUT_SECONDS) -> None:
    """Flush all logging handlers.

    ``_timeout`` is accepted for forward-compatibility with a bounded flush; it is
    currently advisory because ``logging`` handler flushes are synchronous.
    """
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "flush"):
            with contextlib.suppress(Exception):
                handler.flush()

    # Wait a bit for background threads, but don't let cancellation propagate
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(0.1)


async def close_client_sessions() -> None:
    """Close any open HTTP/client sessions."""
    # This would be populated with actual client references
    pass


def create_shutdown_coordinator(clock: Clock = SYSTEM_CLOCK) -> ShutdownCoordinator:
    """Factory for shutdown coordinator."""
    return ShutdownCoordinator(clock)


# ── Systemd Integration ──────────────────────────────────────────────────────

# sd_notify is a Linux systemd protocol; AF_UNIX sockets do not exist on Windows.
# Probing with find_spec keeps ``sdnotify`` an optional dependency without importing
# it, so the engine boots cleanly (no ModuleNotFoundError) on any platform.
# NOTE: os.name (not sys.platform) — mypy const-folds sys.platform comparisons per
# --platform and would mark the systemd branches unreachable on a Windows check.
_SDNOTIFY_AVAILABLE: bool = (
    os.name == "posix" and importlib.util.find_spec("sdnotify") is not None
)

# AF_UNIX is not available on Windows
_AF_UNIX = getattr(socket, "AF_UNIX", None)


def is_systemd() -> bool:
    """Check if running under systemd."""
    if os.name != "posix":
        return False
    return Path("/run/systemd/system").exists() or "SYSTEMD_EXEC_PID" in os.environ


def notify_systemd_ready() -> None:
    """Notify systemd that service is ready (sd_notify).

    No-op outside systemd (Windows, bare-metal, containers): the sdnotify
    module is only imported inside the guard, so a missing install can never
    raise ModuleNotFoundError on the boot path.
    """
    if not is_systemd():
        return

    if _SDNOTIFY_AVAILABLE:
        import sdnotify

        notifier = sdnotify.SystemdNotifier()
        notifier.notify("READY=1")
        return

    # Manual notification via socket
    if _AF_UNIX is not None:
        notify_socket = os.environ.get("NOTIFY_SOCKET")
        if notify_socket:
            try:
                sock = socket.socket(_AF_UNIX, socket.SOCK_DGRAM)
                sock.sendto(b"READY=1", notify_socket)
            except Exception:
                pass


def notify_systemd_stopping() -> None:
    """Notify systemd that service is stopping."""
    if not is_systemd():
        return

    if _SDNOTIFY_AVAILABLE:
        import sdnotify

        notifier = sdnotify.SystemdNotifier()
        notifier.notify("STOPPING=1")
