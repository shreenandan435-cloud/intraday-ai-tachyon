"""Event loop selection — CLAUDE.md §2.2.

``winloop`` on Windows, ``uvloop`` on POSIX. Both are libuv, both are meaningfully faster than
the stdlib loop, and — the part that is not merely an optimisation — **both implement the
``add_reader`` family of methods that ``zmq.asyncio`` requires.**

The Windows trap
----------------
Python's default event loop on Windows is the *Proactor* loop, which does **not** implement
``add_reader``. ``pyzmq`` needs it to watch a ZeroMQ socket's file descriptor, so an
``AsyncSubscriber`` created on a Proactor loop dies at its first receive with::

    RuntimeError: Proactor event loop does not implement add_reader family of methods
                  required for zmq.

Measured on this machine: the default ``asyncio.run`` fails exactly that way, and
``winloop.run`` delivers every frame. That is a *deployment* constraint, not a performance
preference — a Brain started the naive way would consume no ticks at all.

So this module does three things, in order:

1. Build a libuv loop (``winloop`` / ``uvloop``) when one is available.
2. Fall back to the stdlib **selector** loop — never the Proactor — when it is not.
3. **Verify** the resulting loop before handing it a single coroutine, and refuse to start
   otherwise. Failing at boot with a legible message beats failing at the first tick with a
   traceback from inside pyzmq.

Loop *policies* are deliberately not used: they are deprecated in Python 3.14 and slated for
removal in 3.16. :class:`asyncio.Runner` with an explicit ``loop_factory`` is the supported
mechanism and is scoped to the call rather than to the process.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, Final

from tachyon.core.logger import get_logger

_log = get_logger(__name__)

#: Held as a plain bool rather than testing ``sys.platform`` inline. A direct comparison is
#: narrowed by type checkers to whichever platform they are configured for, which makes the
#: *other* branch statically unreachable and hides real errors in it. Both branches ship.
IS_WINDOWS: Final[bool] = sys.platform == "win32"

#: The accelerator expected on this platform (CLAUDE.md §2.2).
ACCELERATOR: Final[str] = "winloop" if IS_WINDOWS else "uvloop"


class EventLoopUnsuitableError(RuntimeError):
    """The event loop cannot support the ZeroMQ tick spine.

    Raised at startup rather than allowing the process to run and silently receive nothing.
    A Brain that consumes no ticks still looks alive: the watchdog runs, the log ticks over,
    and the feed-staleness rule fires two seconds later blaming the ingestor.
    """


def new_event_loop() -> asyncio.AbstractEventLoop:
    """Create the best available loop for this platform.

    Preference order: libuv accelerator, then the stdlib **selector** loop. The Proactor loop
    is never returned, because ``zmq.asyncio`` cannot work on it.
    """
    if IS_WINDOWS:
        try:
            import winloop
        except ImportError:
            _log.warning(
                "eventloop.accelerator_missing",
                accelerator="winloop",
                fallback="asyncio.SelectorEventLoop",
                impact="slower, and capped at 512 sockets — install winloop (requirements.txt)",
            )
            # Explicitly the selector loop, not asyncio.new_event_loop(), which would hand
            # back the Proactor loop that pyzmq cannot use.
            return asyncio.SelectorEventLoop()
        # winloop/uvloop ship no type information, so their factories are Any at the boundary.
        windows_loop: asyncio.AbstractEventLoop = winloop.new_event_loop()
        return windows_loop

    try:
        import uvloop
    except ImportError:
        _log.warning(
            "eventloop.accelerator_missing",
            accelerator="uvloop",
            fallback="asyncio.new_event_loop",
        )
        return asyncio.new_event_loop()
    posix_loop: asyncio.AbstractEventLoop = uvloop.new_event_loop()
    return posix_loop


def assert_zmq_compatible(loop: asyncio.AbstractEventLoop) -> None:
    """Refuse a loop that cannot host a ``zmq.asyncio`` socket.

    Raises:
        EventLoopUnsuitableError: the loop lacks ``add_reader``/``remove_reader``.
    """
    missing = [name for name in ("add_reader", "remove_reader") if not hasattr(loop, name)]
    if missing:
        raise EventLoopUnsuitableError(
            f"{type(loop).__name__} does not implement {', '.join(missing)}, which "
            f"zmq.asyncio requires to watch a ZeroMQ socket. On Windows this is the default "
            f"Proactor loop — install {ACCELERATOR} (requirements.txt) or start the process "
            f"through tachyon.core.eventloop.run()."
        )


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on a verified, accelerated loop. The only entry point processes should use.

    Equivalent to :func:`asyncio.run`, minus the Windows failure mode. The loop is verified
    *before* the coroutine starts, so an unsuitable loop is a startup error rather than a
    mysterious silence.

    Example::

        if __name__ == "__main__":
            raise SystemExit(eventloop.run(_main()))
    """
    with asyncio.Runner(loop_factory=new_event_loop) as runner:
        loop = runner.get_loop()
        assert_zmq_compatible(loop)
        _log.info("eventloop.started", loop=type(loop).__name__, platform=sys.platform)
        return runner.run(coro)
