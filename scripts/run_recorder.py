"""Track 1 entrypoint — the ZeroMQ sidecar that harvests the order book.

Run::

    .venv\\Scripts\\python.exe scripts/run_recorder.py

A passive SUB on the tick spine. It places no orders, holds no positions, reads no credentials
and publishes nothing. Its entire job is to turn a live session into
``data/ticks/{depth,tick}/date=<date>/symbol=<symbol>/<HHMM>.parquet`` for the Phase 2 offline
training pipeline.

**It cannot slow the trading path.** The ingestor's PUB socket sends with ``zmq.NOBLOCK``
against ``SNDHWM=10_000`` and drops on ``zmq.Again``, so a recorder that stops draining starves
only itself. This is a separate OS process on top of that, and inside it the socket loop and
the disk are separated by a bounded queue. There is no code path from here back to the feed.

Start it *before* the ingestor when you can. ZeroMQ's slow-joiner window means a SUB that
connects after the PUB has bound misses whatever was published in between; subscribing first
costs nothing and closes the hole.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.core.clock import MonotonicDeadline  # noqa: E402
from tachyon.core.config import get_settings  # noqa: E402
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging  # noqa: E402
from tachyon.ipc.schemas import (  # noqa: E402
    TOPIC_DEPTH,
    TOPIC_HEARTBEAT,
    TOPIC_TICK,
    OrderBook,
    Tick,
    symbol_from_topic,
)
from tachyon.ipc.subscriber import (  # noqa: E402
    SchemaVersionMismatchError,
    Subscriber,
    SubscriberRole,
)
from tachyon.persistence.tick_recorder import TickRecorder  # noqa: E402

#: How long a ``recv`` waits before the loop re-checks the stop flag. Short enough that Ctrl+C
#: and SIGTERM land promptly, long enough not to spin a core on a quiet market.
RECV_TIMEOUT_MS: Final[int] = 500

#: Cadence of the ``recorder.stats`` heartbeat line, in seconds.
STATS_INTERVAL_SECONDS: Final[float] = 60.0

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_CONFIG_FAULT: Final[int] = 2


def _install_signal_handlers(stopping: threading.Event) -> None:
    """Request a graceful shutdown on SIGINT/SIGTERM.

    Graceful matters more here than it looks: a Parquet footer is written at close, and a file
    whose writer never closed cannot be read back at all.
    """

    def request_stop(*_args: object) -> None:
        stopping.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)


def _stop_time(value: str) -> dtime:
    """Parse ``--stop-at HH:MM`` into an IST wall-clock time."""
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected HH:MM in IST, got {value!r}") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive L2 order book recorder (Track 1).")
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Dataset root. Defaults to data/ticks. Point it elsewhere to smoke-test the "
            "sidecar without writing synthetic rows into the real training corpus, which "
            "nothing downstream could tell apart from harvested ones."
        ),
    )
    parser.add_argument(
        "--stop-at",
        type=_stop_time,
        default=None,
        metavar="HH:MM",
        help=(
            "Shut down gracefully at this IST time and write every Parquet footer. Set this "
            "for unattended runs: Windows cannot deliver a Ctrl+C to a headless child, so the "
            "alternative is taskkill, which loses the open rotation window."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(
        role="recorder",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )
    log = get_logger("run_recorder")

    recorder_settings = settings.recorder
    if not recorder_settings.enabled:
        log.warning("recorder.disabled", hint="set recorder.enabled in config/settings.yaml")
        shutdown_logging()
        return EXIT_OK
    if not recorder_settings.streams:
        log.critical("recorder.no_streams", hint="recorder.streams must list depth and/or tick")
        shutdown_logging()
        return EXIT_CONFIG_FAULT

    stopping = threading.Event()
    _install_signal_handlers(stopping)

    recorder = TickRecorder(
        directory=args.directory,
        streams=tuple(recorder_settings.streams),
        queue_size=recorder_settings.queue_size,
        row_group_size=recorder_settings.row_group_size,
        flush_interval_seconds=recorder_settings.flush_interval_seconds,
        rotate_minutes=recorder_settings.rotate_minutes,
        compression=recorder_settings.compression,
    )

    # FEED.HEARTBEAT is subscribed but never recorded. It is the only frame carrying
    # SCHEMA_VERSION, and decode_frames raises SchemaVersionMismatchError on the first
    # mismatched one. That exception is left fatal on purpose: writing a session's worth of
    # books decoded under the wrong schema is strictly worse than recording nothing.
    topics = (TOPIC_DEPTH, TOPIC_TICK, TOPIC_HEARTBEAT)
    subscriber = Subscriber(
        topics,
        role=SubscriberRole.TELEMETRY,
        endpoint=settings.zmq_tick_endpoint,
        conflate=False,  # TELEMETRY *may* conflate; a recorder that samples is worthless
        settings=settings,
    )
    # Pinned to the monotonic clock at arm-time, so an NTP correction mid-session cannot move
    # the shutdown — the same rule the square-off watchdog runs under.
    deadline = MonotonicDeadline.arm("recorder-stop", args.stop_at) if args.stop_at else None

    log.info(
        "recorder.subscribed",
        endpoint=subscriber.endpoint,
        topics=list(topics),
        streams=sorted(recorder_settings.streams),
        stop_at=deadline.target_wall.isoformat() if deadline else None,
    )

    exit_code = EXIT_OK
    started_mono = time.monotonic()
    next_stats = started_mono + STATS_INTERVAL_SECONDS
    try:
        while not stopping.is_set():
            if deadline is not None and deadline.has_elapsed():
                log.info("recorder.stop_time_reached", stop_at=deadline.target_wall.isoformat())
                break
            envelope = subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
            if envelope is not None:
                message = envelope.message
                if isinstance(message, OrderBook):
                    recorder.record_depth(symbol_from_topic(envelope.topic), message)
                elif isinstance(message, Tick):
                    recorder.record_tick(symbol_from_topic(envelope.topic), message)

            now_mono = time.monotonic()
            if now_mono >= next_stats:
                next_stats = now_mono + STATS_INTERVAL_SECONDS
                log.info("recorder.stats", received=subscriber.received, **recorder.stats.as_dict())
    except SchemaVersionMismatchError as exc:
        log.critical(
            "recorder.schema_mismatch",
            error=str(exc),
            impact="stopped recording rather than write books decoded under the wrong schema",
        )
        exit_code = EXIT_CONFIG_FAULT
    except KeyboardInterrupt:
        log.info("recorder.interrupted")
    except Exception as exc:  # noqa: BLE001 - log what killed it before the footers are written
        log.critical(
            "recorder.crashed",
            error=str(exc),
            error_type=type(exc).__name__,
            impact="recording stopped; trading is unaffected",
        )
        exit_code = EXIT_ERROR
    finally:
        # Order matters: stop reading, then close the recorder so every footer is written.
        subscriber.close()
        recorder.close()
        log.info(
            "recorder.session_complete",
            received=subscriber.received,
            decode_errors=subscriber.decode_errors,
            **recorder.stats.as_dict(),
        )
        shutdown_logging()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
