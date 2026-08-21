"""Structured logging — CLAUDE.md §8.

Design constraints, each of which shows up in the implementation below:

* **Deterministic JSON on disk.** Post-mortems on an async/ZMQ system are impossible without
  machine-parseable logs. ``logs/*.jsonl`` is one JSON object per line, keys sorted.
* **Readable on a console.** A human watching a live session gets colourised key-value output.
  The choice is automatic: a TTY gets pretty, a pipe or service manager gets JSON.
* **IST on every event.** Plus a monotonic reading — when an NTP correction moves the wall
  clock, ``ts_mono`` is what still orders events correctly across processes.
* **Never block the tick path.** Disk writes go through a
  :class:`~logging.handlers.QueueListener` on a background thread; the trading threads only
  ever append to an in-memory queue.
* **Secrets never reach a log line.** A redaction processor runs before any renderer.

Call :func:`configure_logging` exactly once per process, at boot, before any logger is bound.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
import time as _time
from pathlib import Path
from typing import Any, Final

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from tachyon.core.clock import now_ist
from tachyon.core.constants import LOG_DIR, TradingMode

#: Redaction targets. Exact key names plus the suffix rules in :func:`redact_secrets`.
#:
#: Note the deliberate omission of a bare ``token``: SmartAPI *instrument* tokens are not
#: secret and are essential for debugging the feed. Only auth-bearing token names are listed.
_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "authorization",
        "feed_token",
        "jwt",
        "jwt_token",
        "pin",
        "refresh_token",
        "totp",
        "session_token",
        # Anyone holding a Telegram bot token can post as the bot. Listed explicitly rather
        # than caught by a suffix rule, because `_SECRET_SUFFIXES` deliberately does not
        # match a bare `_token` — SmartAPI *instrument* tokens are not secret.
        "telegram_bot_token",
    }
)

_SECRET_SUFFIXES: Final[tuple[str, ...]] = ("password", "_secret", "api_key", "apikey", "_pwd")

_REDACTED: Final[str] = "«redacted»"

_listener: logging.handlers.QueueListener | None = None
_configured: bool = False
_atexit_registered: bool = False


class _StructlogQueueHandler(logging.handlers.QueueHandler):
    """Queue handler that keeps ``record.msg`` as structlog's event dict.

    The stdlib :meth:`~logging.handlers.QueueHandler.prepare` formats the record and replaces
    ``record.msg`` with a plain string, so that the payload survives pickling to another
    *process*. That destroys the event dict that
    :class:`~structlog.stdlib.ProcessorFormatter` needs downstream, and the file sink then
    dies with ``'str' object has no attribute 'copy'``.

    Our queue is an in-process :class:`queue.Queue`, so nothing is pickled and the record can
    be passed through untouched. This is safe against cross-thread mutation because
    ``ProcessorFormatter.format`` copies the record and the event dict before rendering.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


# ──────────────────────────────────────────────────────────────────────────────
# Processors
# ──────────────────────────────────────────────────────────────────────────────


def add_ist_timestamp(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Stamp every event with IST wall time and a monotonic reading.

    ``ts_ist``   ISO-8601 with offset, millisecond precision — for humans and for grep.
    ``ts_epoch`` Unix seconds — for time-series joins.
    ``ts_mono``  monotonic seconds — the only ordering that survives a clock correction,
                 which is exactly the situation where the logs matter most.
    """
    moment = now_ist()
    event_dict["ts_ist"] = moment.isoformat(timespec="milliseconds")
    event_dict["ts_epoch"] = moment.timestamp()
    event_dict["ts_mono"] = _time.monotonic()
    return event_dict


def _is_secret(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES)


def scrub_secrets(value: object) -> object:
    """Recursively replace credential-bearing values in an arbitrary structure.

    Exposed separately from the log processor below because the order journal
    (``tachyon.execution.journal``) writes broker request/response payloads to disk and must
    apply exactly these rules. Two independent redaction lists would drift, and the one that
    drifted would leak a JWT into a file that outlives the session.
    """
    if isinstance(value, dict):
        return {
            key: (_REDACTED if _is_secret(str(key)) else scrub_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(scrub_secrets(item) for item in value)
    return value


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace credential-bearing values, at any nesting depth, before rendering.

    A credential in a log file is a breach that outlives the session, so this runs
    unconditionally — there is no debug flag that disables it.
    """
    for key in list(event_dict):
        if _is_secret(str(key)):
            event_dict[key] = _REDACTED
        else:
            event_dict[key] = scrub_secrets(event_dict[key])
    return event_dict


def add_process_role(role: str) -> Processor:
    """Bind a static ``role`` (``ingestor`` / ``brain`` / ``ui``) onto every event.

    With three processes interleaving into one log directory, this is what makes a merged
    timeline readable.
    """

    def processor(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("role", role)
        return event_dict

    return processor


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


def _shared_processors(role: str) -> list[Processor]:
    """Processor chain applied to structlog and stdlib records alike."""
    return [
        structlog.contextvars.merge_contextvars,
        add_process_role(role),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_ist_timestamp,
        redact_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _console_renderer(pretty: bool) -> Processor:
    if pretty:
        return structlog.dev.ConsoleRenderer(colors=True, pad_event=28)
    return structlog.processors.JSONRenderer(sort_keys=True)


def configure_logging(
    *,
    role: str = "app",
    level: str = "INFO",
    log_dir: Path = LOG_DIR,
    json_lines: bool = True,
    pretty_console: bool | None = None,
    mode: TradingMode = TradingMode.PAPER,
    session_date: str | None = None,
) -> None:
    """Configure structlog + stdlib logging for this process. Idempotent.

    Args:
        role: process identity stamped on every event — ``ingestor``, ``brain``, ``ui``.
        level: minimum level for both sinks.
        log_dir: directory for ``tachyon_<role>_<date>.jsonl``. Created if absent.
        json_lines: write the JSON file sink at all. ``False`` gives console-only, for tests.
        pretty_console: force console style. ``None`` auto-detects — TTY gets colour,
            a redirected stream gets JSON so log shippers see structured data either way.
        mode: recorded on every event so a PAPER line can never be mistaken for a LIVE one
            during a post-mortem.
        session_date: override the filename date. Defaults to today in IST.
    """
    global _listener, _configured, _atexit_registered

    if _configured:
        return

    resolved_pretty = sys.stderr.isatty() if pretty_console is None else pretty_console
    shared = _shared_processors(role)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _console_renderer(resolved_pretty),
            ],
        )
    )
    root.addHandler(console_handler)

    if json_lines:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = session_date if session_date is not None else now_ist().date().isoformat()
        file_handler = logging.FileHandler(
            log_dir / f"tachyon_{role}_{stamp}.jsonl", encoding="utf-8"
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(sort_keys=True),
                ],
            )
        )

        # The tick path must never wait on a disk write (CLAUDE.md §8). Emitting threads put
        # records on an unbounded in-memory queue; a single background thread drains it.
        record_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        root.addHandler(_StructlogQueueHandler(record_queue))
        _listener = logging.handlers.QueueListener(
            record_queue, file_handler, respect_handler_level=True
        )
        _listener.start()

        if not _atexit_registered:
            atexit.register(shutdown_logging)
            _atexit_registered = True

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    structlog.contextvars.bind_contextvars(mode=mode.value)
    _configured = True


def shutdown_logging() -> None:
    """Drain and stop the background writer. Safe to call more than once."""
    global _listener, _configured
    if _listener is not None:
        _listener.stop()
        _listener = None
    logging.shutdown()
    _configured = False


def reset_logging() -> None:
    """Tear down all configuration. Test-support only."""
    shutdown_logging()
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    logging.getLogger().handlers.clear()


def get_logger(name: str, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Use ``__name__`` as ``name``.

    Safe to call before :func:`configure_logging`; structlog binds lazily.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger.bind(**initial) if initial else logger
