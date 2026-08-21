"""Process A entrypoint — the market data ingestion daemon (CLAUDE.md §2).

Run::

    .venv\\Scripts\\python.exe scripts/run_ingestor.py

Requires ``SMARTAPI_API_KEY``, ``SMARTAPI_CLIENT_CODE``, ``SMARTAPI_PASSWORD`` and
``SMARTAPI_TOTP_SECRET`` in ``.env``. The feed token is obtained by logging in through
:class:`~tachyon.execution.api.SmartApiClient`; ``SMARTAPI_FEED_TOKEN`` remains as a manual
fallback for a machine that cannot reach the REST endpoint, but a pasted token expires and
will start failing the handshake.

This process publishes market data and nothing else. It holds no positions, places no orders
and keeps running when the Brain is down.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.core import eventloop  # noqa: E402
from tachyon.core.config import Settings, get_settings  # noqa: E402
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging  # noqa: E402
from tachyon.execution.api import SmartApiClient, SmartApiError  # noqa: E402
from tachyon.ingestion.instruments import (  # noqa: E402
    InstrumentMaster,
    InstrumentMasterError,
    log_findings,
)
from tachyon.ingestion.service import IngestionService  # noqa: E402
from tachyon.ingestion.ws_client import FeedCredentials  # noqa: E402


async def _resolve_feed_credentials(settings: Settings) -> tuple[str, str]:
    """Log in for a fresh ``(feed_token, jwt_token)``, falling back to the one in ``.env``.

    The login path is preferred because a token pasted into ``.env`` expires silently: the
    handshake starts failing mid-session and the only symptom is a feed that will not come
    back. A login failure is logged and degraded to the fallback rather than being fatal —
    this process publishes market data, and refusing to start would take the Brain's view of
    the market with it.

    The fallback yields no JWT, so the handshake sends the feed token as its bearer. That is
    the degraded path, not the intended one.
    """
    log = get_logger("run_ingestor")
    fallback = settings.smartapi_feed_token.get_secret_value().strip()

    if not settings.smartapi_password.get_secret_value():
        return fallback, ""

    client = SmartApiClient(settings=settings)
    try:
        session = await client.login()
    except SmartApiError as exc:  # SmartApiAuthError subclasses this
        log.error(
            "ingestor.login_failed",
            error=str(exc),
            action="falling back to SMARTAPI_FEED_TOKEN from .env" if fallback else "no fallback",
        )
        return fallback, ""
    finally:
        await client.aclose()
    return session.feed_token, session.jwt_token


async def _verify_instruments(settings: Settings) -> bool:
    """Refresh the instrument master and check the watchlist against it.

    Returns False if the configured tokens provably do not identify the configured symbols —
    a configuration fault the supervisor must not retry (CLAUDE.md §9.1). A master that cannot
    be fetched or read is *not* such a proof: it degrades to a warning, because an unreachable
    CDN must not be able to stop the session.
    """
    log = get_logger("run_ingestor")
    master = InstrumentMaster()
    try:
        await master.ensure_fresh()
        findings = master.verify_watchlist(settings)
    except InstrumentMasterError as exc:
        log.error(
            "ingestor.instrument_master_unavailable",
            error=str(exc),
            impact="watchlist tokens are UNVERIFIED; a retired token subscribes silently and "
            "delivers no ticks",
        )
        return True

    if not findings:
        log.info("ingestor.watchlist_verified", symbols=len(settings.watchlist))
        return True
    return not log_findings(findings)


def _install_signal_handlers(service: IngestionService) -> None:
    """Request a graceful shutdown on SIGINT/SIGTERM (CLAUDE.md §9).

    ``loop.add_signal_handler`` is POSIX-only, so fall back to ``signal.signal`` on Windows.
    """
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        loop.create_task(service.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError, AttributeError:
            signal.signal(sig, lambda _s, _f: request_stop())


async def _main() -> int:
    settings = get_settings()
    configure_logging(
        role="ingestor",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )
    log = get_logger("run_ingestor")

    if not settings.watchlist:
        log.critical("ingestor.empty_watchlist", hint="add symbols to config/settings.yaml")
        return 2

    # Before the socket opens: prove the tokens we are about to subscribe to are real. A
    # retired token is accepted by the broker and then silently starves the feed.
    if not await _verify_instruments(settings):
        log.critical(
            "ingestor.watchlist_rejected",
            hint="fix the token values in config/settings.yaml against the logged master_token",
        )
        return 2

    feed_token, jwt_token = await _resolve_feed_credentials(settings)
    credentials = FeedCredentials.from_settings(settings, feed_token, jwt_token)
    if not credentials.has_bearer():
        log.warning(
            "ingestor.no_jwt_bearer",
            impact="handshake will send the feed token as its Authorization bearer",
            hint="this is the SMARTAPI_FEED_TOKEN fallback path; prefer a real login",
        )
    if not credentials.is_complete():
        log.critical(
            "ingestor.credentials_missing",
            hint="set SMARTAPI_API_KEY, SMARTAPI_CLIENT_CODE, SMARTAPI_PASSWORD and "
            "SMARTAPI_TOTP_SECRET in .env (or SMARTAPI_FEED_TOKEN as a manual fallback)",
        )
        return 2

    service = IngestionService(credentials, settings=settings)
    _install_signal_handlers(service)

    try:
        await service.run()
    except asyncio.CancelledError:
        log.info("ingestor.cancelled")
    except ConnectionError as exc:
        log.critical("ingestor.feed_gave_up", error=str(exc))
        return 1
    finally:
        shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        # eventloop.run, not asyncio.run: on Windows the default Proactor loop cannot host a
        # zmq.asyncio socket at all (see tachyon.core.eventloop).
        raise SystemExit(eventloop.run(_main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
