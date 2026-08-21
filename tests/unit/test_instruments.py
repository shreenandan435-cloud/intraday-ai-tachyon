"""Instrument master cache, expiry and watchlist verification — tachyon.ingestion.instruments.

The expiry rules carry the weight here. A cache that wrongly reports itself fresh means the
watchlist is verified against yesterday's contract set, which is the failure this module was
written to prevent.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import msgspec
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import Settings, WatchlistItem
from tachyon.ingestion import instruments as mod
from tachyon.ingestion.instruments import (
    InstrumentMaster,
    InstrumentMasterError,
    Staleness,
    WatchlistIssue,
    log_findings,
)


def _at(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def _master_rows() -> list[dict[str, Any]]:
    return [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5.000000",
        },
        {
            "token": "1333",
            "symbol": "HDFCBANK-EQ",
            "name": "HDFCBANK",
            "expiry": "",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5.000000",
        },
        {
            "token": "99999",
            "symbol": "SOMETHINGELSE-EQ",
            "name": "SOMETHINGELSE",
            "expiry": "",
            "lotsize": 1,
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": 5,
        },
        {
            "token": "54321",
            "symbol": "NIFTY28AUG25FUT",
            "name": "NIFTY",
            "expiry": "28AUG2025",
            "lotsize": "75",
            "instrumenttype": "FUTIDX",
            "exch_seg": "NFO",
            "tick_size": "5.000000",
        },
    ]


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "instruments"


def _write_cache(cache_dir: Path, *, fetched_at: datetime, rows: list[dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows).encode()
    (cache_dir / mod.MASTER_FILENAME).write_bytes(payload)
    (cache_dir / mod.META_FILENAME).write_bytes(
        msgspec.json.encode(
            {
                "fetched_at": fetched_at.isoformat(),
                "trading_date": fetched_at.date().isoformat(),
                "size_bytes": len(payload),
                "url": mod.MASTER_URL,
            }
        )
    )


def _master(cache_dir: Path, now: datetime, *, responder: Any = None) -> InstrumentMaster:
    factory = None if responder is None else (lambda: httpx.AsyncClient(transport=responder))
    return InstrumentMaster(cache_dir=cache_dir, clock=ManualClock(now), client_factory=factory)


def _settings(**overrides: Any) -> Settings:
    watchlist = overrides.pop(
        "watchlist",
        (
            WatchlistItem(symbol="RELIANCE", token="2885"),
            WatchlistItem(symbol="HDFCBANK", token="1333"),
        ),
    )
    return Settings(watchlist=watchlist, **overrides)


# ── expiry ───────────────────────────────────────────────────────────────────


class TestStaleness:
    def test_missing_cache_is_stale(self, cache_dir: Path) -> None:
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 9)).staleness()
        assert verdict is Staleness.MISSING

    def test_fresh_within_eight_hours_same_day(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 14)).staleness()
        assert verdict is Staleness.FRESH

    def test_age_beyond_eight_hours_expires(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 1), rows=_master_rows())
        verdict, reason = _master(cache_dir, _at(2026, 8, 11, 10)).staleness()
        assert verdict is Staleness.AGE_EXCEEDED
        assert "9.0h" in reason

    def test_exactly_eight_hours_is_still_fresh(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 1), rows=_master_rows())
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 9)).staleness()
        assert verdict is Staleness.FRESH

    def test_previous_calendar_day_expires_even_when_young(self, cache_dir: Path) -> None:
        """The rule that matters: 17:00 yesterday is 16 h old at 09:00, under the age limit,
        but predates today's republished master."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 10, 17), rows=_master_rows())
        verdict, reason = _master(cache_dir, _at(2026, 8, 11, 9)).staleness()
        assert verdict is Staleness.PREVIOUS_SESSION
        assert "2026-08-10" in reason

    def test_calendar_day_is_evaluated_in_ist_not_utc(self, cache_dir: Path) -> None:
        """23:00 IST and 01:00 IST are different trading dates despite being 2 h apart —
        and the same instants share a UTC date, so a UTC comparison would call this fresh."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 10, 23), rows=_master_rows())
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 1)).staleness()
        assert verdict is Staleness.PREVIOUS_SESSION

    def test_missing_sidecar_is_stale_not_fresh(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        (cache_dir / mod.META_FILENAME).unlink()
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 9)).staleness()
        assert verdict is Staleness.UNREADABLE

    def test_corrupt_sidecar_is_stale_not_fresh(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        (cache_dir / mod.META_FILENAME).write_bytes(b"{not json")
        assert _master(cache_dir, _at(2026, 8, 11, 9)).is_stale()

    def test_naive_timestamp_in_sidecar_is_rejected(self, cache_dir: Path) -> None:
        """Naive datetimes are banned (CLAUDE.md §8) — one here cannot be assumed to mean IST."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        (cache_dir / mod.META_FILENAME).write_bytes(
            msgspec.json.encode(
                {
                    "fetched_at": "2026-08-11T08:00:00",
                    "trading_date": "2026-08-11",
                    "size_bytes": 10,
                    "url": mod.MASTER_URL,
                }
            )
        )
        verdict, _ = _master(cache_dir, _at(2026, 8, 11, 9)).staleness()
        assert verdict is Staleness.UNREADABLE

    def test_rejects_non_positive_max_age(self, cache_dir: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            InstrumentMaster(cache_dir=cache_dir, max_age=timedelta(0))


# ── downloading ──────────────────────────────────────────────────────────────


class TestDownload:
    @pytest.mark.asyncio
    async def test_download_writes_master_and_sidecar(self, cache_dir: Path) -> None:
        body = json.dumps(_master_rows()).encode()
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_serving(200, body))

        await master.download()

        assert master.path.read_bytes() == body
        meta = master.read_meta()
        assert meta is not None
        assert meta.trading_date.isoformat() == "2026-08-11"
        assert meta.size_bytes == len(body)
        assert not master.is_stale()

    @pytest.mark.asyncio
    async def test_failed_download_leaves_no_partial_file(self, cache_dir: Path) -> None:
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_serving(503, b"nope"))

        with pytest.raises(InstrumentMasterError):
            await master.download()

        assert not master.path.exists()
        assert not master.path.with_suffix(".part").exists()

    @pytest.mark.asyncio
    async def test_empty_body_is_rejected(self, cache_dir: Path) -> None:
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_serving(200, b""))
        with pytest.raises(InstrumentMasterError, match="empty"):
            await master.download()

    @pytest.mark.asyncio
    async def test_transport_error_is_wrapped(self, cache_dir: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=httpx.MockTransport(handler))
        with pytest.raises(InstrumentMasterError, match="download failed"):
            await master.download()

    @pytest.mark.asyncio
    async def test_ensure_fresh_skips_download_when_fresh(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        seen: list[httpx.Request] = []
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_recording(seen))

        verdict = await master.ensure_fresh()

        assert verdict is Staleness.FRESH
        assert seen == []

    @pytest.mark.asyncio
    async def test_ensure_fresh_downloads_when_expired(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 10, 17), rows=_master_rows())
        seen: list[httpx.Request] = []
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_recording(seen))

        verdict = await master.ensure_fresh()

        assert verdict is Staleness.PREVIOUS_SESSION
        assert len(seen) == 1
        assert str(seen[0].url) == mod.MASTER_URL

    @pytest.mark.asyncio
    async def test_force_downloads_even_when_fresh(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        seen: list[httpx.Request] = []
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_recording(seen))

        await master.ensure_fresh(force=True)

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_download_failure_over_usable_cache_degrades(self, cache_dir: Path) -> None:
        """An unreachable CDN must not stop the session — a stale master still catches a
        token retired last month."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 10, 17), rows=_master_rows())
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_serving(503, b""))

        verdict = await master.ensure_fresh()

        assert verdict is Staleness.PREVIOUS_SESSION
        assert master.path.exists()

    @pytest.mark.asyncio
    async def test_download_failure_with_no_cache_raises(self, cache_dir: Path) -> None:
        master = _master(cache_dir, _at(2026, 8, 11, 9), responder=_serving(503, b""))
        with pytest.raises(InstrumentMasterError):
            await master.ensure_fresh()


# ── verification ─────────────────────────────────────────────────────────────


class TestVerifyWatchlist:
    def test_matching_watchlist_produces_no_findings(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(_settings())
        assert findings == ()

    def test_retired_token_is_fatal(self, cache_dir: Path) -> None:
        """The exact shape of the reported bug: the broker accepts this subscription and
        never sends a tick."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        settings = _settings(watchlist=(WatchlistItem(symbol="RELIANCE", token="404404"),))

        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(settings)

        assert len(findings) == 1
        assert findings[0].issue is WatchlistIssue.TOKEN_NOT_IN_MASTER
        assert findings[0].is_fatal
        assert findings[0].master_token == "2885"  # the correction is offered

    def test_token_naming_a_different_instrument_is_fatal(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        settings = _settings(watchlist=(WatchlistItem(symbol="RELIANCE", token="99999"),))

        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(settings)

        assert findings[0].issue is WatchlistIssue.TOKEN_NAME_MISMATCH
        assert findings[0].is_fatal
        assert "SOMETHINGELSE" in findings[0].detail

    def test_lot_size_drift_is_advisory_not_fatal(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        settings = _settings(
            watchlist=(WatchlistItem(symbol="NIFTY", token="54321", exchange="NFO", lot_size=50),)
        )

        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(settings)

        assert findings[0].issue is WatchlistIssue.CONTRACT_DRIFT
        assert not findings[0].is_fatal
        assert "75" in findings[0].detail

    def test_tick_size_is_converted_from_paise(self, cache_dir: Path) -> None:
        """Master publishes "5.000000" for a ₹0.05 tick; a missing divisor would report every
        NSE equity as drifted."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        records = _master(cache_dir, _at(2026, 8, 11, 9)).load(exchanges=frozenset({"NSE"}))
        assert records[0].tick_size == pytest.approx(0.05)

    def test_load_filters_by_exchange(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        records = _master(cache_dir, _at(2026, 8, 11, 9)).load(exchanges=frozenset({"NFO"}))
        assert [record.name for record in records] == ["NIFTY"]

    def test_numeric_columns_survive_both_encodings(self, cache_dir: Path) -> None:
        """Angel publishes lotsize/tick_size quoted in some rows and bare in others."""
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        records = _master(cache_dir, _at(2026, 8, 11, 9)).load(exchanges=frozenset({"NSE"}))
        by_name = {record.name: record for record in records}
        assert by_name["SOMETHINGELSE"].lot_size == 1  # was an int
        assert by_name["RELIANCE"].lot_size == 1  # was a string

    def test_empty_watchlist_produces_no_findings(self, cache_dir: Path) -> None:
        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(_settings(watchlist=()))
        assert findings == ()

    def test_undecodable_master_raises(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True)
        (cache_dir / mod.MASTER_FILENAME).write_bytes(b"{not json")
        with pytest.raises(InstrumentMasterError, match="decode"):
            _master(cache_dir, _at(2026, 8, 11, 9)).load()

    def test_missing_master_raises(self, cache_dir: Path) -> None:
        with pytest.raises(InstrumentMasterError, match="cannot read"):
            _master(cache_dir, _at(2026, 8, 11, 9)).load()


class TestLogFindings:
    def test_reports_fatal(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        settings = _settings(watchlist=(WatchlistItem(symbol="RELIANCE", token="404404"),))
        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(settings)
        assert log_findings(findings) is True

    def test_drift_alone_is_not_fatal(self, cache_dir: Path) -> None:
        _write_cache(cache_dir, fetched_at=_at(2026, 8, 11, 8), rows=_master_rows())
        settings = _settings(
            watchlist=(WatchlistItem(symbol="NIFTY", token="54321", exchange="NFO", lot_size=50),)
        )
        findings = _master(cache_dir, _at(2026, 8, 11, 9)).verify_watchlist(settings)
        assert log_findings(findings) is False

    def test_no_findings_is_not_fatal(self) -> None:
        assert log_findings(()) is False


# ── helpers ──────────────────────────────────────────────────────────────────


def _serving(status: int, body: bytes) -> httpx.MockTransport:
    """A transport that answers every request identically."""
    return httpx.MockTransport(lambda _request: httpx.Response(status, content=body))


def _recording(seen: list[httpx.Request]) -> httpx.MockTransport:
    """A transport that serves a valid master and appends each request to ``seen``."""
    body = json.dumps(_master_rows()).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)
