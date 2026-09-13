"""Binance Vision scraper — tachyon.ingestion.binance_vision_scraper.

Covers the offline conversion path end-to-end with a fake aiohttp session:
date selection, in-memory zip handling, CSV vintage parsing (with and without
header), chunked ZSTD Parquet writing, idempotent skips, and the structured
log events the operator sees. No test touches the network.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from tachyon.ingestion import binance_vision_scraper as bvs
from tachyon.ingestion.binance_vision_scraper import (
    DayStatus,
    _parse_trades_csv,
    build_url,
    download_zip,
    extract_csv_from_zip,
    parquet_path_for,
    process_day,
    recent_days,
    run_scraper,
    write_chunked_parquet,
)

# ─── fixtures / helpers ───────────────────────────────────────────────────────

HEADERLESS_CSV = b"\n".join(
    [
        b"0,100.5,1.0,100.5,1755129600000,True",
        b"1,100.6,2.0,201.2,1755129600123,False",
        b"2,100.7,0.5,50.35,1755129600456,True",
    ]
)

HEADER_CSV = (
    b"id,price,qty,quote_qty,time,is_buyer_maker\n"
    b"0,100.5,1.0,100.5,1755129600000,true\n"
    b"1,100.6,2.0,201.2,1755129600123,false\n"
)


def make_zip(csv_bytes: bytes, name: str = "BTCUSDT-trades-2026-08-14.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv_bytes)
    return buf.getvalue()


class _FakeContent:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def iter_chunked(self, chunk_size: int):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


class _FakeResponse:
    def __init__(self, payload: bytes | None, status: int) -> None:
        self.status = status
        self.content = _FakeContent(payload or b"")

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeSession:
    """Duck-typed stand-in for aiohttp.ClientSession."""

    def __init__(self, payload: bytes | None, status: int = 200) -> None:
        self._payload = payload
        self._status = status
        self.requested: list[str] = []

    def get(self, url: str) -> _FakeResponse:
        self.requested.append(url)
        return _FakeResponse(self._payload, self._status)


class _ExplodingSession:
    def get(self, url: str) -> None:
        raise AssertionError(f"network must not be touched, got {url}")


# ─── date selection ───────────────────────────────────────────────────────────


class TestRecentDays:
    def test_fourteen_days_newest_first(self) -> None:
        days = recent_days(14, end=date(2026, 8, 27))
        assert len(days) == 14
        assert days[0] == date(2026, 8, 27)
        assert days[-1] == date(2026, 8, 14)

    def test_default_end_is_yesterday(self) -> None:
        days = recent_days(1)
        assert days[0] == date.today() - timedelta(days=1)

    def test_non_positive_days_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            recent_days(0)


class TestPaths:
    def test_build_url(self) -> None:
        url = build_url(date(2026, 8, 14), "BTCUSDT")
        assert url == (
            "https://data.binance.vision/data/futures/um/daily/trades/"
            "BTCUSDT/BTCUSDT-trades-2026-08-14.zip"
        )

    def test_parquet_path_is_hive_partitioned(self, tmp_path: Path) -> None:
        p = parquet_path_for(date(2026, 8, 14), "BTCUSDT", tmp_path)
        assert p == tmp_path / "trades" / "date=2026-08-14" / "symbol=BTCUSDT" / "0000.parquet"


# ─── in-memory zip extraction ─────────────────────────────────────────────────


class TestExtractCsv:
    def test_roundtrip(self) -> None:
        assert extract_csv_from_zip(make_zip(HEADERLESS_CSV)) == HEADERLESS_CSV

    def test_no_csv_member(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("README.txt", "no csv here")
        with pytest.raises(ValueError, match="no CSV member"):
            extract_csv_from_zip(buf.getvalue())

    def test_corrupt_archive(self) -> None:
        with pytest.raises(ValueError, match="not a valid ZIP"):
            extract_csv_from_zip(b"this is not a zip")


# ─── CSV parsing (both vintages) ──────────────────────────────────────────────


class TestParseTradesCsv:
    def test_headerless_vintage(self) -> None:
        df = _parse_trades_csv(HEADERLESS_CSV)
        assert list(df.columns) == ["id", "price", "qty", "quote_qty", "time_ms", "is_buyer_maker"]
        assert len(df) == 3
        assert df["id"].dtype == "int64"
        assert df["time_ms"].dtype == "int64"
        assert df["is_buyer_maker"].dtype == bool
        assert df["is_buyer_maker"].tolist() == [True, False, True]
        assert df["time_ms"].iloc[0] == 1755129600000

    def test_header_vintage_lowercase_bool(self) -> None:
        df = _parse_trades_csv(HEADER_CSV)
        assert len(df) == 2
        assert df["is_buyer_maker"].tolist() == [True, False]
        assert df["price"].tolist() == [100.5, 100.6]

    def test_empty_csv_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_trades_csv(b"   \n  ")

    def test_unknown_header_rejected(self) -> None:
        with pytest.raises(ValueError, match="unexpected CSV header"):
            _parse_trades_csv(b"foo,bar\n1,2\n")

    def test_malformed_rows_dropped_and_counted(self) -> None:
        csv = HEADERLESS_CSV + b"\n3,not_a_price,x,y,z,True\n"
        df = _parse_trades_csv(csv)
        assert len(df) == 3
        assert df.attrs["rows_dropped"] == 1


# ─── chunked parquet writing ──────────────────────────────────────────────────


class TestWriteChunkedParquet:
    def test_row_groups_schema_and_zstd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bvs, "ROW_GROUP_ROWS", 2)
        df = _parse_trades_csv(HEADERLESS_CSV)
        out = tmp_path / "date=2026-08-14" / "0000.parquet"

        row_groups = write_chunked_parquet(df, out)

        assert row_groups == 2  # 3 rows at 2 rows/group
        pf = pq.ParquetFile(out)
        assert pf.metadata.num_row_groups == 2
        assert pf.metadata.num_rows == 3
        assert pf.schema_arrow.names == [
            "id", "price", "qty", "quote_qty", "time_ms", "is_buyer_maker",
        ]
        assert pf.metadata.row_group(0).column(0).compression == "ZSTD"
        read = pf.read()
        assert read.column("id").to_pylist() == [0, 1, 2]
        assert read.column("is_buyer_maker").to_pylist() == [True, False, True]

    def test_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        df = _parse_trades_csv(HEADERLESS_CSV)
        out = tmp_path / "0000.parquet"
        write_chunked_parquet(df, out)
        assert out.exists()
        assert not out.with_name(out.name + ".tmp").exists()


# ─── async download + per-day pipeline ────────────────────────────────────────


class TestDownloadZip:
    async def test_success_returns_bytes(self) -> None:
        payload = make_zip(HEADERLESS_CSV)
        session = _FakeSession(payload)
        got = await download_zip(session, "http://x/y.zip", asyncio.Semaphore(2))  # type: ignore[arg-type]
        assert got == payload
        assert session.requested == ["http://x/y.zip"]

    async def test_404_returns_none_without_retry(self) -> None:
        session = _FakeSession(None, status=404)
        got = await download_zip(session, "http://x/y.zip", asyncio.Semaphore(1))  # type: ignore[arg-type]
        assert got is None
        assert len(session.requested) == 1  # 404 is definitive — no retry


class TestProcessDay:
    async def test_skip_existing_without_force(self, tmp_path: Path) -> None:
        day = date(2026, 8, 14)
        out = parquet_path_for(day, "BTCUSDT", tmp_path)
        out.parent.mkdir(parents=True)
        out.write_bytes(b"already here")

        result = await process_day(
            _ExplodingSession(),  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            day,
            "BTCUSDT",
            tmp_path,
            force=False,
        )
        assert result.status is DayStatus.SKIPPED

    async def test_full_pipeline_writes_parquet(self, tmp_path: Path) -> None:
        day = date(2026, 8, 14)
        session = _FakeSession(make_zip(HEADERLESS_CSV))
        result = await process_day(
            session,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            day,
            "BTCUSDT",
            tmp_path,
            force=False,
        )
        assert result.status is DayStatus.WRITTEN
        assert result.rows == 3
        out = parquet_path_for(day, "BTCUSDT", tmp_path)
        assert out.exists()
        assert pq.ParquetFile(out).metadata.num_rows == 3

    async def test_missing_archive_reported(self, tmp_path: Path) -> None:
        session = _FakeSession(None, status=404)
        result = await process_day(
            session,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            date(2026, 8, 14),
            "BTCUSDT",
            tmp_path,
            force=False,
        )
        assert result.status is DayStatus.MISSING

    async def test_corrupt_zip_reported_as_failed(self, tmp_path: Path) -> None:
        session = _FakeSession(b"not a zip at all")
        result = await process_day(
            session,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            date(2026, 8, 14),
            "BTCUSDT",
            tmp_path,
            force=False,
        )
        assert result.status is DayStatus.FAILED


class TestRunScraper:
    async def test_idempotent_second_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second harvest over the same window must skip, not re-download."""
        requests: list[str] = []

        class _SessionCtx(_FakeSession):
            async def __aenter__(self) -> _SessionCtx:
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        def fake_session(*args: Any, **kwargs: Any) -> _SessionCtx:
            ctx = _SessionCtx(make_zip(HEADERLESS_CSV))
            ctx.requested = requests  # shared request log across both runs
            return ctx

        monkeypatch.setattr(bvs.aiohttp, "ClientSession", fake_session)

        end = date(2026, 8, 14)
        first = await run_scraper(days=2, symbol="BTCUSDT", output_dir=tmp_path,
                                  concurrency=2, force=False, end=end)
        assert [r.status for r in first] == [DayStatus.WRITTEN, DayStatus.WRITTEN]
        assert len(requests) == 2

        second = await run_scraper(days=2, symbol="BTCUSDT", output_dir=tmp_path,
                                   concurrency=2, force=False, end=end)
        assert [r.status for r in second] == [DayStatus.SKIPPED, DayStatus.SKIPPED]
        # no new downloads on the idempotent run
        assert len(requests) == 2
