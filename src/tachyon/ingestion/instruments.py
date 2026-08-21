"""Angel One instrument master (scrip master) — download, cache, expire, verify.

Why this exists
---------------
``config/settings.yaml`` carries a hand-written ``watchlist`` of ``(symbol, token)`` pairs, and
the token is the **only** identity that reaches the wire: the WebSocket subscribes by token and
:class:`~tachyon.ingestion.service.IngestionService` maps the token back to a symbol. Nothing in
the system ever checks that the pairing is true.

That is a silent, expensive failure mode in both directions:

* A token that no longer exists is *accepted* by the broker. The handshake succeeds, the
  subscription is acknowledged, and no data ever arrives — which surfaces downstream as
  ``FEED_STALE`` with a healthy-looking socket.
* A token that exists but belongs to a *different* instrument is worse. Ticks arrive, the
  indicators compute, the signal fires, and the order is placed against the symbol name we
  believe it is — while the prices came from something else entirely.

So this module fetches Angel One's published master, caches it, expires the cache, and
verifies the configured watchlist against it.

Expiry
------
Stale if **either** the cache is older than :data:`DEFAULT_MAX_AGE` (8 h) **or** it was fetched
on an earlier IST calendar day. The calendar rule is the load-bearing one: the master is
republished daily and a file downloaded at 16:00 yesterday is only 17 h old at 09:00 today but
predates today's contract set. Age alone would pass it during exactly the window that matters.

An unreadable or unparseable cache counts as **stale**, never as fresh — the same fail-safe
posture as the daily lock (CLAUDE.md §1.2). If we cannot prove the cache is current, it is not.

Failure policy
--------------
A download failure is **not** fatal here. This module runs in the ingestor, and refusing to
boot because a CDN is down takes the Brain's entire view of the market with it — including the
prices the 15:15 flatten needs. It reports the failure and lets the caller decide. A *verified
mismatch* is a different fact and callers are expected to treat it as fatal: it means the
configured token does not identify the instrument we think it does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final

import httpx
import msgspec

from tachyon.core.clock import IST, SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, WatchlistItem
from tachyon.core.constants import DATA_DIR
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

#: Angel One's published scrip master. A plain JSON array, republished daily, no auth required.
#: Documented at https://smartapi.angelone.in/docs/Instruments
MASTER_URL: Final[str] = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)

INSTRUMENT_DIR: Final[Path] = DATA_DIR / "instruments"
MASTER_FILENAME: Final[str] = "OpenAPIScripMaster.json"
META_FILENAME: Final[str] = "OpenAPIScripMaster.meta.json"

#: Force a re-download beyond this age even within the same calendar day.
DEFAULT_MAX_AGE: Final[timedelta] = timedelta(hours=8)

#: Download budget. The master is large (~100 MB) and slow; this is generous on purpose.
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 120.0

#: Sanity ceiling. The whole file is decoded in memory, so a pathological response must not be
#: allowed to exhaust the ingestor's address space.
MAX_MASTER_BYTES: Final[int] = 512 * 1024 * 1024

#: Angel One publishes ``tick_size`` in paise ("5.000000" means ₹0.05).
TICK_SIZE_DIVISOR: Final[int] = 100


class Staleness(StrEnum):
    """Why the cache is (or is not) usable. Anything but :attr:`FRESH` forces a download."""

    FRESH = "FRESH"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"
    AGE_EXCEEDED = "AGE_EXCEEDED"
    PREVIOUS_SESSION = "PREVIOUS_SESSION"


class WatchlistIssue(StrEnum):
    """A disagreement between ``settings.yaml`` and the broker's master."""

    #: The configured token is absent from the master. Subscribing to it yields no data.
    TOKEN_NOT_IN_MASTER = "TOKEN_NOT_IN_MASTER"
    #: The token exists but names a different instrument. The dangerous one.
    TOKEN_NAME_MISMATCH = "TOKEN_NAME_MISMATCH"
    #: The symbol has no tradable entry on the configured exchange.
    SYMBOL_NOT_IN_MASTER = "SYMBOL_NOT_IN_MASTER"
    #: Token and name agree, but lot size or tick size does not. Advisory.
    CONTRACT_DRIFT = "CONTRACT_DRIFT"


#: Issues that mean the configured token does not identify the intended instrument.
FATAL_ISSUES: Final[frozenset[WatchlistIssue]] = frozenset(
    {
        WatchlistIssue.TOKEN_NOT_IN_MASTER,
        WatchlistIssue.TOKEN_NAME_MISMATCH,
        WatchlistIssue.SYMBOL_NOT_IN_MASTER,
    }
)


class InstrumentMasterError(RuntimeError):
    """The master could not be downloaded or parsed."""


class _MasterRow(msgspec.Struct, gc=False):
    """One row of the published master. Unknown fields are ignored by msgspec.

    Every field defaults, because a schema change upstream must not turn into a decode error
    for a file we can still partly use. Numeric columns are published inconsistently (sometimes
    quoted, sometimes not), so they are accepted as either and coerced on read.
    """

    token: str = ""
    symbol: str = ""
    name: str = ""
    expiry: str = ""
    lotsize: str | int | float = ""
    instrumenttype: str = ""
    exch_seg: str = ""
    tick_size: str | int | float = ""


_DECODER: Final[msgspec.json.Decoder[list[_MasterRow]]] = msgspec.json.Decoder(list[_MasterRow])


@dataclass(frozen=True, slots=True)
class InstrumentRecord:
    """A resolved instrument, as the broker defines it."""

    token: str
    trading_symbol: str
    name: str
    exchange: str
    instrument_type: str
    expiry: str
    lot_size: int
    tick_size: float

    @classmethod
    def from_row(cls, row: _MasterRow) -> InstrumentRecord:
        return cls(
            token=row.token,
            trading_symbol=row.symbol,
            name=row.name,
            exchange=row.exch_seg,
            instrument_type=row.instrumenttype,
            expiry=row.expiry,
            lot_size=_as_int(row.lotsize),
            tick_size=_as_float(row.tick_size) / TICK_SIZE_DIVISOR,
        )


@dataclass(frozen=True, slots=True)
class WatchlistFinding:
    """One disagreement between the configured watchlist and the master."""

    symbol: str
    issue: WatchlistIssue
    configured_token: str
    detail: str
    master_token: str | None = None

    @property
    def is_fatal(self) -> bool:
        return self.issue in FATAL_ISSUES


@dataclass(frozen=True, slots=True)
class CacheMeta:
    """Sidecar provenance for the cached file.

    Written beside the master rather than inferred from the filesystem mtime: copying,
    restoring or syncing a file rewrites mtime and would make a stale master look current.
    """

    fetched_at: datetime
    trading_date: date
    size_bytes: int
    url: str


class _MetaOnDisk(msgspec.Struct):
    fetched_at: str
    trading_date: str
    size_bytes: int
    url: str


#: Builds the client used for the download. Injected so tests never patch ``httpx`` globally.
ClientFactory = Callable[[], httpx.AsyncClient]


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True)


def _as_int(value: str | int | float) -> int:
    try:
        return int(float(value))
    except TypeError, ValueError:
        return 0


def _as_float(value: str | int | float) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0


class InstrumentMaster:
    """Cached, self-expiring access to Angel One's scrip master.

    Args:
        cache_dir: where the master and its sidecar live.
        url: override for tests.
        max_age: force a refresh beyond this age even on the same calendar day.
        clock: injected for tests; all dates are resolved in IST.
        client_factory: builds the HTTP client. Injected rather than monkeypatched so a test
            can serve a fixture without reaching into the ``httpx`` module namespace.
    """

    __slots__ = ("_cache_dir", "_clock", "_client_factory", "_max_age", "_url")

    def __init__(
        self,
        *,
        cache_dir: Path = INSTRUMENT_DIR,
        url: str = MASTER_URL,
        max_age: timedelta = DEFAULT_MAX_AGE,
        clock: Clock = SYSTEM_CLOCK,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self._cache_dir = cache_dir
        self._url = url
        self._max_age = max_age
        self._clock = clock
        self._client_factory = client_factory if client_factory is not None else _default_client

    @property
    def path(self) -> Path:
        """Where the cached master lives."""
        return self._cache_dir / MASTER_FILENAME

    @property
    def meta_path(self) -> Path:
        return self._cache_dir / META_FILENAME

    # ── expiry ───────────────────────────────────────────────────────────────

    def read_meta(self) -> CacheMeta | None:
        """Provenance of the cached file, or ``None`` if it is missing or unreadable."""
        try:
            raw = self.meta_path.read_bytes()
            on_disk = msgspec.json.decode(raw, type=_MetaOnDisk)
            fetched_at = datetime.fromisoformat(on_disk.fetched_at)
            if fetched_at.tzinfo is None:
                # Naive timestamps are banned (CLAUDE.md §8); a sidecar carrying one was not
                # written by us and cannot be trusted to mean IST.
                return None
            return CacheMeta(
                fetched_at=fetched_at.astimezone(IST),
                trading_date=date.fromisoformat(on_disk.trading_date),
                size_bytes=on_disk.size_bytes,
                url=on_disk.url,
            )
        except (OSError, msgspec.DecodeError, msgspec.ValidationError, ValueError) as exc:
            _log.warning("instruments.meta_unreadable", path=str(self.meta_path), error=str(exc))
            return None

    def staleness(self) -> tuple[Staleness, str]:
        """Classify the cache. Returns the verdict and a human-readable reason.

        Unknown provenance resolves to stale, never to fresh.
        """
        if not self.path.exists():
            return Staleness.MISSING, "no cached master on disk"

        meta = self.read_meta()
        if meta is None:
            return Staleness.UNREADABLE, "cache present but its provenance sidecar is unusable"

        today = now_ist(self._clock).date()
        if meta.trading_date < today:
            return (
                Staleness.PREVIOUS_SESSION,
                f"fetched on {meta.trading_date.isoformat()} (IST), today is {today.isoformat()}",
            )

        age = now_ist(self._clock) - meta.fetched_at
        if age > self._max_age:
            hours = age.total_seconds() / 3600
            return (
                Staleness.AGE_EXCEEDED,
                f"age {hours:.1f}h exceeds the {self._max_age.total_seconds() / 3600:.0f}h limit",
            )

        return Staleness.FRESH, f"fetched {meta.fetched_at.isoformat()}"

    def is_stale(self) -> bool:
        verdict, _ = self.staleness()
        return verdict is not Staleness.FRESH

    # ── fetching ─────────────────────────────────────────────────────────────

    async def ensure_fresh(self, *, force: bool = False) -> Staleness:
        """Download the master if the cache is stale. Returns the verdict acted upon.

        Raises:
            InstrumentMasterError: the download failed *and* no usable cache exists. A failed
                download over a still-usable cache is logged and tolerated.
        """
        verdict, reason = self.staleness()
        if verdict is Staleness.FRESH and not force:
            _log.info("instruments.cache_fresh", reason=reason, path=str(self.path))
            return verdict

        _log.info(
            "instruments.refreshing",
            verdict=str(verdict),
            reason="forced" if force else reason,
            url=self._url,
        )
        try:
            await self.download()
        except InstrumentMasterError as exc:
            if self.path.exists():
                # Better a master we know is stale than none at all: verification against a
                # day-old file still catches a token that was retired last month.
                _log.error(
                    "instruments.refresh_failed_using_stale_cache",
                    error=str(exc),
                    staleness=str(verdict),
                    impact="watchlist verified against an out-of-date master",
                )
                return verdict
            raise
        return verdict

    async def download(self) -> Path:
        """Stream the master to disk and write its sidecar. Atomic — never a half file.

        Raises:
            InstrumentMasterError: on any transport, HTTP or size failure.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".part")
        started = now_ist(self._clock)
        written = 0

        try:
            async with (
                self._client_factory() as client,
                client.stream("GET", self._url) as response,
            ):
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > MAX_MASTER_BYTES:
                            raise InstrumentMasterError(
                                f"instrument master exceeded {MAX_MASTER_BYTES} bytes; "
                                "refusing to buffer it"
                            )
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            raise InstrumentMasterError(f"instrument master download failed: {exc}") from exc
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise InstrumentMasterError(f"could not write {temp_path}: {exc}") from exc
        except InstrumentMasterError:
            temp_path.unlink(missing_ok=True)
            raise

        if written == 0:
            temp_path.unlink(missing_ok=True)
            raise InstrumentMasterError("instrument master download was empty")

        try:
            temp_path.replace(self.path)
            self._write_meta(CacheMeta(started, started.date(), written, self._url))
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise InstrumentMasterError(f"could not install {self.path}: {exc}") from exc

        _log.info(
            "instruments.downloaded",
            path=str(self.path),
            megabytes=round(written / 1_048_576, 1),
            seconds=round((now_ist(self._clock) - started).total_seconds(), 1),
        )
        return self.path

    def _write_meta(self, meta: CacheMeta) -> None:
        payload = msgspec.json.encode(
            _MetaOnDisk(
                fetched_at=meta.fetched_at.isoformat(),
                trading_date=meta.trading_date.isoformat(),
                size_bytes=meta.size_bytes,
                url=meta.url,
            )
        )
        self.meta_path.write_bytes(payload)

    # ── reading ──────────────────────────────────────────────────────────────

    def load(self, *, exchanges: frozenset[str] | None = None) -> tuple[InstrumentRecord, ...]:
        """Decode the cached master, keeping only ``exchanges`` (all of them if ``None``).

        The published file is a single JSON array of ~1.5 M rows, so it is decoded whole and
        then projected. Peak memory is roughly a gigabyte; this runs once at ingestor boot,
        before the socket is open, and the rows are dropped immediately afterwards. Filtering
        to ``NSE`` alone brings the retained set down to a few thousand records.

        Raises:
            InstrumentMasterError: the cache is missing or will not decode.
        """
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise InstrumentMasterError(f"cannot read {self.path}: {exc}") from exc

        try:
            rows = _DECODER.decode(raw)
        except (msgspec.DecodeError, msgspec.ValidationError) as exc:
            raise InstrumentMasterError(f"cannot decode {self.path}: {exc}") from exc

        return tuple(
            InstrumentRecord.from_row(row)
            for row in rows
            if row.token and (exchanges is None or row.exch_seg in exchanges)
        )

    # ── verification ─────────────────────────────────────────────────────────

    def verify_watchlist(self, settings: Settings) -> tuple[WatchlistFinding, ...]:
        """Check every configured ``(symbol, token)`` pair against the master.

        Returns every disagreement found. An empty tuple means the watchlist is provably
        consistent with the broker's own contract list.

        Raises:
            InstrumentMasterError: the master could not be read.
        """
        if not settings.watchlist:
            return ()

        exchanges = frozenset(item.exchange for item in settings.watchlist)
        records = self.load(exchanges=exchanges)
        by_token = {record.token: record for record in records}

        findings: list[WatchlistFinding] = []
        for item in settings.watchlist:
            record = by_token.get(item.token)
            if record is None:
                findings.append(
                    WatchlistFinding(
                        symbol=item.symbol,
                        issue=WatchlistIssue.TOKEN_NOT_IN_MASTER,
                        configured_token=item.token,
                        detail=(
                            f"token {item.token} does not appear in the {item.exchange} master; "
                            "the broker will accept the subscription and send nothing"
                        ),
                        master_token=self._suggest_token(records, item.symbol, item.exchange),
                    )
                )
                continue

            if record.name.upper() != item.symbol.upper():
                findings.append(
                    WatchlistFinding(
                        symbol=item.symbol,
                        issue=WatchlistIssue.TOKEN_NAME_MISMATCH,
                        configured_token=item.token,
                        detail=(
                            f"token {item.token} is {record.name!r} "
                            f"({record.trading_symbol}), not {item.symbol!r}"
                        ),
                        master_token=self._suggest_token(records, item.symbol, item.exchange),
                    )
                )
                continue

            drift = self._contract_drift(item, record)
            if drift is not None:
                findings.append(drift)

        return tuple(findings)

    @staticmethod
    def _suggest_token(
        records: tuple[InstrumentRecord, ...], symbol: str, exchange: str
    ) -> str | None:
        """The token the master says this symbol should have, if it can be identified."""
        wanted = symbol.upper()
        matches = [
            record
            for record in records
            if record.name.upper() == wanted and record.exchange == exchange
        ]
        if not matches:
            return None
        # Cash equity carries the "-EQ" series; prefer it over any derivative sharing the name.
        equity = [record for record in matches if record.trading_symbol.upper().endswith("-EQ")]
        chosen = equity or matches
        return chosen[0].token if len(chosen) == 1 else None

    @staticmethod
    def _contract_drift(item: WatchlistItem, record: InstrumentRecord) -> WatchlistFinding | None:
        """Advisory check on lot and tick size. Never fatal — see ``FATAL_ISSUES``."""
        configured_tick = float(item.tick_size)

        mismatches: list[str] = []
        if record.lot_size and record.lot_size != item.lot_size:
            mismatches.append(f"lot_size {item.lot_size} != master {record.lot_size}")
        # The tick-size column's unit is inferred (paise), so disagreement is reported, not
        # trusted: a wrong assumption here must not manufacture a boot failure.
        if record.tick_size and abs(record.tick_size - configured_tick) > 1e-9:
            mismatches.append(f"tick_size {configured_tick} != master {record.tick_size}")

        if not mismatches:
            return None
        return WatchlistFinding(
            symbol=item.symbol,
            issue=WatchlistIssue.CONTRACT_DRIFT,
            configured_token=item.token,
            detail="; ".join(mismatches),
            master_token=record.token,
        )


def log_findings(findings: tuple[WatchlistFinding, ...]) -> bool:
    """Log every finding at its correct severity. Returns True if any is fatal."""
    fatal = False
    for finding in findings:
        if finding.is_fatal:
            fatal = True
            _log.critical(
                "instruments.watchlist_invalid",
                symbol=finding.symbol,
                issue=str(finding.issue),
                configured_token=finding.configured_token,
                master_token=finding.master_token,
                detail=finding.detail,
            )
        else:
            _log.warning(
                "instruments.watchlist_drift",
                symbol=finding.symbol,
                issue=str(finding.issue),
                configured_token=finding.configured_token,
                detail=finding.detail,
            )
    return fatal
