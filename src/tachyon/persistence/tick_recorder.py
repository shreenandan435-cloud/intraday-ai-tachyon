"""Level-2 order book harvesting — the Track 1 half of the Phase 2 pipeline.

The Brain consumes ticks into ring buffers and throws them away, deliberately: nothing on the
tick path may wait on a disk. That leaves the live Indian-market LOB data the Transformer-PPO
agent needs to transfer-learn onto unrecorded. This module is the recorder, and it runs in the
**sidecar process** (``scripts/run_recorder.py``) precisely so that "record everything" and
"never block the tick path" stop being in tension.

Why it cannot touch the trading path
------------------------------------
Three independent layers, none of which rely on this code behaving:

1. It is a separate OS process — its own interpreter, its own GIL, its own fate.
2. ``Publisher.publish_raw`` already sends with ``zmq.NOBLOCK`` against ``SNDHWM=10_000`` and
   drops on ``zmq.Again``. A subscriber that stops draining can only starve itself; there is no
   back-pressure path from here to the feed.
3. Inside the sidecar, the socket thread and the disk are separated by a bounded queue. A
   stalled disk cannot reach the socket read loop until the queue fills, at which point rows are
   dropped and counted rather than waited on.

Divergence from ``trade_logger``
--------------------------------
That module opens, appends, flushes and closes on **every row**. At tens of trades a day that
is the right trade: each row is durable the instant it is recorded. At L2 rates it would be
catastrophic, and Parquet cannot work that way regardless — the format is columnar and its
footer is written at close. So the recorder holds writers open and batches row groups.

That buys throughput and pays for it in crash exposure, which is bounded in two places:
:attr:`RecorderSettings.flush_interval_seconds` caps how long a row sits in memory, and
:attr:`RecorderSettings.rotate_minutes` caps how much a *hard* kill destroys. The second one
matters more than it looks: **an open Parquet file has no footer and is unreadable**, so
without rotation a killed unattended session would lose its entire day, not its last batch.

Layout
------
Hive-partitioned, so the offline loader needs no glob logic and can push a symbol filter down
into the scan::

    data/ticks/depth/date=2026-08-17/symbol=UFLEX/0915.parquet
    data/ticks/tick/date=2026-08-17/symbol=UFLEX/0915.parquet

``symbol`` and ``date`` are carried by the *path* and materialised as columns on read, so they
are not repeated in every row. ``token`` stays in the file so a single file lifted out of the
tree still identifies its instrument.

Failure rules, inherited from ``persistence.journal`` and ``persistence.trade_logger``:

* **A write failure never reaches the caller.** Logged at ``CRITICAL``, counted, dropped.
* **A full queue drops the row rather than blocking.** Blocking is the one thing this design
  exists to make impossible.
"""

from __future__ import annotations

import atexit
import queue
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import DATA_DIR
from tachyon.core.logger import get_logger
from tachyon.ipc.schemas import DEPTH_LEVELS

if TYPE_CHECKING:
    from tachyon.ipc.schemas import OrderBook, Tick

_log = get_logger(__name__)

#: Root of the harvested dataset. ``data/ticks/*`` is already gitignored.
#: Resolved at *construction*, never at import, so tests can redirect this constant.
TICKS_DIR: Final[Path] = DATA_DIR / "ticks"

DEPTH_STREAM: Final[str] = "depth"
TICK_STREAM: Final[str] = "tick"

#: Column order for each stream. Append only — Parquet stores the schema in the file, so a
#: reordering does not corrupt what is already on disk, but it does split the dataset into two
#: incompatible halves that no single scan can read.
DEPTH_COLUMNS: Final[tuple[str, ...]] = (
    "token",
    "ts_epoch",
    "recv_epoch",
    *(f"bid_price_{level}" for level in range(DEPTH_LEVELS)),
    *(f"bid_qty_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_price_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_qty_{level}" for level in range(DEPTH_LEVELS)),
)

TICK_COLUMNS: Final[tuple[str, ...]] = (
    "token",
    "ts_epoch",
    "recv_epoch",
    "ltp",
    "volume",
    "seq",
)

#: ``recv_epoch`` is wall-clock-at-receipt, **not** ``Envelope.received_mono``: a monotonic
#: reading is relative to an arbitrary per-process origin and means nothing once the session is
#: over. ``recv_epoch - ts_epoch`` is exchange-to-us latency, which is a feature worth training
#: on rather than an artefact.
DEPTH_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("token", pa.string()),
        pa.field("ts_epoch", pa.float64()),
        pa.field("recv_epoch", pa.float64()),
        *(pa.field(f"bid_price_{level}", pa.float64()) for level in range(DEPTH_LEVELS)),
        *(pa.field(f"bid_qty_{level}", pa.int64()) for level in range(DEPTH_LEVELS)),
        *(pa.field(f"ask_price_{level}", pa.float64()) for level in range(DEPTH_LEVELS)),
        *(pa.field(f"ask_qty_{level}", pa.int64()) for level in range(DEPTH_LEVELS)),
    ]
)

TICK_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("token", pa.string()),
        pa.field("ts_epoch", pa.float64()),
        pa.field("recv_epoch", pa.float64()),
        pa.field("ltp", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("seq", pa.int64()),
    ]
)

SCHEMAS: Final[dict[str, pa.Schema]] = {
    DEPTH_STREAM: DEPTH_SCHEMA,
    TICK_STREAM: TICK_SCHEMA,
}

#: Deep enough that filling it means the writer is wedged, not that the market was busy: the
#: whole watchlist at full SmartAPI cadence is a few hundred rows a second.
DEFAULT_QUEUE_SIZE: Final[int] = 100_000

DEFAULT_ROW_GROUP_SIZE: Final[int] = 4_096
DEFAULT_FLUSH_INTERVAL_SECONDS: Final[float] = 30.0
DEFAULT_ROTATE_MINUTES: Final[int] = 60
DEFAULT_COMPRESSION: Final[str] = "zstd"

#: How long :meth:`TickRecorder.close` waits for the writer to drain and write its footers.
#: More generous than ``trade_logger``'s five seconds because this drain can involve tens of
#: thousands of buffered rows, and a truncated footer costs the whole file rather than one row.
DRAIN_TIMEOUT_SECONDS: Final[float] = 15.0

#: How often the writer thread wakes when the queue is empty, so the time-based flush still
#: fires during a quiet market instead of waiting for the next message to push it.
_POLL_SECONDS: Final[float] = 0.5

#: ZSTD level 3. Above it the ratio gain on float/int columns is under a percent and the CPU
#: cost is not; the sidecar has a core to itself but no reason to burn it.
_ZSTD_LEVEL: Final[int] = 3

_SENTINEL: Final[object] = object()


@dataclass(frozen=True, slots=True)
class _PendingRow:
    """One row bound for one partition.

    The date and rotation window are resolved when the row is *created*, not when it is
    written — the ``trade_logger`` lesson applied to a finer boundary: a book captured at
    09:59:59.9 must land in the 09:00 file even if the writer thread gets to it at 10:00:00.1.

    ``values`` is positional, in the stream's column order, so the writer appends straight into
    its column buffers without rebuilding a dict per row.
    """

    stream: str
    symbol: str
    on_date: date
    window: str
    values: tuple[Any, ...]


@dataclass(slots=True)
class RecorderStats:
    """Session counters. Asserted on by tests, logged on shutdown for the operator.

    ``rows_dropped`` being non-zero is the signal that the harvest is incomplete — cross-check
    it against the ingestor's ``books_published`` before trusting a session's dataset.
    """

    depth_rows: int = 0
    tick_rows: int = 0
    rows_dropped: int = 0
    write_failures: int = 0
    row_groups_written: int = 0
    files_rolled: int = 0
    seq_gaps: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "depth_rows": self.depth_rows,
            "tick_rows": self.tick_rows,
            "rows_dropped": self.rows_dropped,
            "write_failures": self.write_failures,
            "row_groups_written": self.row_groups_written,
            "files_rolled": self.files_rolled,
            "seq_gaps": self.seq_gaps,
        }


@dataclass(slots=True)
class _Sink:
    """One open Parquet file plus the column buffers feeding it.

    The writer is opened lazily on the first flush rather than on the first row, so a symbol
    that never trades leaves no empty file behind to confuse the offline loader.
    """

    stream: str
    symbol: str
    on_date: date
    window: str
    schema: pa.Schema
    columns: list[list[Any]]
    last_flush_mono: float
    writer: pq.ParquetWriter | None = None
    path: Path | None = None

    @property
    def rows(self) -> int:
        return len(self.columns[0])

    def append(self, values: tuple[Any, ...]) -> None:
        for column, value in zip(self.columns, values, strict=True):
            column.append(value)

    def take_table(self) -> pa.Table:
        """Drain the buffers into one Arrow table, leaving the sink empty.

        Types come from the schema rather than inference: a window in which every quantity
        happened to be zero would otherwise produce a different column type than the window
        beside it, and the two files would no longer scan as one dataset.
        """
        arrays = [
            pa.array(column, type=self.schema.field(index).type)
            for index, column in enumerate(self.columns)
        ]
        for column in self.columns:
            column.clear()
        return pa.Table.from_arrays(arrays, schema=self.schema)


class TickRecorder:
    """Buffers L2 books and trade prints, writes them to partitioned Parquet off-thread.

    Args:
        directory: dataset root. Defaults to :data:`TICKS_DIR` *at construction*, which is what
            lets the test suite redirect the module constant.
        clock: injected so partitioning and rotation are deterministic in tests.
        streams: which spines to persist. A stream absent here is dropped at ``record_*``.
        queue_size: rows buffered between the socket thread and the writer thread.
        row_group_size: rows accumulated before a row group is written.
        flush_interval_seconds: force a partial row group after this long.
        rotate_minutes: how often to close the current file and open the next.
        compression: Parquet codec.
        enabled: ``False`` makes every ``record_*`` call a no-op.
        start: ``False`` leaves the writer thread unstarted — tests drive
            :meth:`drain_for_test` instead.

    The recording methods are safe to call from one thread only (the socket thread); everything
    they touch is either the queue, which is thread-safe, or producer-private state.
    """

    __slots__ = (
        "_clock",
        "_compression",
        "_directory",
        "_enabled",
        "_flush_interval",
        "_last_seq",
        "_queue",
        "_rotate_minutes",
        "_row_group_size",
        "_sinks",
        "_started",
        "_stopping",
        "_streams",
        "_thread",
        "stats",
    )

    def __init__(
        self,
        *,
        directory: Path | None = None,
        clock: Clock = SYSTEM_CLOCK,
        streams: tuple[str, ...] = (DEPTH_STREAM, TICK_STREAM),
        queue_size: int = DEFAULT_QUEUE_SIZE,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        rotate_minutes: int = DEFAULT_ROTATE_MINUTES,
        compression: str = DEFAULT_COMPRESSION,
        enabled: bool = True,
        start: bool = True,
    ) -> None:
        self._directory = directory if directory is not None else TICKS_DIR
        self._clock = clock
        self._streams = frozenset(streams)
        self._row_group_size = row_group_size
        self._flush_interval = flush_interval_seconds
        self._rotate_minutes = rotate_minutes
        self._compression = compression
        self._enabled = enabled

        self._queue: queue.Queue[_PendingRow | object] = queue.Queue(maxsize=queue_size)
        self._sinks: dict[tuple[str, str], _Sink] = {}
        self._last_seq: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = threading.Event()
        self.stats = RecorderStats()

        if start:
            self.start()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the writer thread. Idempotent."""
        if self._started or not self._enabled:
            self._started = True
            return
        self._thread = threading.Thread(target=self._run, name="tick-recorder", daemon=True)
        self._thread.start()
        self._started = True
        atexit.register(self.close)
        _log.info(
            "recorder.started",
            directory=str(self._directory),
            streams=sorted(self._streams),
            rotate_minutes=self._rotate_minutes,
            row_group_size=self._row_group_size,
        )

    def close(self) -> None:
        """Drain, flush and close every open file. Safe to call twice.

        Closing is what writes each Parquet footer, so this is not a courtesy: a file whose
        writer was never closed cannot be read back at all.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        thread = self._thread
        if thread is None:
            # Never started a thread (``start=False``, or disabled). Finalise inline — the
            # writer state is ours alone, so there is nothing to race with.
            self._close_all_sinks()
            return

        self._queue.put(_SENTINEL)
        thread.join(timeout=DRAIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            # Deliberately do NOT finalise from here: the writer thread still owns those
            # handles, and two threads closing one ParquetWriter corrupts the footer we are
            # trying to save.
            _log.critical(
                "recorder.writer_stuck",
                seconds=DRAIN_TIMEOUT_SECONDS,
                queued=self._queue.qsize(),
                impact="buffered rows were not written and open Parquet files have no footer",
            )
        self._thread = None
        _log.info("recorder.stopped", **self.stats.as_dict())

    def __enter__(self) -> TickRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── recording (socket thread) ────────────────────────────────────────────

    def record_depth(self, symbol: str, book: OrderBook) -> None:
        """Queue one L2 snapshot. Never blocks, never raises."""
        if not self._enabled or DEPTH_STREAM not in self._streams:
            return
        try:
            now = now_ist(self._clock)
            values = (
                book.token,
                book.ts_epoch,
                now.timestamp(),
                *book.bid_price,
                *book.bid_qty,
                *book.ask_price,
                *book.ask_qty,
            )
            self._enqueue(DEPTH_STREAM, symbol, now, values)
            self.stats.depth_rows += 1
        except Exception as exc:  # noqa: BLE001 - recording must never break the socket loop
            self.stats.write_failures += 1
            _log.error(
                "recorder.depth_capture_failed",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this book is lost; the feed is unaffected",
            )

    def record_tick(self, symbol: str, tick: Tick) -> None:
        """Queue one trade print, and check its sequence. Never blocks, never raises.

        ``Tick.seq`` is the only message-loss signal on the wire — ``OrderBook`` carries none —
        so a gap here is also the best available evidence that the depth stream lost frames.
        """
        if not self._enabled or TICK_STREAM not in self._streams:
            return
        try:
            previous = self._last_seq.get(symbol)
            if previous is not None and tick.seq != previous + 1:
                self.stats.seq_gaps += 1
                _log.warning(
                    "recorder.seq_gap",
                    symbol=symbol,
                    expected=previous + 1,
                    got=tick.seq,
                    impact="frames were dropped in transit; the depth stream lost them too",
                )
            self._last_seq[symbol] = tick.seq

            now = now_ist(self._clock)
            values = (
                tick.token,
                tick.ts_epoch,
                now.timestamp(),
                tick.ltp,
                tick.volume,
                tick.seq,
            )
            self._enqueue(TICK_STREAM, symbol, now, values)
            self.stats.tick_rows += 1
        except Exception as exc:  # noqa: BLE001 - recording must never break the socket loop
            self.stats.write_failures += 1
            _log.error(
                "recorder.tick_capture_failed",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this tick is lost; the feed is unaffected",
            )

    def _enqueue(self, stream: str, symbol: str, now: datetime, values: tuple[Any, ...]) -> None:
        pending = _PendingRow(
            stream=stream,
            symbol=symbol,
            on_date=now.date(),
            window=_window_label(now, self._rotate_minutes),
            values=values,
        )
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            self.stats.rows_dropped += 1
            if self.stats.rows_dropped % 1_000 == 1:
                _log.critical(
                    "recorder.queue_full",
                    stream=stream,
                    symbol=symbol,
                    dropped_total=self.stats.rows_dropped,
                    impact="rows are being lost; the writer thread is not draining",
                )

    # ── writing (writer thread) ──────────────────────────────────────────────

    def _run(self) -> None:
        """Writer thread.

        Absorbs everything. A recorder thread that dies takes the harvest with it and does so
        silently, which is the failure mode most likely to be discovered months later when the
        training set turns out to be half a session long.
        """
        while True:
            try:
                item = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                self._flush_due()
                continue

            if item is _SENTINEL:
                self._close_all_sinks()
                return
            if not isinstance(item, _PendingRow):
                continue
            try:
                self._accept(item)
            except Exception as exc:  # noqa: BLE001 - the writer thread must not die
                self.stats.write_failures += 1
                _log.critical(
                    "recorder.row_failed",
                    stream=item.stream,
                    symbol=item.symbol,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    impact="this row is lost; recording continues",
                )

    def _accept(self, pending: _PendingRow) -> None:
        sink = self._sink_for(pending)
        sink.append(pending.values)
        if sink.rows >= self._row_group_size:
            self._flush(sink)

    def _sink_for(self, pending: _PendingRow) -> _Sink:
        """Return the sink for this row, rotating the file if the partition changed."""
        key = (pending.stream, pending.symbol)
        sink = self._sinks.get(key)
        if sink is not None:
            if sink.on_date == pending.on_date and sink.window == pending.window:
                return sink
            self._close_sink(sink)
            self.stats.files_rolled += 1
            del self._sinks[key]

        schema = SCHEMAS[pending.stream]
        sink = _Sink(
            stream=pending.stream,
            symbol=pending.symbol,
            on_date=pending.on_date,
            window=pending.window,
            schema=schema,
            columns=[[] for _ in schema.names],
            last_flush_mono=self._clock.monotonic(),
        )
        self._sinks[key] = sink
        return sink

    def _flush_due(self) -> None:
        """Write a partial row group for any sink that has been holding rows too long."""
        now_mono = self._clock.monotonic()
        for sink in self._sinks.values():
            if sink.rows and now_mono - sink.last_flush_mono >= self._flush_interval:
                self._flush(sink)

    def _flush(self, sink: _Sink) -> None:
        """Write one row group. Failures are counted and swallowed, never raised."""
        if not sink.rows:
            return
        sink.last_flush_mono = self._clock.monotonic()
        try:
            table = sink.take_table()
            writer = self._writer_for(sink)
            writer.write_table(table)
            self.stats.row_groups_written += 1
        except (OSError, pa.ArrowException, ValueError) as exc:
            self.stats.write_failures += 1
            for column in sink.columns:
                column.clear()
            _log.critical(
                "recorder.write_failed",
                stream=sink.stream,
                symbol=sink.symbol,
                path=str(sink.path) if sink.path is not None else None,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this batch is lost; trading is unaffected",
            )

    def _writer_for(self, sink: _Sink) -> pq.ParquetWriter:
        if sink.writer is not None:
            return sink.writer
        path = self._path_for(sink)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"compression": self._compression}
        if self._compression == "zstd":
            kwargs["compression_level"] = _ZSTD_LEVEL
        sink.writer = pq.ParquetWriter(path, sink.schema, **kwargs)
        sink.path = path
        _log.info("recorder.file_opened", stream=sink.stream, symbol=sink.symbol, path=str(path))
        return sink.writer

    def _path_for(self, sink: _Sink) -> Path:
        """Hive-partitioned path, disambiguated if the window is already on disk.

        A restart inside a rotation window would otherwise reopen the previous run's file and
        truncate it. An extra ``-1`` file is a cosmetic wart; overwriting a morning's harvest
        is not.
        """
        partition = (
            self._directory
            / sink.stream
            / f"date={sink.on_date.isoformat()}"
            / f"symbol={sink.symbol}"
        )
        path = partition / f"{sink.window}.parquet"
        attempt = 0
        while path.exists():
            attempt += 1
            path = partition / f"{sink.window}-{attempt}.parquet"
        return path

    def _close_sink(self, sink: _Sink) -> None:
        """Flush what is buffered and write the file's footer."""
        self._flush(sink)
        writer = sink.writer
        if writer is None:
            return
        sink.writer = None
        try:
            writer.close()
        except (OSError, pa.ArrowException) as exc:
            self.stats.write_failures += 1
            _log.critical(
                "recorder.close_failed",
                path=str(sink.path) if sink.path is not None else None,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this file has no footer and cannot be read back",
            )

    def _close_all_sinks(self) -> None:
        for sink in self._sinks.values():
            self._close_sink(sink)
        self._sinks.clear()

    # ── testing ──────────────────────────────────────────────────────────────

    def drain_for_test(self) -> int:
        """Write everything queued, synchronously, and flush every buffer. Tests only.

        Lets a test assert without sleeping on a background thread. Note that the files are
        still **open** afterwards and therefore not yet readable — Parquet's footer is written
        at close. Call :meth:`close` before reading anything back.
        """
        written = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _PendingRow):
                self._accept(item)
                written += 1
        for sink in self._sinks.values():
            self._flush(sink)
        return written


def _window_label(moment: datetime, rotate_minutes: int) -> str:
    """``HHMM`` of the rotation window containing ``moment``, e.g. ``"0915"`` at 09:47/60min."""
    minute_of_day = moment.hour * 60 + moment.minute
    start = (minute_of_day // rotate_minutes) * rotate_minutes
    return f"{start // 60:02d}{start % 60:02d}"


def read_session(
    stream: str,
    *,
    directory: Path | None = None,
    on_date: date | None = None,
    symbol: str | None = None,
) -> pa.Table:
    """Load a harvested stream back, with ``date`` and ``symbol`` materialised from the path.

    This is an offline analysis tool, not part of the recording path, so it reports failure
    rather than swallowing it — the same split ``trade_logger.session_summary_from_disk`` makes.

    Raises:
        FileNotFoundError: the stream has never been recorded under ``directory``.
    """
    root = (directory if directory is not None else TICKS_DIR) / stream
    if not root.exists():
        raise FileNotFoundError(f"No recorded {stream!r} data under {root}")

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    predicates = []
    if on_date is not None:
        predicates.append(ds.field("date") == on_date.isoformat())
    if symbol is not None:
        predicates.append(ds.field("symbol") == symbol)

    if not predicates:
        return dataset.to_table()
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = combined & predicate
    return dataset.to_table(filter=combined)
