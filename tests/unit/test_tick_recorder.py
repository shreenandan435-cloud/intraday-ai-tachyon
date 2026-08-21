"""LOB harvesting — tachyon.persistence.tick_recorder.

Every test writes into ``tmp_path`` and drives the writer synchronously via ``drain_for_test``.
Nothing here touches the operator's ``data/ticks`` tree, and only the end-to-end test opens a
socket — on an ephemeral port that is never 5555 or 5556.

One recurring shape is worth stating once: Parquet writes its footer at **close**, so a file
that has only been flushed is not yet readable. Tests that assert on contents call
``recorder.close()`` first, which is exactly what the sidecar does on shutdown.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import OrderBook, Tick
from tachyon.ipc.subscriber import Subscriber, SubscriberRole
from tachyon.persistence.tick_recorder import (
    DEPTH_COLUMNS,
    DEPTH_STREAM,
    TICK_COLUMNS,
    TICK_STREAM,
    TickRecorder,
    read_session,
)

AT = datetime(2026, 8, 17, 9, 47, tzinfo=IST)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(AT)


@pytest.fixture
def recorder(tmp_path: Path, clock: ManualClock) -> Iterator[TickRecorder]:
    """A recorder with no background thread — tests drain it themselves."""
    instance = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)
    yield instance
    instance.close()


def _book(
    *,
    token: str = "1053",
    ts_epoch: float = 1_755_000_000.0,
    bid_price: tuple[float, ...] = (100.5, 100.4, 100.3, 100.2, 100.1),
    bid_qty: tuple[int, ...] = (10, 20, 30, 40, 50),
    ask_price: tuple[float, ...] = (100.6, 100.7, 100.8, 100.9, 101.0),
    ask_qty: tuple[int, ...] = (11, 21, 31, 41, 51),
) -> OrderBook:
    return OrderBook(
        token=token,
        bid_price=bid_price,  # type: ignore[arg-type]
        bid_qty=bid_qty,  # type: ignore[arg-type]
        ask_price=ask_price,  # type: ignore[arg-type]
        ask_qty=ask_qty,  # type: ignore[arg-type]
        ts_epoch=ts_epoch,
    )


def _tick(
    *,
    token: str = "1053",
    ltp: float = 100.55,
    volume: int = 12_345,
    ts_epoch: float = 1_755_000_000.0,
    seq: int = 1,
) -> Tick:
    return Tick(token=token, ltp=ltp, volume=volume, ts_epoch=ts_epoch, seq=seq)


def _partition(root: Path, stream: str, symbol: str, on_date: str = "2026-08-17") -> Path:
    return root / stream / f"date={on_date}" / f"symbol={symbol}"


def _files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


class TestDepthSchema:
    def test_round_trip_preserves_every_level(self, recorder: TickRecorder, tmp_path: Path) -> None:
        recorder.record_depth("UFLEX", _book())
        assert recorder.drain_for_test() == 1
        recorder.close()

        path = _partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX") / "0900.parquet"
        table = pq.read_table(path)

        assert table.column_names == list(DEPTH_COLUMNS)
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["token"] == "1053"
        assert row["ts_epoch"] == 1_755_000_000.0
        assert row["bid_price_0"] == 100.5
        assert row["bid_price_4"] == 100.1
        assert row["ask_qty_0"] == 11
        assert row["ask_qty_4"] == 51

    def test_empty_levels_persist_as_zero_not_null(
        self, recorder: TickRecorder, tmp_path: Path
    ) -> None:
        """The ingestor pads short ladders with 0.0/0 so the shape stays fixed for the Numba
        OBI kernel. If the recorder turned those into nulls, the offline loader would have to
        guess whether a level was empty or unrecorded — and those mean different things."""
        recorder.record_depth(
            "UFLEX",
            _book(
                bid_price=(100.5, 100.4, 0.0, 0.0, 0.0),
                bid_qty=(10, 20, 0, 0, 0),
            ),
        )
        recorder.drain_for_test()
        recorder.close()

        table = pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX"))
        row = table.to_pylist()[0]
        assert row["bid_price_2"] == 0.0
        assert row["bid_qty_4"] == 0
        assert table.column("bid_price_2").null_count == 0

    def test_recv_epoch_is_wall_clock_not_monotonic(
        self, recorder: TickRecorder, clock: ManualClock, tmp_path: Path
    ) -> None:
        """A monotonic reading is relative to a per-process origin and is meaningless offline.
        ``recv_epoch - ts_epoch`` must be a real latency."""
        recorder.record_depth("UFLEX", _book(ts_epoch=AT.timestamp() - 0.25))
        recorder.drain_for_test()
        recorder.close()

        row = pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")).to_pylist()[0]
        assert row["recv_epoch"] == pytest.approx(AT.timestamp())
        assert row["recv_epoch"] - row["ts_epoch"] == pytest.approx(0.25)
        assert clock.monotonic() == 0.0  # the monotonic clock never entered the file


class TestTickSchema:
    def test_round_trip(self, recorder: TickRecorder, tmp_path: Path) -> None:
        recorder.record_tick("UFLEX", _tick(seq=7, volume=999))
        recorder.drain_for_test()
        recorder.close()

        table = pq.read_table(_partition(tmp_path / "ticks", TICK_STREAM, "UFLEX"))
        assert table.column_names == list(TICK_COLUMNS)
        row = table.to_pylist()[0]
        assert row["ltp"] == 100.55
        assert row["volume"] == 999
        assert row["seq"] == 7

    def test_streams_are_written_to_separate_trees(
        self, recorder: TickRecorder, tmp_path: Path
    ) -> None:
        recorder.record_depth("UFLEX", _book())
        recorder.record_tick("UFLEX", _tick())
        recorder.drain_for_test()
        recorder.close()

        root = tmp_path / "ticks"
        assert (_partition(root, DEPTH_STREAM, "UFLEX") / "0900.parquet").exists()
        assert (_partition(root, TICK_STREAM, "UFLEX") / "0900.parquet").exists()


class TestPartitioning:
    def test_hive_layout(self, recorder: TickRecorder, tmp_path: Path) -> None:
        recorder.record_depth("UFLEX", _book())
        recorder.record_depth("ZAGGLE", _book(token="18608"))
        recorder.drain_for_test()
        recorder.close()

        written = {p.relative_to(tmp_path / "ticks").as_posix() for p in _files(tmp_path / "ticks")}
        assert written == {
            "depth/date=2026-08-17/symbol=UFLEX/0900.parquet",
            "depth/date=2026-08-17/symbol=ZAGGLE/0900.parquet",
        }

    def test_read_session_materialises_date_and_symbol(
        self, recorder: TickRecorder, tmp_path: Path
    ) -> None:
        recorder.record_depth("UFLEX", _book())
        recorder.record_depth("ZAGGLE", _book(token="18608"))
        recorder.drain_for_test()
        recorder.close()

        table = read_session(DEPTH_STREAM, directory=tmp_path / "ticks", symbol="UFLEX")
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["symbol"] == "UFLEX"
        assert row["date"] == "2026-08-17"
        assert row["token"] == "1053"

    def test_read_session_on_missing_stream_raises(self, tmp_path: Path) -> None:
        """An operator tool reports failure rather than swallowing it."""
        with pytest.raises(FileNotFoundError):
            read_session(DEPTH_STREAM, directory=tmp_path / "ticks")


class TestRotation:
    def test_crossing_the_window_rolls_the_file(
        self, recorder: TickRecorder, clock: ManualClock, tmp_path: Path
    ) -> None:
        recorder.record_depth("UFLEX", _book())
        clock.advance(20 * 60)  # 09:47 -> 10:07, across the hourly boundary
        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        recorder.close()

        partition = _partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")
        assert (partition / "0900.parquet").exists()
        assert (partition / "1000.parquet").exists()
        assert recorder.stats.files_rolled == 1
        assert pq.read_table(partition / "0900.parquet").num_rows == 1
        assert pq.read_table(partition / "1000.parquet").num_rows == 1

    def test_crossing_midnight_rolls_the_date_partition(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        clock.wall = datetime(2026, 8, 17, 23, 59, tzinfo=IST)
        recorder = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)
        recorder.record_depth("UFLEX", _book())
        clock.advance(120)
        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        recorder.close()

        root = tmp_path / "ticks"
        assert (_partition(root, DEPTH_STREAM, "UFLEX", "2026-08-17") / "2300.parquet").exists()
        assert (_partition(root, DEPTH_STREAM, "UFLEX", "2026-08-18") / "0000.parquet").exists()

    def test_partition_is_stamped_at_capture_not_at_write(
        self, recorder: TickRecorder, clock: ManualClock, tmp_path: Path
    ) -> None:
        """A book captured at 09:59:59 must land in the 09:00 file even if the writer thread
        only gets to it after 10:00. Resolving the window at write time would smear the
        boundary by however far behind the writer happened to be."""
        recorder.record_depth("UFLEX", _book())
        clock.advance(60 * 60)  # the writer is now an hour behind
        recorder.drain_for_test()
        recorder.close()

        partition = _partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")
        assert (partition / "0900.parquet").exists()
        assert not (partition / "1000.parquet").exists()

    def test_restart_inside_a_window_does_not_truncate_the_earlier_file(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Reopening the same path would silently destroy the morning's harvest."""
        first = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)
        first.record_depth("UFLEX", _book())
        first.drain_for_test()
        first.close()

        second = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)
        second.record_depth("UFLEX", _book())
        second.drain_for_test()
        second.close()

        partition = _partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")
        assert pq.read_table(partition / "0900.parquet").num_rows == 1
        assert pq.read_table(partition / "0900-1.parquet").num_rows == 1


class TestRowGroups:
    def test_row_group_flushes_at_the_configured_size(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        recorder = TickRecorder(
            directory=tmp_path / "ticks", clock=clock, row_group_size=4, start=False
        )
        for _ in range(9):
            recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        assert recorder.stats.row_groups_written == 3  # 4 + 4 + the forced partial 1
        recorder.close()

        table = pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX"))
        assert table.num_rows == 9

    def test_no_rows_leaves_no_empty_file(self, recorder: TickRecorder, tmp_path: Path) -> None:
        recorder.drain_for_test()
        recorder.close()
        assert _files(tmp_path / "ticks") == []


class TestBackpressure:
    def test_a_full_queue_drops_rows_and_never_raises(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Blocking here would couple disk latency to the socket loop, which is the one thing
        this design exists to make impossible."""
        recorder = TickRecorder(
            directory=tmp_path / "ticks", clock=clock, queue_size=2, start=False
        )
        for _ in range(5):
            recorder.record_depth("UFLEX", _book())  # must not raise

        assert recorder.stats.rows_dropped == 3
        assert recorder.drain_for_test() == 2
        recorder.close()
        assert pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")).num_rows == 2

    def test_streams_can_be_switched_off(self, tmp_path: Path, clock: ManualClock) -> None:
        recorder = TickRecorder(
            directory=tmp_path / "ticks", clock=clock, streams=(DEPTH_STREAM,), start=False
        )
        recorder.record_depth("UFLEX", _book())
        recorder.record_tick("UFLEX", _tick())
        recorder.drain_for_test()
        recorder.close()

        assert recorder.stats.depth_rows == 1
        assert recorder.stats.tick_rows == 0
        assert not (tmp_path / "ticks" / TICK_STREAM).exists()

    def test_disabled_records_nothing(self, tmp_path: Path, clock: ManualClock) -> None:
        recorder = TickRecorder(
            directory=tmp_path / "ticks", clock=clock, enabled=False, start=False
        )
        recorder.record_depth("UFLEX", _book())
        recorder.record_tick("UFLEX", _tick())
        assert recorder.drain_for_test() == 0
        recorder.close()
        assert _files(tmp_path / "ticks") == []


class TestFailureIsolation:
    def test_a_write_failure_is_counted_not_raised(
        self, recorder: TickRecorder, tmp_path: Path
    ) -> None:
        """A plain file where the partition directory belongs — no mocks, a real OSError.

        The obstruction is put one level *above* the Parquet file on purpose. Blocking the file
        itself is not a failure: ``_path_for`` sidesteps an occupied name and writes beside it.
        Blocking the directory is unroutable, which is what this asserts survives.
        """
        date_dir = tmp_path / "ticks" / DEPTH_STREAM / "date=2026-08-17"
        date_dir.mkdir(parents=True)
        (date_dir / "symbol=UFLEX").write_text("not a directory", encoding="utf-8")

        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()  # must not raise

        assert recorder.stats.write_failures == 1
        assert recorder.stats.row_groups_written == 0

    def test_recording_continues_after_a_write_failure(
        self, recorder: TickRecorder, tmp_path: Path
    ) -> None:
        """A full disk must not become 'we stopped harvesting'."""
        date_dir = tmp_path / "ticks" / DEPTH_STREAM / "date=2026-08-17"
        date_dir.mkdir(parents=True)
        blocker = date_dir / "symbol=UFLEX"
        blocker.write_text("not a directory", encoding="utf-8")

        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        assert recorder.stats.write_failures == 1

        blocker.unlink()
        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        recorder.close()

        assert pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")).num_rows == 1

    def test_a_broken_message_does_not_break_the_socket_loop(
        self, recorder: TickRecorder
    ) -> None:
        class Exploding:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError("boom")

        recorder.record_depth("UFLEX", Exploding())  # type: ignore[arg-type]
        recorder.record_tick("UFLEX", Exploding())  # type: ignore[arg-type]

        assert recorder.stats.write_failures == 2
        assert recorder.drain_for_test() == 0


class TestSeqGaps:
    def test_a_gap_is_counted(self, recorder: TickRecorder) -> None:
        """``Tick.seq`` is the only message-loss signal on the wire — ``OrderBook`` has none."""
        recorder.record_tick("UFLEX", _tick(seq=1))
        recorder.record_tick("UFLEX", _tick(seq=2))
        recorder.record_tick("UFLEX", _tick(seq=9))
        assert recorder.stats.seq_gaps == 1

    def test_sequences_are_tracked_per_symbol(self, recorder: TickRecorder) -> None:
        """One symbol's gap must not implicate another's — publishers assign seq per symbol."""
        recorder.record_tick("UFLEX", _tick(seq=1))
        recorder.record_tick("ZAGGLE", _tick(seq=1))
        recorder.record_tick("UFLEX", _tick(seq=2))
        recorder.record_tick("ZAGGLE", _tick(seq=2))
        assert recorder.stats.seq_gaps == 0


class TestLifecycle:
    def test_close_joins_the_writer_thread(self, tmp_path: Path, clock: ManualClock) -> None:
        before = threading.active_count()
        recorder = TickRecorder(directory=tmp_path / "ticks", clock=clock)
        assert recorder.stats.depth_rows == 0
        recorder.record_depth("UFLEX", _book())
        recorder.close()
        assert threading.active_count() == before, "the writer thread was not joined"

    def test_the_running_thread_writes_a_readable_file(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """The real shutdown path: no drain_for_test, just close(). Every footer must land."""
        recorder = TickRecorder(directory=tmp_path / "ticks", clock=clock)
        for index in range(50):
            recorder.record_depth("UFLEX", _book(ts_epoch=1_755_000_000.0 + index))
        recorder.close()

        table = pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX"))
        assert table.num_rows == 50

    def test_close_is_idempotent(self, tmp_path: Path, clock: ManualClock) -> None:
        recorder = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)
        recorder.record_depth("UFLEX", _book())
        recorder.drain_for_test()
        recorder.close()
        recorder.close()  # must not raise or re-close the writer

    def test_context_manager_closes(self, tmp_path: Path, clock: ManualClock) -> None:
        with TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False) as recorder:
            recorder.record_depth("UFLEX", _book())
            recorder.drain_for_test()
        assert pq.read_table(_partition(tmp_path / "ticks", DEPTH_STREAM, "UFLEX")).num_rows == 1


class TestEndToEnd:
    """Real ZeroMQ over real TCP. A mock would happily lie about the frame contract."""

    @pytest.fixture
    def endpoint(self) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return f"tcp://127.0.0.1:{probe.getsockname()[1]}"

    def test_published_books_and_ticks_land_in_parquet(
        self, endpoint: str, tmp_path: Path, clock: ManualClock
    ) -> None:
        publisher = Publisher(endpoint, role="test-ingestor")
        # The publisher binds before the subscriber connects; reversing that leaves the
        # subscription inside ZeroMQ's reconnect interval and the first frames are lost.
        subscriber = Subscriber(
            ("DEPTH.", "TICK."), role=SubscriberRole.TELEMETRY, endpoint=endpoint
        )
        Publisher.settle(0.4)
        recorder = TickRecorder(directory=tmp_path / "ticks", clock=clock, start=False)

        try:
            for index in range(20):
                publisher.publish_orderbook("UFLEX", _book(ts_epoch=1_755_000_000.0 + index))
                publisher.publish_tick("UFLEX", _tick(seq=index + 1))

            for _ in range(40):
                envelope = subscriber.recv(timeout_ms=2000)
                assert envelope is not None, "the publisher went silent"
                if isinstance(envelope.message, OrderBook):
                    recorder.record_depth("UFLEX", envelope.message)
                elif isinstance(envelope.message, Tick):
                    recorder.record_tick("UFLEX", envelope.message)
        finally:
            subscriber.close()
            publisher.close()

        recorder.drain_for_test()
        recorder.close()

        root = tmp_path / "ticks"
        assert pq.read_table(_partition(root, DEPTH_STREAM, "UFLEX")).num_rows == 20
        assert pq.read_table(_partition(root, TICK_STREAM, "UFLEX")).num_rows == 20
        assert recorder.stats.rows_dropped == 0
        assert recorder.stats.seq_gaps == 0
