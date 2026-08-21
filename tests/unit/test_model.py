"""Track 2 data and embedding layers — tachyon.model.dataset, tachyon.model.embedding.

Test fixtures are written by :class:`~tachyon.persistence.tick_recorder.TickRecorder` itself
rather than by a hand-rolled Parquet writer. That is deliberate: the thing under test is a reader
of a format another module owns, and a bespoke fixture writer would let the two drift apart while
every test stayed green. If the recorder's schema or partition layout changes, these tests break
here, which is where the breakage belongs.

Nothing touches the operator's ``data/ticks``. Every dataset is built under ``tmp_path``, and the
one default-constructed dataset relies on the session-wide ``_isolate_tick_output`` fixture in
``tests/conftest.py`` — see :class:`TestCorpusIsolation`, which exists to prove that still holds.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from tachyon.core.clock import IST, ManualClock
from tachyon.ipc.schemas import DEPTH_LEVELS, OrderBook
from tachyon.model.dataset import (
    DEFAULT_MAX_GAP_SECONDS,
    FEATURE_NAMES,
    LOB_FEATURES,
    LOBDataset,
    build_dataloader,
)
from tachyon.model.embedding import (
    DEFAULT_MAX_SEQ_LEN,
    LOBEmbedding,
    RotaryPositionalEmbedding,
)
from tachyon.persistence.tick_recorder import DEPTH_STREAM, TickRecorder

#: Embedding input width: 22 order book features (from dataset) + 3 portfolio state features
#: (position, entry_price_bps, holding_bars) injected by the RL environment.
EMBEDDING_IN_FEATURES: Final[int] = 25

#: Session date every fixture partition is stamped with, unless a test advances the clock.
AT = datetime(2026, 8, 17, 9, 47, tzinfo=IST)
BASE_TS = 1_755_000_000.0

#: One book update every 100 ms — comfortably inside ``DEFAULT_MAX_GAP_SECONDS``, so a run built
#: with it is a single unbroken segment unless a test deliberately punches a hole in it.
TICK_INTERVAL = 0.1


# ── fixture construction ─────────────────────────────────────────────────────


def _book(
    *,
    ts_epoch: float,
    bid_price: Sequence[float] = (100.5, 100.4, 100.3, 100.2, 100.1),
    bid_qty: Sequence[int] = (10, 200, 3_000, 40_000, 500_000),
    ask_price: Sequence[float] = (100.7, 100.8, 100.9, 101.0, 101.1),
    ask_qty: Sequence[int] = (11, 210, 3_100, 41_000, 510_000),
) -> OrderBook:
    """One L2 snapshot. Quantities span five orders of magnitude on purpose — that spread is
    exactly what ``log1p`` exists to compress, and a fixture with flat sizes would not show it."""
    return OrderBook(
        token="1053",
        bid_price=tuple(bid_price),  # type: ignore[arg-type]
        bid_qty=tuple(bid_qty),  # type: ignore[arg-type]
        ask_price=tuple(ask_price),  # type: ignore[arg-type]
        ask_qty=tuple(ask_qty),  # type: ignore[arg-type]
        ts_epoch=ts_epoch,
    )


def _harvest(
    root: Path,
    symbol: str,
    books: Sequence[OrderBook],
    *,
    moment: datetime = AT,
) -> None:
    """Write ``books`` into ``root`` through the real recorder, then close it.

    ``close()`` is what writes each Parquet footer, so a fixture that skipped it would leave
    files no reader can open — the same trap the recorder's own suite documents.
    """
    recorder = TickRecorder(
        directory=root,
        clock=ManualClock(moment),
        streams=(DEPTH_STREAM,),
        start=False,
    )
    for book in books:
        recorder.record_depth(symbol, book)
    recorder.drain_for_test()
    recorder.close()


def _ramp(
    count: int, *, start_ts: float = BASE_TS, interval: float = TICK_INTERVAL
) -> list[OrderBook]:
    """``count`` books whose mid walks steadily upward, evenly spaced in time.

    The *whole* book shifts, not just the touch. Sliding top-of-book while the deeper levels stay
    pinned would leave level 4 drifting arbitrarily far from the mid — an artefact of the fixture
    that no real book exhibits, and one that would make the feature-range assertions meaningless.
    """
    return [
        _book(
            ts_epoch=start_ts + i * interval,
            bid_price=tuple(p + i * 0.01 for p in (100.5, 100.4, 100.3, 100.2, 100.1)),
            ask_price=tuple(p + i * 0.01 for p in (100.7, 100.8, 100.9, 101.0, 101.1)),
        )
        for i in range(count)
    ]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A 200-row single-symbol, single-day harvest — the default subject of most tests."""
    root = tmp_path / "ticks"
    _harvest(root, "UFLEX", _ramp(200))
    return root


# ── feature contract ─────────────────────────────────────────────────────────


class TestFeatureContract:
    def test_the_feature_width_is_four_values_per_depth_level_plus_two(self) -> None:
        # 4 * DEPTH_LEVELS (20) + spread_tick + obi_l1 = 22
        assert LOB_FEATURES == 4 * DEPTH_LEVELS + 2 == 22

    def test_feature_names_describe_every_column_in_order(self) -> None:
        # Pinned rather than merely counted. A checkpoint trained under one ordering produces
        # confident nonsense under another, and the tensor's shape and dtype are identical either
        # way — nothing downstream could detect the swap.
        assert len(FEATURE_NAMES) == LOB_FEATURES
        assert FEATURE_NAMES[:5] == tuple(f"bid_bps_{i}" for i in range(5))
        assert FEATURE_NAMES[5:10] == tuple(f"ask_bps_{i}" for i in range(5))
        assert FEATURE_NAMES[10:15] == tuple(f"bid_logqty_{i}" for i in range(5))
        assert FEATURE_NAMES[15:20] == tuple(f"ask_logqty_{i}" for i in range(5))
        assert FEATURE_NAMES[20] == "spread_tick"
        assert FEATURE_NAMES[21] == "obi_l1"


class TestNormalisation:
    def test_prices_become_signed_basis_points_from_mid(self, corpus: Path) -> None:
        dataset = LOBDataset(directory=corpus, seq_len=4)
        row = dataset[0][0]

        # First fixture book: bid_0 = 100.5, ask_0 = 100.7, so mid = 100.6.
        mid = (100.5 + 100.7) / 2
        assert row[0].item() == pytest.approx((100.5 - mid) / mid * 10_000, abs=1e-3)
        assert row[5].item() == pytest.approx((100.7 - mid) / mid * 10_000, abs=1e-3)
        # Deeper levels stay ordered outward from the mid.
        assert row[1].item() == pytest.approx((100.4 - mid) / mid * 10_000, abs=1e-3)

    def test_bids_are_negative_and_asks_are_positive(self, corpus: Path) -> None:
        window = LOBDataset(directory=corpus, seq_len=8)[0]
        assert (window[:, :DEPTH_LEVELS] < 0).all(), "every bid sits below the mid"
        assert (window[:, DEPTH_LEVELS : 2 * DEPTH_LEVELS] > 0).all(), "every ask sits above it"

    def test_basis_points_are_scale_free_across_instruments(self, tmp_path: Path) -> None:
        """The whole reason for bps: a ₹100 stock and a ₹3,000 stock with the same *relative*
        spread must produce the same feature values, or the network learns the price level."""
        cheap = tmp_path / "cheap"
        rich = tmp_path / "rich"
        _harvest(
            cheap,
            "CHEAP",
            [
                _book(
                    ts_epoch=BASE_TS,
                    bid_price=(99.0, 98.0, 97.0, 96.0, 95.0),
                    ask_price=(101.0, 102.0, 103.0, 104.0, 105.0),
                )
            ]
            * 4,
        )
        _harvest(
            rich,
            "RICH",
            [
                _book(
                    ts_epoch=BASE_TS,
                    bid_price=(2_970.0, 2_940.0, 2_910.0, 2_880.0, 2_850.0),
                    ask_price=(3_030.0, 3_060.0, 3_090.0, 3_120.0, 3_150.0),
                )
            ]
            * 4,
        )

        cheap_row = LOBDataset(directory=cheap, seq_len=4)[0][0]
        rich_row = LOBDataset(directory=rich, seq_len=4)[0][0]
        # 1% away from mid on both, at a 30x price difference.
        assert torch.allclose(cheap_row[:10], rich_row[:10], atol=1e-2)

    def test_quantities_are_log1p_compressed(self, corpus: Path) -> None:
        row = LOBDataset(directory=corpus, seq_len=4)[0][0]
        for level, qty in enumerate((10, 200, 3_000, 40_000, 500_000)):
            assert row[10 + level].item() == pytest.approx(math.log1p(qty), abs=1e-4)
        for level, qty in enumerate((11, 210, 3_100, 41_000, 510_000)):
            assert row[15 + level].item() == pytest.approx(math.log1p(qty), abs=1e-4)

    def test_an_unquoted_level_is_an_exact_zero_in_both_halves(self, tmp_path: Path) -> None:
        """The single most damaging thing this normalisation prevents.

        The recorder stores an absent depth level as price ``0.0`` and quantity ``0``. Run
        naively through ``(price - mid) / mid``, a zero price is **-10 000 bps** — an outlier two
        hundred times the magnitude of any real feature, present in a large fraction of rows on a
        thin symbol, and more than enough on its own to dominate the gradient.
        """
        root = tmp_path / "ticks"
        _harvest(
            root,
            "THIN",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.5, 100.4, 0.0, 0.0, 0.0),
                    bid_qty=(10, 20, 0, 0, 0),
                    ask_price=(100.7, 100.8, 0.0, 0.0, 0.0),
                    ask_qty=(11, 21, 0, 0, 0),
                )
                for i in range(8)
            ],
        )

        row = LOBDataset(directory=root, seq_len=4)[0][0]
        assert row[2].item() == 0.0 and row[3].item() == 0.0 and row[4].item() == 0.0
        assert row[7].item() == 0.0 and row[8].item() == 0.0 and row[9].item() == 0.0
        assert row[12].item() == 0.0 and row[13].item() == 0.0 and row[14].item() == 0.0
        assert row[17].item() == 0.0 and row[18].item() == 0.0 and row[19].item() == 0.0
        # And nothing anywhere near the -10 000 bps a naive formula would have produced.
        assert row.abs().max().item() < 100.0

    def test_features_stay_in_a_range_gradient_descent_can_use(self, corpus: Path) -> None:
        """The bar is the order of magnitude, not a tight bound.

        Raw inputs would be ~1e2 (price) and ~1e5 (size), spanning three decades between the two
        halves of the same vector. Both halves land in the tens here: the fixture's outermost
        level sits ₹0.50 from a ₹100.60 mid, so ~50 bps, and ``log1p(500_000)`` is ~13.
        """
        window = LOBDataset(directory=corpus, seq_len=32)[0]
        assert window.abs().max().item() < 100.0
        assert torch.isfinite(window).all()

    def test_spread_tick_is_normalised_by_tick_size(self, tmp_path: Path) -> None:
        """spread_tick = (ask_price_0 - bid_price_0) / tick_size.

        For a ₹0.05 tick size, a ₹0.20 spread = 4 ticks.
        """
        root = tmp_path / "ticks"
        # tick_size = 0.05, spread = 0.20 → 4 ticks
        _harvest(
            root,
            "TEST",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.0, 99.9, 99.8, 99.7, 99.6),
                    ask_price=(100.2, 100.3, 100.4, 100.5, 100.6),
                    bid_qty=(100,) * 5,
                    ask_qty=(100,) * 5,
                )
                for i in range(8)
            ],
        )

        dataset = LOBDataset(directory=root, seq_len=4, tick_size_map={"TEST": 0.05})
        row = dataset[0][0]
        # spread_tick is at index 20
        assert row[20].item() == pytest.approx(4.0, abs=1e-3)

    def test_obi_l1_is_normalised_order_book_imbalance(self, tmp_path: Path) -> None:
        """obi_l1 = (bid_qty_0 - ask_qty_0) / (bid_qty_0 + ask_qty_0 + eps).

        Range [-1, 1]. Positive = bid heavy, negative = ask heavy.
        """
        root = tmp_path / "ticks"
        _harvest(
            root,
            "TEST",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.0,) * 5,
                    ask_price=(100.1,) * 5,
                    bid_qty=(1000, 200, 300, 400, 500),
                    ask_qty=(500, 200, 300, 400, 500),
                )
                for i in range(8)
            ],
        )

        dataset = LOBDataset(directory=root, seq_len=4, tick_size_map={"TEST": 0.05})
        row = dataset[0][0]
        # obi_l1 is at index 21: (1000 - 500) / (1000 + 500) = 500/1500 = 1/3
        assert row[21].item() == pytest.approx(1 / 3, abs=1e-3)

    def test_obi_l1_zero_when_both_sides_empty(self, tmp_path: Path) -> None:
        """When bid_qty_0 == 0 and ask_qty_0 == 0, obi_l1 = 0 (not NaN).

        Prices must be non-zero so mid > 0 and the book is usable.
        """
        root = tmp_path / "ticks"
        _harvest(
            root,
            "TEST",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.0,) * 5,
                    ask_price=(100.1,) * 5,
                    bid_qty=(0,) * 5,
                    ask_qty=(0,) * 5,
                )
                for i in range(8)
            ],
        )

        dataset = LOBDataset(directory=root, seq_len=4, tick_size_map={"TEST": 0.05})
        row = dataset[0][0]
        assert row[21].item() == 0.0

    def test_obi_l1_negative_when_ask_heavy(self, tmp_path: Path) -> None:
        """obi_l1 negative when ask quantity dominates."""
        root = tmp_path / "ticks"
        _harvest(
            root,
            "TEST",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.0,) * 5,
                    ask_price=(100.1,) * 5,
                    bid_qty=(100, 200, 300, 400, 500),
                    ask_qty=(1000, 200, 300, 400, 500),
                )
                for i in range(8)
            ],
        )

        dataset = LOBDataset(directory=root, seq_len=4, tick_size_map={"TEST": 0.05})
        row = dataset[0][0]
        # (100 - 1000) / (100 + 1000) = -900/1100 ≈ -0.818
        assert row[21].item() == pytest.approx(-900 / 1100, abs=1e-3)

    def test_spread_tick_clipped_for_fp16_safety(self, tmp_path: Path) -> None:
        """spread_tick clipped to [0, 255] to prevent fp16 overflow."""
        root = tmp_path / "ticks"
        # Extreme spread: 1000 ticks = ₹50 spread on ₹0.05 tick
        _harvest(
            root,
            "TEST",
            [
                _book(
                    ts_epoch=BASE_TS + i * TICK_INTERVAL,
                    bid_price=(100.0,) * 5,
                    ask_price=(150.0,) * 5,  # ₹50 spread = 1000 ticks
                    bid_qty=(100,) * 5,
                    ask_qty=(100,) * 5,
                )
                for i in range(8)
            ],
        )

        dataset = LOBDataset(directory=root, seq_len=4, tick_size_map={"TEST": 0.05})
        row = dataset[0][0]
        # Should be clipped to 255
        assert row[20].item() == 255.0

    def test_a_book_with_no_mid_is_dropped_and_counted(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        books = _ramp(10)
        # A wholly unquoted book: mid would be 0.0, and every level infinitely far from it.
        books[5] = _book(
            ts_epoch=BASE_TS + 5 * TICK_INTERVAL,
            bid_price=(0.0,) * 5,
            bid_qty=(0,) * 5,
            ask_price=(0.0,) * 5,
            ask_qty=(0,) * 5,
        )
        _harvest(root, "HALTED", books)

        dataset = LOBDataset(directory=root, seq_len=3)
        assert dataset.stats.rows_read == 10
        assert dataset.stats.rows_unusable == 1
        assert torch.isfinite(torch.stack([dataset[i] for i in range(len(dataset))])).all()


# ── windowing ────────────────────────────────────────────────────────────────


class TestWindowing:
    def test_a_sample_has_the_shape_the_embedding_expects(self, corpus: Path) -> None:
        window = LOBDataset(directory=corpus, seq_len=64)[0]
        assert window.shape == (64, LOB_FEATURES)
        assert window.dtype is torch.float32

    def test_window_count_follows_from_rows_seq_len_and_stride(self, corpus: Path) -> None:
        assert len(LOBDataset(directory=corpus, seq_len=50, stride=1)) == 200 - 50 + 1
        assert len(LOBDataset(directory=corpus, seq_len=50, stride=10)) == (200 - 50) // 10 + 1

    def test_windows_advance_by_exactly_stride_rows(self, corpus: Path) -> None:
        dataset = LOBDataset(directory=corpus, seq_len=8, stride=3)
        assert dataset.describe(0)[2] == 0
        assert dataset.describe(1)[2] == 3
        assert dataset.describe(2)[2] == 6

    def test_a_block_shorter_than_the_window_yields_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        _harvest(root, "TINY", _ramp(10))
        dataset = LOBDataset(directory=root, seq_len=64)
        assert len(dataset) == 0
        assert dataset.stats.rows_read == 10

    def test_no_window_spans_two_symbols(self, tmp_path: Path) -> None:
        """Two 100-row symbols must give 2 x (100 - T + 1) windows, not 200 - T + 1.

        The difference is the T-1 windows that would straddle the boundary, each of which splices
        one instrument's book onto another's and presents it as a single price series.
        """
        root = tmp_path / "ticks"
        _harvest(root, "AAA", _ramp(100))
        _harvest(root, "BBB", _ramp(100))

        dataset = LOBDataset(directory=root, seq_len=20)
        assert len(dataset) == 2 * (100 - 20 + 1)
        assert {dataset.describe(i)[1] for i in range(len(dataset))} == {"AAA", "BBB"}

    def test_no_window_spans_two_sessions(self, tmp_path: Path) -> None:
        """A window crossing midnight would present an overnight gap as one book update."""
        root = tmp_path / "ticks"
        _harvest(root, "UFLEX", _ramp(60), moment=AT)
        _harvest(root, "UFLEX", _ramp(60, start_ts=BASE_TS + 86_400), moment=AT + timedelta(days=1))

        dataset = LOBDataset(directory=root, seq_len=20)
        assert len(dataset) == 2 * (60 - 20 + 1)
        assert {dataset.describe(i)[0] for i in range(len(dataset))} == {
            date(2026, 8, 17),
            date(2026, 8, 18),
        }

    def test_a_feed_outage_splits_the_block_rather_than_spanning_it(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        first = _ramp(40)
        # A 30-second hole — a reconnect, or a stretch the recorder's queue overflowed.
        second = _ramp(40, start_ts=BASE_TS + 40 * TICK_INTERVAL + 30.0)
        _harvest(root, "UFLEX", [*first, *second])

        dataset = LOBDataset(directory=root, seq_len=10)
        assert dataset.stats.segments == 2
        assert len(dataset) == 2 * (40 - 10 + 1)

    def test_a_gap_inside_the_tolerance_does_not_split(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        gap = DEFAULT_MAX_GAP_SECONDS / 2
        _harvest(root, "UFLEX", [_book(ts_epoch=BASE_TS + i * gap) for i in range(20)])
        assert LOBDataset(directory=root, seq_len=5).stats.segments == 1

    def test_rows_are_time_ordered_even_when_files_are_not(self, tmp_path: Path) -> None:
        """A restart inside a rotation window leaves ``0915-1.parquet`` beside ``0915.parquet``.

        The dataset scan returns those by filename, so the second file's rows arrive after the
        first file's regardless of when they actually happened. Two harvests are done here in
        reverse time order to force exactly that; the loader must sort them back.
        """
        root = tmp_path / "ticks"
        _harvest(root, "UFLEX", _ramp(30, start_ts=BASE_TS + 1_000.0))  # later, written first
        _harvest(root, "UFLEX", _ramp(30, start_ts=BASE_TS))  # earlier, written second

        partition = root / DEPTH_STREAM / "date=2026-08-17" / "symbol=UFLEX"
        assert len(list(partition.glob("*.parquet"))) == 2, "the collision suffix path was taken"

        dataset = LOBDataset(directory=root, seq_len=5)
        stamps = np.concatenate([dataset.timestamp_window(i) for i in range(len(dataset))])
        assert (np.diff(dataset.timestamp_window(0)) > 0).all()
        assert stamps.min() == pytest.approx(BASE_TS)


class TestConstruction:
    def test_a_non_positive_seq_len_is_rejected(self, corpus: Path) -> None:
        with pytest.raises(ValueError, match="seq_len must be positive"):
            LOBDataset(directory=corpus, seq_len=0)

    def test_a_non_positive_stride_is_rejected(self, corpus: Path) -> None:
        with pytest.raises(ValueError, match="stride must be positive"):
            LOBDataset(directory=corpus, stride=-1)

    def test_an_unharvested_root_reports_where_it_looked(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No harvested 'depth' data"):
            LOBDataset(directory=tmp_path / "nothing-here")

    def test_symbols_and_dates_narrow_the_scan(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        _harvest(root, "AAA", _ramp(50))
        _harvest(root, "BBB", _ramp(50))

        dataset = LOBDataset(directory=root, seq_len=10, symbols=["AAA"])
        assert dataset.stats.groups == 1
        assert {dataset.describe(i)[1] for i in range(len(dataset))} == {"AAA"}

        by_date = LOBDataset(directory=root, seq_len=10, dates=[date(2026, 8, 17)])
        assert by_date.stats.groups == 2
        assert LOBDataset(directory=root, seq_len=10, dates=[date(2020, 1, 1)]).stats.groups == 0

    def test_a_stray_directory_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        root = tmp_path / "ticks"
        _harvest(root, "UFLEX", _ramp(30))
        (root / DEPTH_STREAM / "date=not-a-date").mkdir()

        assert LOBDataset(directory=root, seq_len=10).stats.groups == 1


class TestZeroCopy:
    def test_a_sample_is_a_view_not_a_copy(self, corpus: Path) -> None:
        """Overlapping windows must not each allocate.

        At ``stride=1`` a single row belongs to ``seq_len`` different windows. If ``__getitem__``
        copied, one epoch would move ``seq_len`` times the dataset's size through the allocator —
        128x at the default, which is the difference between a loader that keeps a GPU fed and one
        that does not.
        """
        dataset = LOBDataset(directory=corpus, seq_len=16)
        first, second = dataset[0], dataset[1]
        assert first.data_ptr() != second.data_ptr(), "different windows start at different rows"
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr(), (
            "but both are views into the one block buffer"
        )

    def test_repeated_reads_return_the_same_storage(self, corpus: Path) -> None:
        dataset = LOBDataset(directory=corpus, seq_len=16)
        assert dataset[7].data_ptr() == dataset[7].data_ptr()


class TestProvenance:
    def test_mid_prices_are_reachable_without_entering_the_batch(self, corpus: Path) -> None:
        """PPO's reward comes from mid returns, so the raw series must be recoverable — while
        staying out of ``__getitem__``, which the network sees."""
        dataset = LOBDataset(directory=corpus, seq_len=16)
        mid = dataset.mid_window(0)
        assert mid.shape == (16,)
        assert mid[0] == pytest.approx((100.5 + 100.7) / 2, abs=1e-3)
        assert (np.diff(mid) > 0).all(), "the fixture mid ramps upward"

    def test_a_window_reports_the_partition_it_came_from(self, corpus: Path) -> None:
        assert LOBDataset(directory=corpus, seq_len=16).describe(3) == (
            date(2026, 8, 17),
            "UFLEX",
            3,
        )


# ── dataloader ───────────────────────────────────────────────────────────────


class TestDataLoader:
    def test_a_batch_is_b_t_20(self, corpus: Path) -> None:
        loader = build_dataloader(
            LOBDataset(directory=corpus, seq_len=32), batch_size=8, shuffle=False
        )
        batch = next(iter(loader))
        assert batch.shape == (8, 32, LOB_FEATURES)
        assert batch.dtype is torch.float32

    def test_the_batch_feeds_the_embedding_unchanged(self, corpus: Path) -> None:
        """The seam between the two deliverables, asserted rather than assumed.

        Dataset produces 22 order book features; embedding accepts 25 (22 + 3 portfolio).
        This test verifies backward compatibility with n_features=22.
        """
        loader = build_dataloader(
            LOBDataset(directory=corpus, seq_len=32), batch_size=4, shuffle=False
        )
        batch = next(iter(loader))
        # Use n_features=LOB_FEATURES (22) to match dataset output
        embedded = LOBEmbedding(d_model=64, n_features=LOB_FEATURES).eval()(batch)
        assert embedded.shape == (4, 32, 64)
        assert torch.isfinite(embedded).all()

    def test_embedding_accepts_25_features_with_portfolio_state(self) -> None:
        """Embedding accepts 25 features: 22 order book + 3 portfolio state."""
        module = LOBEmbedding(d_model=64).eval()  # default n_features=25
        x = torch.randn(2, 16, EMBEDDING_IN_FEATURES)
        out = module(x)
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    def test_drop_last_keeps_every_batch_the_same_width(self, corpus: Path) -> None:
        dataset = LOBDataset(directory=corpus, seq_len=32)
        loader = build_dataloader(dataset, batch_size=7, shuffle=False, drop_last=True)
        assert {tuple(b.shape) for b in loader} == {(7, 32, LOB_FEATURES)}

    def test_pin_memory_defaults_to_whether_a_gpu_exists(self, corpus: Path) -> None:
        """``None`` means "pin if it would help". Forcing ``True`` on a CPU-only box buys nothing
        and emits a UserWarning on every construction, which is how real warnings get tuned out.
        """
        loader = build_dataloader(LOBDataset(directory=corpus, seq_len=8))
        assert loader.pin_memory is torch.cuda.is_available()
        assert (
            build_dataloader(LOBDataset(directory=corpus, seq_len=8), pin_memory=False).pin_memory
            is False
        )

    def test_worker_only_options_are_absent_without_workers(self, corpus: Path) -> None:
        """``persistent_workers`` and ``prefetch_factor`` raise when ``num_workers == 0``, so
        they must be passed conditionally rather than defaulted."""
        loader = build_dataloader(LOBDataset(directory=corpus, seq_len=8), num_workers=0)
        assert loader.num_workers == 0
        assert loader.persistent_workers is False


class TestCorpusIsolation:
    def test_a_default_dataset_never_reads_the_operators_corpus(self) -> None:
        """``_isolate_tick_output`` in ``tests/conftest.py`` redirects the recorder's module-level
        ``TICKS_DIR``. This module must resolve that attribute at call time for the redirect to
        reach it — a ``from ... import TICKS_DIR`` would have captured the real path at import.
        """
        from tachyon.persistence import tick_recorder

        assert "AppData" in str(tick_recorder.TICKS_DIR) or "pytest" in str(
            tick_recorder.TICKS_DIR
        ), "the session fixture should have redirected this into a tmp dir"
        with pytest.raises(FileNotFoundError) as excinfo:
            LOBDataset(seq_len=8)
        assert str(tick_recorder.TICKS_DIR) in str(excinfo.value)


# ── rotary position embedding ────────────────────────────────────────────────


class TestRotaryPositionalEmbedding:
    def test_rotation_preserves_shape_and_dtype(self) -> None:
        rope = RotaryPositionalEmbedding(64)
        x = torch.randn(2, 16, 64)
        assert rope(x).shape == x.shape
        assert rope(x).dtype is x.dtype

    def test_rotation_preserves_vector_norm(self) -> None:
        """RoPE is an orthogonal transform. If the norm moves, the construction is wrong — and a
        rotation that quietly rescales would show up only as a training instability."""
        rope = RotaryPositionalEmbedding(64)
        x = torch.randn(2, 32, 64)
        assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_position_zero_is_the_identity(self) -> None:
        rope = RotaryPositionalEmbedding(64)
        x = torch.randn(1, 8, 64)
        assert torch.allclose(rope(x)[0, 0], x[0, 0], atol=1e-6)

    def test_attention_scores_depend_only_on_relative_offset(self) -> None:
        """The property that makes RoPE worth having over absolute encodings.

        ``<R_m q, R_n k>`` must be a function of ``m - n`` alone. A window is a moving frame over
        the session; what matters is that a sweep happened four updates ago, not that it landed at
        index 71 of this particular slice.
        """
        rope = RotaryPositionalEmbedding(64, max_seq_len=256)
        q = torch.randn(1, 1, 64).expand(1, 256, 64).contiguous()
        k = torch.randn(1, 1, 64).expand(1, 256, 64).contiguous()
        rq, rk = rope(q), rope(k)

        near = (rq[0, 10] * rk[0, 14]).sum()
        far = (rq[0, 200] * rk[0, 204]).sum()
        assert near.item() == pytest.approx(far.item(), abs=1e-4)

        different_offset = (rq[0, 10] * rk[0, 30]).sum()
        assert different_offset.item() != pytest.approx(near.item(), abs=1e-3)

    def test_it_broadcasts_over_a_head_dimension(self) -> None:
        """RoPE's proper home is inside attention, applied to queries and keys per head. The
        tables broadcast as ``(T, dim)``, so the same module serves both call sites."""
        rope = RotaryPositionalEmbedding(32)
        assert rope(torch.randn(2, 4, 16, 32)).shape == (2, 4, 16, 32)

    def test_the_tables_stay_out_of_the_checkpoint(self) -> None:
        """They are a pure function of ``(dim, max_seq_len, theta)``. Persisting them would add
        derivable megabytes and let a stale checkpoint override a corrected table on load."""
        assert RotaryPositionalEmbedding(64).state_dict() == {}

    def test_an_odd_width_cannot_be_rotated(self) -> None:
        with pytest.raises(ValueError, match="positive and even"):
            RotaryPositionalEmbedding(63)

    def test_a_non_positive_table_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            RotaryPositionalEmbedding(64, max_seq_len=0)

    def test_a_window_longer_than_the_tables_fails_loudly(self) -> None:
        """Truncating silently would be far worse: the rotation would be applied to the first
        ``max_seq_len`` rows and the rest left unpositioned, and nothing would raise."""
        rope = RotaryPositionalEmbedding(32, max_seq_len=16)
        with pytest.raises(RuntimeError):
            rope(torch.randn(1, 64, 32))


# ── embedding ────────────────────────────────────────────────────────────────


class TestLOBEmbedding:
    def test_it_projects_b_t_25_to_b_t_d_model(self) -> None:
        module = LOBEmbedding(d_model=128).eval()
        assert module(torch.randn(4, 64, EMBEDDING_IN_FEATURES)).shape == (4, 64, 128)

    def test_the_input_width_tracks_the_embedding_schema(self) -> None:
        # Embedding accepts 25 features: 22 from dataset + 3 portfolio state
        assert LOBEmbedding().n_features == EMBEDDING_IN_FEATURES
        assert LOBEmbedding().projection.in_features == EMBEDDING_IN_FEATURES

    def test_eval_output_is_deterministic(self) -> None:
        module = LOBEmbedding(d_model=32, dropout=0.5).eval()
        x = torch.randn(2, 8, EMBEDDING_IN_FEATURES)
        assert torch.equal(module(x), module(x)), "dropout must be inert in eval()"

    def test_dropout_is_live_in_train_mode(self) -> None:
        module = LOBEmbedding(d_model=32, dropout=0.5).train()
        x = torch.randn(2, 8, EMBEDDING_IN_FEATURES)
        assert not torch.equal(module(x), module(x))

    def test_rope_can_be_deferred_to_the_attention_layers(self) -> None:
        """The more standard placement is inside attention on q/k only. Turning it off here must
        actually change the output, or the flag is decorative."""
        x = torch.randn(2, 8, EMBEDDING_IN_FEATURES)
        torch.manual_seed(0)
        with_rope = LOBEmbedding(d_model=32, apply_rope=True).eval()
        torch.manual_seed(0)
        without = LOBEmbedding(d_model=32, apply_rope=False).eval()
        assert not torch.allclose(with_rope(x), without(x))

    def test_an_odd_d_model_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="d_model must be positive and even"):
            LOBEmbedding(d_model=127)

    def test_a_non_positive_feature_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_features must be positive"):
            LOBEmbedding(n_features=0)

    def test_gradients_reach_the_projection(self) -> None:
        module = LOBEmbedding(d_model=32)
        module(torch.randn(2, 16, EMBEDDING_IN_FEATURES)).sum().backward()
        assert module.projection.weight.grad is not None
        assert torch.isfinite(module.projection.weight.grad).all()

    def test_the_default_window_fits_inside_the_rotary_tables(self) -> None:
        from tachyon.model.dataset import DEFAULT_SEQ_LEN

        assert DEFAULT_SEQ_LEN <= DEFAULT_MAX_SEQ_LEN


class TestHalfPrecision:
    """FP16 safety, which autocast alone does not give you.

    Under ``.half()`` the *parameters* are fp16 too, so relying on autocast's fp32 op list is not
    enough — the LayerNorm and the rotary rotation have to up-cast their own weights.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_it_is_transparent_under_autocast(self, dtype: torch.dtype) -> None:
        """The training-time half of the story: weights stay fp32, activations drop to half.

        The module must not force the output back up to fp32, or every downstream block inherits
        an fp32 residual stream and the tensor cores go unused for the rest of the network.
        """
        module = LOBEmbedding(d_model=64).eval()
        with torch.autocast("cpu", dtype=dtype):
            out = module(torch.randn(2, 16, EMBEDDING_IN_FEATURES))
        assert out.dtype is dtype
        assert torch.isfinite(out).all()

    def test_a_fully_halved_module_still_runs(self) -> None:
        module = LOBEmbedding(d_model=64).eval().half()
        out = module(torch.randn(2, 16, EMBEDDING_IN_FEATURES, dtype=torch.float16))
        assert out.dtype is torch.float16
        assert torch.isfinite(out).all()

    def test_layer_norm_does_not_overflow_on_a_wide_input(self) -> None:
        """LayerNorm's reduction accumulates ``d_model`` squared values. In fp16 that sum
        saturates at 65 504, and a saturated variance yields NaN out of the reciprocal sqrt."""
        module = LOBEmbedding(d_model=512).eval().half()
        # 200 bps is a violent but entirely possible move on a thin Indian mid-cap.
        out = module(torch.full((1, 8, EMBEDDING_IN_FEATURES), 200.0, dtype=torch.float16))
        assert torch.isfinite(out).all()

    def test_half_and_float_agree_to_half_precision(self) -> None:
        """Halving must cost precision, not correctness — an FP16 TensorRT engine that diverges
        from the trained model is a silently different policy."""
        torch.manual_seed(7)
        module = LOBEmbedding(d_model=64).eval()
        x = torch.randn(2, 16, EMBEDDING_IN_FEATURES)
        reference = module(x)
        halved = copy.deepcopy(module).half()
        assert torch.allclose(reference, halved(x.half()).float(), atol=2e-2)

    def test_the_rotary_tables_are_kept_in_fp32(self) -> None:
        """fp16 spacing near 1.0 is ~5e-4. At ``seq_len=128`` that quantises neighbouring
        positions into each other, destroying the position signal without raising anything."""
        module = LOBEmbedding(d_model=64).eval().half()
        assert module.rope.cos_table.dtype is torch.float32
        assert module.rope.sin_table.dtype is torch.float32


# ── export safety: the hard constraint ───────────────────────────────────────


class TestExportSafety:
    """The inference target is ONNX Runtime compiled to TensorRT at <1ms.

    A module that trains fine but cannot be traced is worthless here, and that is discovered at
    conversion time — after the training run — unless it is asserted now.
    """

    @staticmethod
    def _module() -> LOBEmbedding:
        return LOBEmbedding(d_model=64, max_seq_len=256).eval()

    # torch.jit.script is deprecated on Python 3.14 and torch warns on every call. The check is
    # kept anyway, and the warning suppressed rather than the test deleted: scripting is the only
    # one of these three that *rejects* Python-level constructs instead of silently baking their
    # first-call result into the graph, so it fails loudly on exactly the mistakes tracing hides.
    @pytest.mark.filterwarnings("ignore:`torch.jit.script`:DeprecationWarning")
    def test_it_compiles_under_torch_jit_script(self) -> None:
        module = self._module()
        scripted = torch.jit.script(module)
        x = torch.randn(2, 32, EMBEDDING_IN_FEATURES)
        assert torch.allclose(scripted(x), module(x), atol=1e-6)

    def test_it_compiles_under_torch_export(self) -> None:
        """``torch.export`` is the front half of the dynamo ONNX path — if this fails, the ONNX
        exporter has nothing to work from."""
        module = self._module()
        x = torch.randn(2, 32, EMBEDDING_IN_FEATURES)
        exported = torch.export.export(module, (x,))
        assert torch.allclose(exported.module()(x), module(x), atol=1e-6)

    def test_the_exported_graph_holds_no_conditional(self) -> None:
        """A data-dependent branch survives export as a control-flow op that the TensorRT parser
        either rejects or falls back to a non-fused subgraph for."""
        x = torch.randn(2, 32, EMBEDDING_IN_FEATURES)
        graph = torch.export.export(self._module(), (x,)).graph
        rendered = str(graph)
        assert "torch.ops.higher_order.cond" not in rendered
        assert "while_loop" not in rendered

    def test_batch_and_sequence_length_export_as_dynamic_axes(self) -> None:
        """Batch must be dynamic for serving, and sequence length is a training hyper-parameter
        that must not be baked into the artefact."""
        module = self._module()
        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq", min=2, max=256)
        exported = torch.export.export(
            module,
            (torch.randn(4, 32, EMBEDDING_IN_FEATURES),),
            dynamic_shapes={"x": {0: batch, 1: seq}},
        )
        for shape in ((1, 16, EMBEDDING_IN_FEATURES), (9, 64, EMBEDDING_IN_FEATURES)):
            sample = torch.randn(*shape)
            assert torch.allclose(exported.module()(sample), module(sample), atol=1e-6)

    def test_it_exports_to_onnx_and_onnxruntime_reproduces_it(self, tmp_path: Path) -> None:
        """The end of the chain, run for real rather than asserted about.

        A graph that exports but computes something else is the failure this catches — and it is
        the plausible one, because RoPE's table slicing and the fp32 LayerNorm up-cast are exactly
        the constructs an exporter is most likely to mistranslate.
        """
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        onnxruntime = pytest.importorskip("onnxruntime")

        module = self._module()
        x = torch.randn(2, 32, EMBEDDING_IN_FEATURES)
        path = tmp_path / "lob_embedding.onnx"
        torch.onnx.export(
            module,
            (x,),
            str(path),
            input_names=["lob_state"],
            output_names=["embedding"],
            # dynamic_shapes, not dynamic_axes: the dynamo exporter converts the legacy form
            # through a deprecated shim that warns and can violate its own constraints.
            dynamic_shapes={"x": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq", min=2)}},
            dynamo=True,
        )
        assert path.exists()

        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (produced,) = session.run(None, {"lob_state": x.numpy()})
        assert produced.shape == (2, 32, 64)
        np.testing.assert_allclose(produced, module(x).detach().numpy(), atol=1e-4)
