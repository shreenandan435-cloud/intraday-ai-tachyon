"""LOB windowing dataset — Arrow on disk to ``(B, T, 22)`` tensors, with no pandas anywhere.

Reads the Hive-partitioned Parquet that :mod:`tachyon.persistence.tick_recorder` writes::

    data/ticks/depth/date=2026-08-17/symbol=UFLEX/0915.parquet

and yields fixed-length windows of *normalised* book state. Raw prices and sizes are never
handed to the network: a ₹3,200 stock and a ₹18 stock share no scale, quantities are heavy-tailed
across four orders of magnitude, and a linear layer fed either one learns the instrument's price
level instead of its microstructure.

Feature construction (per row, from the 20 raw depth columns)
-------------------------------------------------------------
1. ``mid = (bid_price_0 + ask_price_0) / 2``
2. Every one of the 10 price levels becomes **signed basis points from mid**:
   ``(price - mid) / mid * 10_000``. Bids come out negative, asks positive, so the sign carries
   the side and the magnitude carries the distance. The result is scale-free across instruments.
3. Every one of the 10 quantity levels becomes ``log1p(qty)``, which compresses the volume tail
   without the ``log(0)`` singularity that an empty level would otherwise hit.
4. ``spread_tick = (ask_price_0 - bid_price_0) / tick_size`` — spread in tick units, clipped to
   ``[0, 255]`` for FP16 safety. Captures the absolute cost of crossing the book.
5. ``obi_l1 = (bid_qty_0 - ask_qty_0) / (bid_qty_0 + ask_qty_0 + eps)`` — top-level Order Book
   Imbalance in ``[-1, 1]``. Positive = bid heavy, negative = ask heavy. Zero when both sides
   are empty (unquoted).

Absent levels
-------------
The recorder persists an unquoted depth level as ``0.0`` price and ``0`` quantity, not as null.
Passed through the bps formula a zero price yields **-10 000 bps**, an outlier that would
dominate every gradient in the batch. Zero-priced levels are therefore mapped to ``0.0`` bps,
which — paired with ``log1p(0) == 0.0`` — encodes "this level is not quoted" as an exact zero in
both halves of the vector. See :func:`_normalise`.

Why windows are built from segments, not from row offsets
----------------------------------------------------------
A sliding window is only meaningful over rows that are genuinely contiguous in time. Three
things break contiguity, and all three are handled in :func:`_segment_bounds`:

* **Symbol and date boundaries.** A window spanning two ``date=`` partitions would splice 15:30
  onto the next morning's 09:15 and present an overnight gap as a single-tick move.
* **Feed outages.** A reconnect, or a stretch where the recorder's queue overflowed, leaves a
  hole. ``max_gap_seconds`` splits the block there rather than training the model on a jump it
  will never see live.
* **Rows dropped as unusable.** A book with a non-positive mid has no meaningful normalisation
  and is discarded, which itself creates a hole — caught by the same gap rule.

Memory
------
Everything requested is resident: roughly 110 bytes per row (22 float32 features, plus
``ts_epoch`` and ``mid``). One symbol-day at full SmartAPI cadence is on the order of tens of MB,
so scope a run with ``dates=`` / ``symbols=`` rather than pointing it at an entire corpus.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, Dataset

from tachyon.core.logger import get_logger
from tachyon.ipc.schemas import DEPTH_LEVELS

# Imported as a *module*, not ``from ... import TICKS_DIR``. The suite's ``_isolate_tick_output``
# fixture redirects the recorder by rebinding that module attribute, and a ``from`` import would
# have captured the original path at import time — leaving a default-constructed dataset reading
# the operator's real corpus while every other component was safely sandboxed.
from tachyon.persistence import tick_recorder
from tachyon.persistence.tick_recorder import DEPTH_STREAM

_log = get_logger(__name__)

#: Width of the model's input vector: 5 bid + 5 ask prices, then 5 bid + 5 ask quantities,
#: plus spread_tick and obi_l1.
#: Derived from :data:`~tachyon.ipc.schemas.DEPTH_LEVELS` so it tracks the wire contract — if the
#: feed ever publishes 10 levels this widens with it instead of silently truncating.
LOB_FEATURES: Final[int] = 4 * DEPTH_LEVELS + 2

#: Column order of the feature vector's last axis. Pinned by the test suite: a checkpoint trained
#: under one ordering is silently wrong under another, and nothing at inference time would catch
#: it — the tensor has the same shape and dtype either way.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"bid_bps_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_bps_{level}" for level in range(DEPTH_LEVELS)),
    *(f"bid_logqty_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_logqty_{level}" for level in range(DEPTH_LEVELS)),
    "spread_tick",
    "obi_l1",
)

#: Raw Parquet columns pulled off disk. A strict subset of ``DEPTH_COLUMNS`` — ``token`` and
#: ``recv_epoch`` are not features, and reading only what is used is most of the point of a
#: columnar format.
_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "ts_epoch",
    *(f"bid_price_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_price_{level}" for level in range(DEPTH_LEVELS)),
    *(f"bid_qty_{level}" for level in range(DEPTH_LEVELS)),
    *(f"ask_qty_{level}" for level in range(DEPTH_LEVELS)),
)

DEFAULT_SEQ_LEN: Final[int] = 128

#: Default stride of 1 — every row starts a window. Overlap is the point: at L2 rates adjacent
#: windows differ by one book update, and that is the signal.
DEFAULT_STRIDE: Final[int] = 1

#: Longer than this between consecutive books and the window is cut. Two seconds is loose enough
#: to survive a thin symbol at midday and tight enough to catch a reconnect.
DEFAULT_MAX_GAP_SECONDS: Final[float] = 2.0

_BPS_PER_UNIT: Final[float] = 10_000.0

#: Maximum spread in tick units for FP16 safety. 255 fits in uint8 and fp16 without overflow.
_MAX_SPREAD_TICK: Final[float] = 255.0

#: Epsilon for OBI denominator to avoid division by zero.
_OBI_EPS: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class _Block:
    """One contiguous ``(date, symbol)`` group, already normalised.

    ``features`` is ``(rows, LOB_FEATURES)`` float32 and is the array the returned tensors are
    views into — it is never copied per sample.
    """

    on_date: date_cls
    symbol: str
    features: npt.NDArray[np.float32]
    timestamps: npt.NDArray[np.float64]
    mid: npt.NDArray[np.float32]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """What the loader found, and what it threw away.

    ``rows_unusable`` above zero is worth reading a log line about — it means books arrived with a
    non-positive mid, which is a feed problem, not a modelling one.
    """

    groups: int = 0
    rows_read: int = 0
    rows_unusable: int = 0
    segments: int = 0
    windows: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "groups": self.groups,
            "rows_read": self.rows_read,
            "rows_unusable": self.rows_unusable,
            "segments": self.segments,
            "windows": self.windows,
        }


class LOBDataset(Dataset[torch.Tensor]):
    """Sliding windows of normalised L2 book state.

    ``dataset[i]`` is a ``(seq_len, LOB_FEATURES)`` float32 tensor. Under the default collate that
    batches to ``(B, T, 22)``, which is the contract the embedding layer is written against.

    Args:
        directory: dataset root. Defaults to :data:`~tachyon.persistence.tick_recorder.TICKS_DIR`
            *at construction*, so the suite's ``_isolate_tick_output`` fixture redirects this the
            same way it redirects the recorder.
        stream: which harvested spine to read. Only ``depth`` carries the 22 book columns.
        seq_len: window length ``T``.
        stride: rows between consecutive window starts.
        dates: restrict to these session dates. ``None`` reads every date present.
        symbols: restrict to these symbols. ``None`` reads every symbol present.
        max_gap_seconds: split a segment wherever consecutive books are further apart than this.
        tick_size_map: mapping from symbol to tick size (in INR). Required for ``spread_tick``
            normalisation. If a symbol is missing, ``spread_tick`` defaults to 0.0.
        dtype: feature dtype. float32 by default — float64 doubles the memory for precision the
            network cannot use, and the FP16 cast belongs at the autocast boundary, not on disk.

    Raises:
        FileNotFoundError: nothing has been harvested under ``directory`` for ``stream``.
        ValueError: ``seq_len``/``stride`` are non-positive.

    The dataset is read-only and holds no file handles once constructed, so it is safe to hand to
    DataLoader workers — though on Windows that is rarely the right call (see
    :func:`build_dataloader`).
    """

    def __init__(
        self,
        *,
        directory: Path | None = None,
        stream: str = DEPTH_STREAM,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = DEFAULT_STRIDE,
        dates: Iterable[date_cls] | None = None,
        symbols: Iterable[str] | None = None,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
        tick_size_map: dict[str, float] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        root = directory if directory is not None else tick_recorder.TICKS_DIR
        self._root = root / stream
        self._seq_len = seq_len
        self._stride = stride
        self._dtype = dtype
        self._tick_size_map = tick_size_map or {}

        if not self._root.exists():
            raise FileNotFoundError(f"No harvested {stream!r} data under {self._root}")

        wanted_dates = frozenset(dates) if dates is not None else None
        wanted_symbols = frozenset(symbols) if symbols is not None else None

        groups = 0
        rows_read = 0
        rows_unusable = 0
        blocks: list[_Block] = []
        # (block index, first row of the window). Materialised once so __getitem__ is a lookup
        # and a slice — no arithmetic, no bounds search, nothing per-sample that scales with N.
        index: list[tuple[int, int]] = []
        segments = 0

        for on_date, symbol, group_dir in _discover_groups(self._root):
            if wanted_dates is not None and on_date not in wanted_dates:
                continue
            if wanted_symbols is not None and symbol not in wanted_symbols:
                continue
            groups += 1

            tick_size = self._tick_size_map.get(symbol, 0.05)
            block, raw_rows, dropped = _load_block(group_dir, on_date, symbol, tick_size)
            rows_read += raw_rows
            rows_unusable += dropped
            if block is None:
                continue

            block_index = len(blocks)
            blocks.append(block)
            for start, stop in _segment_bounds(block.timestamps, max_gap_seconds):
                segments += 1
                last_start = stop - seq_len
                if last_start < start:
                    continue
                index.extend(
                    (block_index, offset) for offset in range(start, last_start + 1, stride)
                )

        self._blocks = blocks
        self._index: Final[tuple[tuple[int, int], ...]] = tuple(index)
        # One tensor per block, each a zero-copy view over the block's numpy buffer. Every
        # __getitem__ then slices a tensor that already exists, which is a view rather than an
        # allocation — the only copy in the whole path is default_collate stacking the batch.
        self._tensors = [torch.from_numpy(block.features).to(dtype) for block in blocks]

        self.stats = DatasetStats(
            groups=groups,
            rows_read=rows_read,
            rows_unusable=rows_unusable,
            segments=segments,
            windows=len(self._index),
        )
        _log.info(
            "dataset.loaded",
            root=str(self._root),
            seq_len=seq_len,
            stride=stride,
            features=LOB_FEATURES,
            **self.stats.as_dict(),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> torch.Tensor:
        block_index, start = self._index[index]
        return self._tensors[block_index][start : start + self._seq_len]

    # ── provenance, for reward construction and diagnostics ──────────────────

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def n_features(self) -> int:
        return LOB_FEATURES

    def describe(self, index: int) -> tuple[date_cls, str, int]:
        """``(date, symbol, first row)`` behind window ``index``."""
        block_index, start = self._index[index]
        block = self._blocks[block_index]
        return block.on_date, block.symbol, start

    def mid_window(self, index: int) -> npt.NDArray[np.float32]:
        """Raw mid prices under window ``index``, in rupees.

        Deliberately *not* part of ``__getitem__``: the batch contract is ``(B, T, 20)`` and the
        network must never see an un-normalised price. PPO's reward is computed from mid returns
        though, so the series has to be reachable — this is that door.
        """
        block_index, start = self._index[index]
        return self._blocks[block_index].mid[start : start + self._seq_len]

    def timestamp_window(self, index: int) -> npt.NDArray[np.float64]:
        """Exchange timestamps under window ``index``."""
        block_index, start = self._index[index]
        return self._blocks[block_index].timestamps[start : start + self._seq_len]


def build_dataloader(
    dataset: LOBDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = True,
    prefetch_factor: int | None = None,
) -> DataLoader[torch.Tensor]:
    """Wrap a :class:`LOBDataset` in a DataLoader tuned for this pipeline.

    Args:
        pin_memory: ``None`` (the default) enables it exactly when a CUDA device is present.
            Pinning is what makes the host-to-device copy asynchronous and DMA-able, so it is
            wanted on any training box; forcing ``True`` on a CPU-only machine buys nothing and
            emits a warning on every construction, which is how real warnings get ignored.
        num_workers: **0 is the right default here, and that is not a placeholder.** The dataset
            is already fully resident in memory and ``__getitem__`` is a tensor slice, so a worker
            has no I/O to overlap — it would only add a pickle of every batch through a pipe.
            Windows compounds it: ``spawn`` re-imports the parent module and re-materialises the
            arrays in each worker, multiplying resident memory by ``num_workers``.
        drop_last: ``True`` so every batch has a static leading dimension, which keeps the shapes
            that reach ONNX export uniform.

    Returns:
        A DataLoader yielding ``(batch_size, seq_len, LOB_FEATURES)`` float tensors.
    """
    resolved_pin = torch.cuda.is_available() if pin_memory is None else pin_memory

    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": resolved_pin,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        # Both are errors when num_workers == 0, so they are only ever passed alongside workers.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2 if prefetch_factor is None else prefetch_factor

    return DataLoader(dataset, **kwargs)  # type: ignore[arg-type]


# ── loading internals ────────────────────────────────────────────────────────


def _discover_groups(root: Path) -> list[tuple[date_cls, str, Path]]:
    """Enumerate ``date=<iso>/symbol=<sym>`` partitions under ``root``, sorted.

    Walking the tree rather than scanning the dataset is deliberate: the layout is a documented
    contract of :mod:`tachyon.persistence.tick_recorder`, and a directory listing costs nothing
    while ``ds.dataset(root).to_table()`` would pull an entire corpus into memory just to learn
    which symbols exist.
    """
    groups: list[tuple[date_cls, str, Path]] = []
    for date_dir in sorted(root.glob("date=*")):
        if not date_dir.is_dir():
            continue
        try:
            on_date = date_cls.fromisoformat(date_dir.name.removeprefix("date="))
        except ValueError:
            _log.warning(
                "dataset.bad_partition",
                path=str(date_dir),
                impact="not a date= partition; skipped",
            )
            continue
        for symbol_dir in sorted(date_dir.glob("symbol=*")):
            if symbol_dir.is_dir():
                groups.append((on_date, symbol_dir.name.removeprefix("symbol="), symbol_dir))
    return groups


def _load_block(
    group_dir: Path, on_date: date_cls, symbol: str, tick_size: float
) -> tuple[_Block | None, int, int]:
    """Read and normalise one ``(date, symbol)`` partition.

    Returns ``(block, rows_read, rows_dropped)``; ``block`` is ``None`` when the partition held
    nothing usable.
    """
    table = ds.dataset(group_dir, format="parquet").to_table(columns=list(_RAW_COLUMNS))
    rows_read = table.num_rows
    if rows_read == 0:
        return None, 0, 0

    # ``.combine_chunks()`` first: a ChunkedArray spanning several row groups has no single
    # contiguous buffer, so the zero-copy conversion below would refuse it. The recorder writes
    # one row group per flush, and a rotation collision leaves several files in one partition, so
    # multiple chunks is the normal case rather than the exception.
    columns = {
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in _RAW_COLUMNS
    }

    timestamps = columns["ts_epoch"].astype(np.float64, copy=False)
    # Files within a partition are not in time order — a mid-window restart writes ``0915-1``
    # beside ``0915``, and the dataset scan returns them by name. Sorting is what makes the
    # gap-based segmentation below mean anything. Stable, so equal stamps keep arrival order.
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]

    bid_price = _stack(columns, "bid_price", order)
    ask_price = _stack(columns, "ask_price", order)
    bid_qty = _stack(columns, "bid_qty", order)
    ask_qty = _stack(columns, "ask_qty", order)

    features, mid, usable = _normalise(bid_price, ask_price, bid_qty, ask_qty, tick_size)
    dropped = int(rows_read - usable.sum())
    if dropped:
        _log.warning(
            "dataset.rows_unusable",
            symbol=symbol,
            on_date=on_date.isoformat(),
            dropped=dropped,
            impact="books with a non-positive mid cannot be normalised; excluded",
        )
    if features.shape[0] == 0:
        return None, rows_read, dropped

    return (
        _Block(
            on_date=on_date,
            symbol=symbol,
            features=features,
            timestamps=timestamps[usable],
            mid=mid,
        ),
        rows_read,
        dropped,
    )


def _stack(
    columns: dict[str, npt.NDArray[np.generic]], prefix: str, order: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Gather the five levels of one side into an ``(N, DEPTH_LEVELS)`` array, row-sorted.

    ``astype(copy=False)``: the fancy-index ``[order]`` already produced a fresh array —
    a second unconditional astype copy would double this function's memory traffic.
    """
    return np.stack(
        [
            columns[f"{prefix}_{level}"][order].astype(np.float64, copy=False)
            for level in range(DEPTH_LEVELS)
        ],
        axis=1,
    )


def _normalise(
    bid_price: npt.NDArray[np.float64],
    ask_price: npt.NDArray[np.float64],
    bid_qty: npt.NDArray[np.float64],
    ask_qty: npt.NDArray[np.float64],
    tick_size: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Raw book columns to the ``(rows, 22)`` feature matrix.

    Returns ``(features, mid, usable)`` where ``usable`` is the boolean mask of retained rows —
    the caller needs it to keep the timestamp series aligned with the features.

    Everything here is a whole-array numpy operation. Doing it once per partition rather than
    once per ``__getitem__`` is what makes overlapping windows free: at ``stride=1`` a row belongs
    to ``seq_len`` different windows, so a per-sample transform would repeat this arithmetic 128
    times over.
    """
    mid = (bid_price[:, 0] + ask_price[:, 0]) / 2.0
    # A book with no quote on one side normalises to nothing meaningful — every level would be
    # ±10 000 bps from a mid of zero. Excluded rather than clipped, and counted by the caller.
    usable = mid > 0.0
    if not usable.all():
        bid_price, ask_price = bid_price[usable], ask_price[usable]
        bid_qty, ask_qty = bid_qty[usable], ask_qty[usable]
        mid = mid[usable]

    scale = mid[:, None]
    # `where=` guards the division itself, so an unquoted level never even evaluates
    # ``0 / mid`` — numpy leaves the pre-filled 0.0 in place instead of computing -10 000 bps and
    # discarding it. Same result as np.where, without the spurious intermediate.
    bid_bps = np.zeros_like(bid_price)
    ask_bps = np.zeros_like(ask_price)
    np.divide(bid_price - scale, scale, out=bid_bps, where=bid_price > 0.0)
    np.divide(ask_price - scale, scale, out=ask_bps, where=ask_price > 0.0)
    bid_bps *= _BPS_PER_UNIT
    ask_bps *= _BPS_PER_UNIT

    # ``maximum(.., 0)`` is belt-and-braces: log1p of a negative size is a NaN that would
    # propagate silently through the whole batch and only show up as a dead loss curve.
    bid_log = np.log1p(np.maximum(bid_qty, 0.0))
    ask_log = np.log1p(np.maximum(ask_qty, 0.0))

    # spread_tick = (ask_price_0 - bid_price_0) / tick_size, clipped to [0, 255] for FP16 safety
    spread = ask_price[:, 0] - bid_price[:, 0]
    spread_tick = np.clip(spread / tick_size, 0.0, _MAX_SPREAD_TICK)

    # obi_l1 = (bid_qty_0 - ask_qty_0) / (bid_qty_0 + ask_qty_0 + eps)
    bid_qty_0 = bid_qty[:, 0]
    ask_qty_0 = ask_qty[:, 0]
    obi_denom = bid_qty_0 + ask_qty_0 + _OBI_EPS
    obi_l1 = (bid_qty_0 - ask_qty_0) / obi_denom

    features = np.concatenate(
        [bid_bps, ask_bps, bid_log, ask_log, spread_tick[:, None], obi_l1[:, None]], axis=1
    )
    # ``ascontiguousarray`` matters downstream: ``torch.from_numpy`` shares the buffer, and a
    # non-contiguous one would force a copy on the first slice of every window.
    return (
        np.ascontiguousarray(features, dtype=np.float32),
        mid.astype(np.float32),
        usable,
    )


def _segment_bounds(
    timestamps: npt.NDArray[np.float64], max_gap_seconds: float
) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` runs of rows close enough in time to window across.

    A single run is returned when the series is unbroken. Any consecutive pair further apart than
    ``max_gap_seconds`` starts a new run, so no window can straddle a feed outage.
    """
    rows = int(timestamps.shape[0])
    if rows == 0:
        return []
    if rows == 1:
        return [(0, 1)]

    # +1 because diff[i] is the gap between rows i and i+1, and it is row i+1 that begins the
    # new run.
    breaks = np.flatnonzero(np.diff(timestamps) > max_gap_seconds) + 1
    edges = [0, *(int(b) for b in breaks), rows]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
