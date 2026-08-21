"""Pre-allocated ring buffers — CLAUDE.md §3.2.

Zero allocation on the tick path. No ``list.append``, no ``np.concatenate``, no growing
DataFrame. Numba kernels want a contiguous C-array, and building one per tick would defeat
the point of compiling them.

The wrap-around problem
-----------------------
A naive ring buffer stores the newest element at ``i = n % capacity``, so once it wraps, the
chronological data is split into two runs: ``[i:]`` then ``[:i]``. Returning that as a single
array requires ``np.concatenate``, which allocates and copies on *every* tick.

This implementation uses the mirrored-buffer trick: the backing array is ``2 * capacity`` and
every value is written **twice**, at ``i`` and at ``i + capacity``. Any window of ``capacity``
consecutive slots is then guaranteed to be contiguous in memory, so :meth:`RingBuffer.view`
is a plain slice — no allocation, no copy, and the result is already C-contiguous and
correctly ordered oldest-to-newest.

The cost is 2x memory and one extra store per append. At the configured sizes that is
32 KB per float64 tick buffer, which is nothing against never allocating in the hot path.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt

#: Every kernel takes float64. Mixing dtypes forces Numba to recompile per signature.
FLOAT: Final = np.float64
INT: Final = np.int64


class RingBuffer:
    """Fixed-capacity 1-D circular buffer over a pre-allocated NumPy array.

    Args:
        capacity: number of elements retained. Older values are silently overwritten.
        dtype: element type, ``float64`` by default.

    Example::

        buf = RingBuffer(2000)
        buf.append(101.5)
        prices = buf.view()          # contiguous, oldest -> newest, no copy
    """

    __slots__ = ("_capacity", "_data", "_count", "_index")

    def __init__(self, capacity: int, dtype: npt.DTypeLike = FLOAT) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        # Mirrored: slots [0, capacity) and [capacity, 2*capacity) always hold the same value.
        self._data: npt.NDArray[Any] = np.zeros(capacity * 2, dtype=dtype)
        self._index = 0  # where the next value is written, in [0, capacity)
        self._count = 0  # total appends, saturating at capacity

    # ── writing ──────────────────────────────────────────────────────────────

    def append(self, value: float | int) -> None:
        """Add one value. O(1), no allocation."""
        index = self._index
        self._data[index] = value
        self._data[index + self._capacity] = value
        self._index = (index + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def extend(self, values: npt.NDArray[Any]) -> None:
        """Add many values in order. Used for backfill, never on the tick path."""
        for value in values:
            self.append(value)

    def clear(self) -> None:
        """Drop all data. The backing allocation is retained and reused."""
        self._index = 0
        self._count = 0

    # ── reading ──────────────────────────────────────────────────────────────

    def view(self) -> npt.NDArray[Any]:
        """Contiguous view of the valid data, oldest first.

        This is a *view*, not a copy — it stays valid only until the next :meth:`append`.
        Pass it straight into a Numba kernel; never store it.
        """
        if self._count < self._capacity:
            return self._data[: self._count]
        # Full: the oldest element sits at _index. Thanks to the mirror, the window
        # [_index, _index + capacity) is contiguous and chronologically ordered.
        return self._data[self._index : self._index + self._capacity]

    def last(self, n: int) -> npt.NDArray[Any]:
        """Contiguous view of the most recent ``n`` values, oldest first.

        Returns fewer than ``n`` if the buffer does not yet hold that many.
        """
        available = min(n, self._count)
        if available == 0:
            return self._data[:0]
        return self.view()[-available:]

    @property
    def latest(self) -> float:
        """The most recently appended value.

        Raises:
            IndexError: if nothing has been appended.
        """
        if self._count == 0:
            raise IndexError("RingBuffer is empty")
        return float(self._data[(self._index - 1) % self._capacity])

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, len={self._count})"


class CandleBuffer:
    """Parallel ring buffers forming an OHLCV series.

    ATR reads highs, lows and closes as three separate contiguous arrays that must stay
    index-aligned, so they advance together through :meth:`push` and can never drift apart.
    """

    __slots__ = ("_closes", "_highs", "_lows", "_opens", "_starts", "_volumes", "capacity")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._opens = RingBuffer(capacity, FLOAT)
        self._highs = RingBuffer(capacity, FLOAT)
        self._lows = RingBuffer(capacity, FLOAT)
        self._closes = RingBuffer(capacity, FLOAT)
        self._volumes = RingBuffer(capacity, FLOAT)
        self._starts = RingBuffer(capacity, FLOAT)

    def push(
        self,
        *,
        start_epoch: float,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Append one completed candle."""
        self._starts.append(start_epoch)
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._volumes.append(volume)

    def clear(self) -> None:
        for buffer in (
            self._starts,
            self._opens,
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
        ):
            buffer.clear()

    @property
    def highs(self) -> npt.NDArray[Any]:
        return self._highs.view()

    @property
    def lows(self) -> npt.NDArray[Any]:
        return self._lows.view()

    @property
    def closes(self) -> npt.NDArray[Any]:
        return self._closes.view()

    @property
    def opens(self) -> npt.NDArray[Any]:
        return self._opens.view()

    @property
    def volumes(self) -> npt.NDArray[Any]:
        return self._volumes.view()

    @property
    def starts(self) -> npt.NDArray[Any]:
        return self._starts.view()

    def __len__(self) -> int:
        return len(self._closes)

    def __repr__(self) -> str:
        return f"CandleBuffer(capacity={self.capacity}, len={len(self)})"
