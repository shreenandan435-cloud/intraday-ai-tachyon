"""SmartAPI WebSocket 2.0 binary packet decoder — CLAUDE.md §2.

Pure functions over ``bytes``. No sockets, no state, no logging on the happy path — this is
the hottest code in the ingestion process and it is kept trivially testable against synthetic
byte arrays.

Wire format
-----------
Little-endian, packed, **no alignment padding** (hence ``<`` on every format string). Three
subscription modes share a common 51-byte prefix and grow rightward:

===================  =====  ===========================================================
Mode                 Bytes  Adds
===================  =====  ===========================================================
1 — LTP                 51  mode, exchange, token, sequence, timestamp, LTP
2 — Quote              123  LTQ, ATP, volume, total buy/sell qty, OHLC
3 — Snap Quote         379  last-traded ts, OI, OI change, **best five depth**, circuits
===================  =====  ===========================================================

The best-five block is 200 bytes: ten 20-byte entries of
``(buy_sell_flag: int16, quantity: int64, price: int64, orders: int16)``.

Field offsets are expressed as ``struct.Struct`` objects compiled once at import. Sizes are
asserted against the documented packet lengths at import time — an off-by-one in an offset
would otherwise surface as plausible-looking wrong prices rather than a crash.

**Prices arrive as integers scaled by 100** (paise for NSE cash). They are divided on decode.
Currency derivatives use a different scale; see :data:`PRICE_DIVISORS`.

.. warning::
   This is implemented against Angel One's published protocol specification. The synthetic
   round-trip tests prove the unpacking matches that spec exactly; they cannot prove the spec
   matches the live feed. Validate against a real session in PAPER mode before going live —
   in particular the timestamp unit and the price scale for any non-NSE-cash segment.
"""

from __future__ import annotations

import struct
from datetime import datetime
from enum import IntEnum
from typing import Final

import msgspec

from tachyon.core.clock import IST


class SubscriptionMode(IntEnum):
    """SmartAPI subscription modes."""

    LTP = 1
    QUOTE = 2
    SNAP_QUOTE = 3


class ExchangeType(IntEnum):
    """SmartAPI exchange type codes."""

    NSE_CM = 1
    NSE_FO = 2
    BSE_CM = 3
    BSE_FO = 4
    MCX_FO = 5
    NCX_FO = 7
    CDE_FO = 13


#: Integer price scale by exchange. Everything except currency derivatives is paise.
PRICE_DIVISORS: Final[dict[int, float]] = {
    ExchangeType.NSE_CM: 100.0,
    ExchangeType.NSE_FO: 100.0,
    ExchangeType.BSE_CM: 100.0,
    ExchangeType.BSE_FO: 100.0,
    ExchangeType.MCX_FO: 100.0,
    ExchangeType.NCX_FO: 100.0,
    ExchangeType.CDE_FO: 10_000_000.0,
}
DEFAULT_PRICE_DIVISOR: Final[float] = 100.0

DEPTH_LEVELS: Final[int] = 5

# ── Compiled layouts ─────────────────────────────────────────────────────────
# mode, exchange, token[25], sequence, exchange_ts, ltp
_LTP_LAYOUT: Final[struct.Struct] = struct.Struct("<bb25sqqq")

# ... + ltq, atp, volume, total_buy_qty(f64), total_sell_qty(f64), open, high, low, close
_QUOTE_LAYOUT: Final[struct.Struct] = struct.Struct("<bb25sqqqqqqddqqqq")

# ... + last_traded_ts, open_interest, oi_change_pct(f64)   [ends at byte 147]
_SNAP_HEAD_LAYOUT: Final[struct.Struct] = struct.Struct("<bb25sqqqqqqddqqqqqqd")

# 10 x (buy_sell_flag: i16, quantity: i64, price: i64, orders: i16)  [147..347]
_BEST_FIVE_LAYOUT: Final[struct.Struct] = struct.Struct("<" + "hqqh" * 10)

# upper circuit, lower circuit, 52w high, 52w low                    [347..379]
_SNAP_TAIL_LAYOUT: Final[struct.Struct] = struct.Struct("<qqqq")

_BEST_FIVE_OFFSET: Final[int] = _SNAP_HEAD_LAYOUT.size
_SNAP_TAIL_OFFSET: Final[int] = _BEST_FIVE_OFFSET + _BEST_FIVE_LAYOUT.size

LTP_PACKET_SIZE: Final[int] = _LTP_LAYOUT.size
QUOTE_PACKET_SIZE: Final[int] = _QUOTE_LAYOUT.size
SNAP_QUOTE_PACKET_SIZE: Final[int] = _SNAP_TAIL_OFFSET + _SNAP_TAIL_LAYOUT.size

#: Packet length by mode, for framing a stream of concatenated packets.
PACKET_SIZES: Final[dict[int, int]] = {
    SubscriptionMode.LTP: LTP_PACKET_SIZE,
    SubscriptionMode.QUOTE: QUOTE_PACKET_SIZE,
    SubscriptionMode.SNAP_QUOTE: SNAP_QUOTE_PACKET_SIZE,
}

# Guard the documented sizes. A mistyped format character shifts every subsequent field and
# would produce wrong-but-plausible prices rather than an error, so fail at import instead.
assert LTP_PACKET_SIZE == 51, f"LTP packet must be 51 bytes, got {LTP_PACKET_SIZE}"
assert QUOTE_PACKET_SIZE == 123, f"Quote packet must be 123 bytes, got {QUOTE_PACKET_SIZE}"
assert SNAP_QUOTE_PACKET_SIZE == 379, (
    f"Snap Quote packet must be 379 bytes, got {SNAP_QUOTE_PACKET_SIZE}"
)

#: Below this, an "epoch milliseconds" value is implausible (it would be before 1973), which
#: almost certainly means the field is actually in seconds.
_MIN_PLAUSIBLE_EPOCH_MS: Final[int] = 100_000_000_000


class PacketDecodeError(ValueError):
    """A payload could not be decoded against the SmartAPI v2 specification."""


class DepthLevel(msgspec.Struct, frozen=True, gc=False):
    """One rung of the L2 ladder."""

    price: float
    quantity: int
    orders: int


class SnapQuote(msgspec.Struct, frozen=True, gc=False):
    """A decoded Mode 3 (Full Snap Quote) packet.

    Prices are already divided by the exchange scale, so they are rupees.
    """

    token: str
    exchange_type: int
    exchange_sequence: int
    exchange_timestamp_ms: int
    ltp: float
    last_traded_quantity: int
    average_traded_price: float
    volume: int
    total_buy_quantity: float
    total_sell_quantity: float
    open: float
    high: float
    low: float
    close: float
    last_traded_timestamp_ms: int
    open_interest: int
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    upper_circuit: float
    lower_circuit: float
    week_52_high: float
    week_52_low: float

    @property
    def ts_epoch(self) -> float:
        """Exchange timestamp as Unix seconds — what goes on the ZMQ wire."""
        return self.exchange_timestamp_ms / 1000.0

    @property
    def ts_ist(self) -> datetime:
        """Exchange timestamp as a timezone-aware IST datetime (CLAUDE.md §8)."""
        return datetime.fromtimestamp(self.ts_epoch, tz=IST)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0


class LtpPacket(msgspec.Struct, frozen=True, gc=False):
    """A decoded Mode 1 packet — the common prefix of every mode."""

    token: str
    exchange_type: int
    exchange_sequence: int
    exchange_timestamp_ms: int
    ltp: float

    @property
    def ts_epoch(self) -> float:
        return self.exchange_timestamp_ms / 1000.0

    @property
    def ts_ist(self) -> datetime:
        return datetime.fromtimestamp(self.ts_epoch, tz=IST)


def price_divisor(exchange_type: int) -> float:
    """Integer-to-rupee scale for an exchange segment."""
    return PRICE_DIVISORS.get(exchange_type, DEFAULT_PRICE_DIVISOR)


def decode_token(raw: bytes) -> str:
    """Read the null-padded 25-byte ASCII token field.

    The field is fixed width and zero-padded, so trailing NULs must be stripped — otherwise
    the token never matches the watchlist and every tick is silently discarded downstream.
    """
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def peek_mode(payload: bytes, offset: int = 0) -> int:
    """Read the subscription mode byte without decoding the packet."""
    if len(payload) <= offset:
        raise PacketDecodeError("empty payload: no subscription mode byte")
    return payload[offset]


def _decode_depth(
    payload: bytes, divisor: float
) -> tuple[tuple[DepthLevel, ...], tuple[DepthLevel, ...]]:
    """Split the 200-byte best-five block into bid and ask ladders.

    Entries are classified by their ``buy_sell_flag`` rather than by position. The
    specification says the first five entries are buys and the next five sells, but trusting
    position would silently invert the order book if that ever changed — and an inverted book
    flips the sign of every OBI reading, which is worse than a decode error.
    """
    fields = _BEST_FIVE_LAYOUT.unpack_from(payload, _BEST_FIVE_OFFSET)

    bids: list[DepthLevel] = []
    asks: list[DepthLevel] = []
    for index in range(10):
        flag, quantity, raw_price, orders = fields[index * 4 : index * 4 + 4]
        level = DepthLevel(price=raw_price / divisor, quantity=quantity, orders=orders)
        if flag == 1:
            bids.append(level)
        else:
            asks.append(level)

    return tuple(bids), tuple(asks)


def decode_snap_quote(payload: bytes) -> SnapQuote:
    """Decode one Mode 3 (Full Snap Quote) packet.

    Args:
        payload: exactly :data:`SNAP_QUOTE_PACKET_SIZE` bytes.

    Raises:
        PacketDecodeError: wrong length, or the mode byte is not ``SNAP_QUOTE``.
    """
    if len(payload) != SNAP_QUOTE_PACKET_SIZE:
        raise PacketDecodeError(
            f"Snap Quote packet must be {SNAP_QUOTE_PACKET_SIZE} bytes, got {len(payload)}"
        )

    (
        mode,
        exchange_type,
        raw_token,
        exchange_sequence,
        exchange_timestamp_ms,
        raw_ltp,
        last_traded_quantity,
        raw_atp,
        volume,
        total_buy_quantity,
        total_sell_quantity,
        raw_open,
        raw_high,
        raw_low,
        raw_close,
        last_traded_timestamp_ms,
        open_interest,
        _oi_change_pct,
    ) = _SNAP_HEAD_LAYOUT.unpack_from(payload, 0)

    if mode != SubscriptionMode.SNAP_QUOTE:
        raise PacketDecodeError(f"expected mode {SubscriptionMode.SNAP_QUOTE}, got {mode}")

    divisor = price_divisor(exchange_type)
    bids, asks = _decode_depth(payload, divisor)
    upper_circuit, lower_circuit, week_52_high, week_52_low = _SNAP_TAIL_LAYOUT.unpack_from(
        payload, _SNAP_TAIL_OFFSET
    )

    return SnapQuote(
        token=decode_token(raw_token),
        exchange_type=exchange_type,
        exchange_sequence=exchange_sequence,
        exchange_timestamp_ms=exchange_timestamp_ms,
        ltp=raw_ltp / divisor,
        last_traded_quantity=last_traded_quantity,
        average_traded_price=raw_atp / divisor,
        volume=volume,
        total_buy_quantity=total_buy_quantity,
        total_sell_quantity=total_sell_quantity,
        open=raw_open / divisor,
        high=raw_high / divisor,
        low=raw_low / divisor,
        close=raw_close / divisor,
        last_traded_timestamp_ms=last_traded_timestamp_ms,
        open_interest=open_interest,
        bids=bids,
        asks=asks,
        upper_circuit=upper_circuit / divisor,
        lower_circuit=lower_circuit / divisor,
        week_52_high=week_52_high / divisor,
        week_52_low=week_52_low / divisor,
    )


def decode_ltp(payload: bytes) -> LtpPacket:
    """Decode one Mode 1 (LTP) packet."""
    if len(payload) != LTP_PACKET_SIZE:
        raise PacketDecodeError(f"LTP packet must be {LTP_PACKET_SIZE} bytes, got {len(payload)}")
    mode, exchange_type, raw_token, sequence, timestamp_ms, raw_ltp = _LTP_LAYOUT.unpack_from(
        payload, 0
    )
    if mode != SubscriptionMode.LTP:
        raise PacketDecodeError(f"expected mode {SubscriptionMode.LTP}, got {mode}")
    return LtpPacket(
        token=decode_token(raw_token),
        exchange_type=exchange_type,
        exchange_sequence=sequence,
        exchange_timestamp_ms=timestamp_ms,
        ltp=raw_ltp / price_divisor(exchange_type),
    )


def timestamp_is_plausible(exchange_timestamp_ms: int) -> bool:
    """Sanity-check that the timestamp really is epoch milliseconds.

    A value small enough to be epoch *seconds* would place the tick decades in the past and,
    once converted, would make every freshness check nonsensical. Callers log rather than
    discard: a suspect timestamp is a protocol-version signal, not a reason to drop the price.
    """
    return exchange_timestamp_ms >= _MIN_PLAUSIBLE_EPOCH_MS


def iter_packets(payload: bytes) -> list[bytes]:
    """Frame a buffer that may hold several concatenated packets.

    Each message normally carries exactly one packet, but framing by the mode byte costs
    almost nothing and means a batched frame is decoded rather than rejected wholesale.

    Raises:
        PacketDecodeError: an unknown mode byte, or a trailing partial packet. Both mean the
            stream is no longer aligned, and guessing would emit corrupt prices.
    """
    packets: list[bytes] = []
    offset = 0
    total = len(payload)

    while offset < total:
        mode = payload[offset]
        size = PACKET_SIZES.get(mode)
        if size is None:
            raise PacketDecodeError(
                f"unknown subscription mode {mode} at offset {offset}; stream is misaligned"
            )
        end = offset + size
        if end > total:
            raise PacketDecodeError(
                f"truncated mode-{mode} packet at offset {offset}: "
                f"need {size} bytes, {total - offset} remain"
            )
        packets.append(payload[offset:end])
        offset = end

    return packets
