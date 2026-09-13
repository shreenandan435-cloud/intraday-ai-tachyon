"""Phase 5 ingestion tests — CLAUDE.md §2.

The decoder tests build packets byte by byte with ``struct.pack`` from the *documented*
SmartAPI v2 layout, independently of the decoder's own format strings, then assert every
field comes back exactly. A shared constant would let a wrong offset agree with itself; here
the packet builder and the decoder are written from the spec separately and must meet in the
middle.

What these prove: the unpacking matches the published specification.
What they cannot prove: that the specification matches the live feed. Validate in PAPER
against a real session before trusting prices.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
import types
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Final

import pytest

from tachyon.core.clock import IST
from tachyon.core.config import Settings, WatchlistItem
from tachyon.ingestion import angel_adapter as angel_adapter_module
from tachyon.ingestion import ws_client as ws_client_module
from tachyon.ingestion.angel_adapter import (
    REQUIRED_ENV_VARS,
    AngelOneWebSocketClient,
    FeedSecrets,
    generate_totp,
    load_feed_secrets,
    parse_payload,
)
from tachyon.ingestion.decoder import (
    LTP_PACKET_SIZE,
    QUOTE_PACKET_SIZE,
    SNAP_QUOTE_PACKET_SIZE,
    ExchangeType,
    PacketDecodeError,
    SubscriptionMode,
    decode_ltp,
    decode_snap_quote,
    decode_token,
    iter_packets,
    peek_mode,
    price_divisor,
    timestamp_is_plausible,
)
from tachyon.ingestion.sequence import SequenceManager
from tachyon.ingestion.service import IngestionService, _pad_depth
from tachyon.ingestion.shm_writer import (
    CELL_STRIDE,
    CELL_TABLE_OFFSET,
    PAYLOAD_OFFSET,
    RING_CAPACITY,
    SHMWriter,
)
from tachyon.ingestion.ws_client import (
    ACTION_SUBSCRIBE,
    ACTION_UNSUBSCRIBE,
    FeedAuthenticationError,
    FeedCredentials,
    SmartApiFeedClient,
    TokenSubscription,
    subscriptions_from_settings,
)
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import Heartbeat, OrderBook, Tick
from tachyon.ipc.subscriber import Subscriber, SubscriberRole

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic packet builders — written from the spec, not from decoder.py
# ──────────────────────────────────────────────────────────────────────────────

BID_FLAG = 1
ASK_FLAG = 0


def build_depth_entry(*, is_bid: bool, quantity: int, price_paise: int, orders: int) -> bytes:
    """One 20-byte best-five entry: int16 flag, int64 qty, int64 price, int16 orders."""
    return struct.pack("<hqqh", BID_FLAG if is_bid else ASK_FLAG, quantity, price_paise, orders)


def build_snap_quote(
    *,
    token: str = "2885",
    exchange_type: int = ExchangeType.NSE_CM,
    sequence: int = 12345,
    timestamp_ms: int = 1_786_000_000_000,
    ltp_paise: int = 245_675,
    last_traded_quantity: int = 50,
    atp_paise: int = 245_500,
    volume: int = 4_120_000,
    total_buy_qty: float = 12345.0,
    total_sell_qty: float = 54321.0,
    open_paise: int = 244_000,
    high_paise: int = 246_500,
    low_paise: int = 243_500,
    close_paise: int = 244_250,
    last_traded_ts_ms: int = 1_786_000_000_000,
    open_interest: int = 0,
    oi_change_pct: float = 0.0,
    bids: list[tuple[int, int, int]] | None = None,
    asks: list[tuple[int, int, int]] | None = None,
    upper_circuit_paise: int = 269_000,
    lower_circuit_paise: int = 220_000,
    week_52_high_paise: int = 300_000,
    week_52_low_paise: int = 200_000,
    depth_order: str = "bids_first",
) -> bytes:
    """Assemble a Mode 3 packet exactly as the SmartAPI v2 spec describes it.

    Each ``bids``/``asks`` entry is ``(quantity, price_paise, orders)``.
    """
    if bids is None:
        bids = [
            (100, 239_995, 3),
            (200, 239_990, 5),
            (300, 239_985, 7),
            (400, 239_980, 9),
            (500, 239_975, 11),
        ]
    if asks is None:
        asks = [
            (120, 240_005, 4),
            (220, 240_010, 6),
            (320, 240_015, 8),
            (420, 240_020, 10),
            (520, 240_025, 12),
        ]

    head = struct.pack(
        "<bb25sqqqqqqddqqqqqqd",
        int(SubscriptionMode.SNAP_QUOTE),
        exchange_type,
        token.encode("ascii").ljust(25, b"\x00"),
        sequence,
        timestamp_ms,
        ltp_paise,
        last_traded_quantity,
        atp_paise,
        volume,
        total_buy_qty,
        total_sell_qty,
        open_paise,
        high_paise,
        low_paise,
        close_paise,
        last_traded_ts_ms,
        open_interest,
        oi_change_pct,
    )

    bid_entries = [
        build_depth_entry(is_bid=True, quantity=q, price_paise=p, orders=o) for q, p, o in bids
    ]
    ask_entries = [
        build_depth_entry(is_bid=False, quantity=q, price_paise=p, orders=o) for q, p, o in asks
    ]

    if depth_order == "bids_first":
        depth = b"".join(bid_entries + ask_entries)
    elif depth_order == "asks_first":
        depth = b"".join(ask_entries + bid_entries)
    else:  # interleaved
        depth = b"".join(
            entry for pair in zip(bid_entries, ask_entries, strict=True) for entry in pair
        )

    tail = struct.pack(
        "<qqqq",
        upper_circuit_paise,
        lower_circuit_paise,
        week_52_high_paise,
        week_52_low_paise,
    )
    return head + depth + tail


def build_ltp_packet(
    *,
    token: str = "2885",
    exchange_type: int = ExchangeType.NSE_CM,
    sequence: int = 7,
    timestamp_ms: int = 1_786_000_000_000,
    ltp_paise: int = 245_675,
) -> bytes:
    return struct.pack(
        "<bb25sqqq",
        int(SubscriptionMode.LTP),
        exchange_type,
        token.encode("ascii").ljust(25, b"\x00"),
        sequence,
        timestamp_ms,
        ltp_paise,
    )


# ──────────────────────────────────────────────────────────────────────────────
# decoder.py
# ──────────────────────────────────────────────────────────────────────────────


class TestPacketSizes:
    def test_documented_sizes(self) -> None:
        """These three numbers come from Angel One's protocol documentation."""
        assert LTP_PACKET_SIZE == 51
        assert QUOTE_PACKET_SIZE == 123
        assert SNAP_QUOTE_PACKET_SIZE == 379

    def test_builder_agrees_with_decoder_layout(self) -> None:
        assert len(build_snap_quote()) == SNAP_QUOTE_PACKET_SIZE
        assert len(build_ltp_packet()) == LTP_PACKET_SIZE


class TestSnapQuoteDecoding:
    def test_every_scalar_field(self) -> None:
        packet = build_snap_quote(
            token="2885",
            sequence=98_765,
            timestamp_ms=1_786_284_000_123,
            ltp_paise=245_675,
            last_traded_quantity=50,
            atp_paise=245_500,
            volume=4_120_000,
            total_buy_qty=12_345.0,
            total_sell_qty=54_321.0,
            open_paise=244_000,
            high_paise=246_500,
            low_paise=243_500,
            close_paise=244_250,
            open_interest=999,
        )
        quote = decode_snap_quote(packet)

        assert quote.token == "2885"
        assert quote.exchange_type == ExchangeType.NSE_CM
        assert quote.exchange_sequence == 98_765
        assert quote.exchange_timestamp_ms == 1_786_284_000_123
        assert quote.ltp == pytest.approx(2456.75)
        assert quote.last_traded_quantity == 50
        assert quote.average_traded_price == pytest.approx(2455.00)
        assert quote.volume == 4_120_000
        assert quote.total_buy_quantity == pytest.approx(12_345.0)
        assert quote.total_sell_quantity == pytest.approx(54_321.0)
        assert quote.open == pytest.approx(2440.00)
        assert quote.high == pytest.approx(2465.00)
        assert quote.low == pytest.approx(2435.00)
        assert quote.close == pytest.approx(2442.50)
        assert quote.open_interest == 999
        assert quote.upper_circuit == pytest.approx(2690.00)
        assert quote.lower_circuit == pytest.approx(2200.00)
        assert quote.week_52_high == pytest.approx(3000.00)
        assert quote.week_52_low == pytest.approx(2000.00)

    def test_prices_are_divided_by_one_hundred(self) -> None:
        """Paise to rupees. Getting this wrong scales every stop and target by 100x."""
        quote = decode_snap_quote(build_snap_quote(ltp_paise=1))
        assert quote.ltp == pytest.approx(0.01)

    def test_depth_ladders(self) -> None:
        quote = decode_snap_quote(build_snap_quote())

        assert len(quote.bids) == 5
        assert len(quote.asks) == 5
        assert [level.price for level in quote.bids] == pytest.approx(
            [2399.95, 2399.90, 2399.85, 2399.80, 2399.75]
        )
        assert [level.quantity for level in quote.bids] == [100, 200, 300, 400, 500]
        assert [level.orders for level in quote.bids] == [3, 5, 7, 9, 11]
        assert [level.price for level in quote.asks] == pytest.approx(
            [2400.05, 2400.10, 2400.15, 2400.20, 2400.25]
        )
        assert [level.quantity for level in quote.asks] == [120, 220, 320, 420, 520]

    def test_best_bid_is_below_best_ask(self) -> None:
        quote = decode_snap_quote(build_snap_quote())
        assert quote.best_bid < quote.best_ask

    @pytest.mark.parametrize("order", ["bids_first", "asks_first", "interleaved"])
    def test_depth_classified_by_flag_not_position(self, order: str) -> None:
        """Trusting position would silently invert the book and flip every OBI sign."""
        quote = decode_snap_quote(build_snap_quote(depth_order=order))
        assert [level.quantity for level in quote.bids] == [100, 200, 300, 400, 500]
        assert [level.quantity for level in quote.asks] == [120, 220, 320, 420, 520]
        assert quote.best_bid < quote.best_ask

    def test_token_null_padding_is_stripped(self) -> None:
        """A token carrying trailing NULs never matches the watchlist."""
        quote = decode_snap_quote(build_snap_quote(token="99"))
        assert quote.token == "99"
        assert "\x00" not in quote.token

    def test_long_token(self) -> None:
        quote = decode_snap_quote(build_snap_quote(token="1234567890"))
        assert quote.token == "1234567890"

    def test_timestamp_converts_to_ist(self) -> None:
        moment = datetime(2026, 8, 10, 10, 30, 15, tzinfo=IST)
        packet = build_snap_quote(timestamp_ms=int(moment.timestamp() * 1000))
        quote = decode_snap_quote(packet)

        assert quote.ts_ist == moment
        assert quote.ts_ist.utcoffset() == IST.utcoffset(moment)
        assert quote.ts_epoch == pytest.approx(moment.timestamp())

    def test_currency_segment_uses_its_own_scale(self) -> None:
        quote = decode_snap_quote(
            build_snap_quote(exchange_type=ExchangeType.CDE_FO, ltp_paise=873_500_000)
        )
        assert quote.ltp == pytest.approx(87.35)

    def test_zero_padded_depth_levels(self) -> None:
        """A thin book publishes empty rungs rather than omitting them."""
        quote = decode_snap_quote(
            build_snap_quote(
                bids=[(100, 239_995, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
                asks=[(120, 240_005, 3), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
            )
        )
        assert quote.bids[1].quantity == 0
        assert quote.bids[1].price == 0.0

    def test_wrong_length_is_rejected(self) -> None:
        with pytest.raises(PacketDecodeError, match="379"):
            decode_snap_quote(build_snap_quote()[:-1])

    def test_wrong_mode_is_rejected(self) -> None:
        packet = bytearray(build_snap_quote())
        packet[0] = int(SubscriptionMode.QUOTE)
        with pytest.raises(PacketDecodeError, match="expected mode"):
            decode_snap_quote(bytes(packet))


class TestLtpDecoding:
    def test_round_trip(self) -> None:
        packet = decode_ltp(build_ltp_packet(token="1594", ltp_paise=150_025, sequence=7))
        assert packet.token == "1594"
        assert packet.ltp == pytest.approx(1500.25)
        assert packet.exchange_sequence == 7

    def test_wrong_length_is_rejected(self) -> None:
        with pytest.raises(PacketDecodeError, match="51"):
            decode_ltp(build_ltp_packet() + b"\x00")


class TestHelpers:
    def test_peek_mode(self) -> None:
        assert peek_mode(build_snap_quote()) == SubscriptionMode.SNAP_QUOTE
        assert peek_mode(build_ltp_packet()) == SubscriptionMode.LTP

    def test_peek_mode_on_empty(self) -> None:
        with pytest.raises(PacketDecodeError):
            peek_mode(b"")

    def test_decode_token(self) -> None:
        assert decode_token(b"2885" + b"\x00" * 21) == "2885"
        assert decode_token(b"\x00" * 25) == ""

    def test_price_divisor_defaults(self) -> None:
        assert price_divisor(ExchangeType.NSE_CM) == 100.0
        assert price_divisor(ExchangeType.CDE_FO) == 10_000_000.0
        assert price_divisor(99) == 100.0

    def test_timestamp_plausibility(self) -> None:
        assert timestamp_is_plausible(1_786_000_000_000)
        assert not timestamp_is_plausible(1_786_000_000), "seconds must be flagged, not accepted"


class TestPacketFraming:
    def test_single_packet(self) -> None:
        assert len(iter_packets(build_snap_quote())) == 1

    def test_multiple_concatenated_packets(self) -> None:
        payload = build_snap_quote(token="2885") + build_snap_quote(token="1594")
        packets = iter_packets(payload)
        assert len(packets) == 2
        assert decode_snap_quote(packets[0]).token == "2885"
        assert decode_snap_quote(packets[1]).token == "1594"

    def test_mixed_modes(self) -> None:
        payload = build_ltp_packet() + build_snap_quote()
        packets = iter_packets(payload)
        assert [len(p) for p in packets] == [LTP_PACKET_SIZE, SNAP_QUOTE_PACKET_SIZE]

    def test_truncated_trailing_packet_is_rejected(self) -> None:
        with pytest.raises(PacketDecodeError, match="truncated"):
            iter_packets(build_snap_quote() + build_snap_quote()[:100])

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(PacketDecodeError, match="unknown subscription mode"):
            iter_packets(b"\x09" + b"\x00" * 50)

    def test_empty_payload(self) -> None:
        assert iter_packets(b"") == []


# ──────────────────────────────────────────────────────────────────────────────
# sequence.py
# ──────────────────────────────────────────────────────────────────────────────


class TestSequenceManager:
    def test_starts_at_one_and_increments(self) -> None:
        sequences = SequenceManager()
        assert [sequences.next_for("2885") for _ in range(5)] == [1, 2, 3, 4, 5]

    def test_counters_are_independent_per_token(self) -> None:
        """A gap in one symbol must not implicate another's accumulators."""
        sequences = SequenceManager()
        for _ in range(10):
            sequences.next_for("2885")
        assert sequences.next_for("1594") == 1
        assert sequences.next_for("2885") == 11

    def test_current_before_any_issue(self) -> None:
        assert SequenceManager().current("2885") == 0

    def test_strictly_monotonic_over_many_tokens(self) -> None:
        sequences = SequenceManager()
        tokens = [str(t) for t in range(50)]
        for _ in range(100):
            for token in tokens:
                sequences.next_for(token)
        assert all(sequences.current(token) == 100 for token in tokens)
        assert sequences.issued() == 5000

    def test_reset_restarts_at_one(self) -> None:
        sequences = SequenceManager()
        sequences.next_for("2885")
        sequences.reset("2885")
        assert sequences.next_for("2885") == 1

    def test_reset_all(self) -> None:
        sequences = SequenceManager()
        sequences.next_for("2885")
        sequences.next_for("1594")
        sequences.reset_all()
        assert len(sequences) == 0


# ──────────────────────────────────────────────────────────────────────────────
# ws_client.py
# ──────────────────────────────────────────────────────────────────────────────


def _credentials() -> FeedCredentials:
    return FeedCredentials(api_key="k", client_code="C1", feed_token="tok")


def _client(**kwargs: object) -> SmartApiFeedClient:
    return SmartApiFeedClient(
        _credentials(),
        [TokenSubscription(exchange_type=ExchangeType.NSE_CM, tokens=("2885", "1594"))],
        on_binary=lambda _payload: None,
        **kwargs,  # type: ignore[arg-type]
    )


class _FakeConnection:
    """Records what was sent, then reports the socket closed so ``_session`` unwinds."""

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, message: object) -> None:
        self.sent.append(message)

    async def recv(self) -> bytes:
        raise OSError("closed")

    async def close(self) -> None:
        return None


class _FakeConnect:
    """Stands in for ``websockets.asyncio.client.connect`` as an async context manager."""

    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class TestSubscriptionPayload:
    """The SmartStream v2 subscribe request, field by field.

    The whole request is one shot: it is sent once per connection and the broker does not
    acknowledge it. Every property here fails the same silent way — socket up, pongs flowing,
    zero packets — so each gets its own assertion rather than one composite shape check.
    """

    def test_payload_shape_matches_the_specification(self) -> None:
        import json

        payload = json.loads(_client().subscription_payload())
        assert payload == {
            "correlationID": "tachyon",
            "action": ACTION_SUBSCRIBE,
            "params": {
                "mode": 3,
                "tokenList": [{"exchangeType": 1, "tokens": ["2885", "1594"]}],
            },
        }

    def test_payload_is_str_so_websockets_sends_a_text_frame(self) -> None:
        """``websockets`` picks the opcode from the argument type. A ``bytes`` payload becomes
        a Binary frame, which SmartStream discards without replying."""
        assert isinstance(_client().subscription_payload(), str)

    def test_exchange_type_is_the_plain_integer_one_for_nse(self) -> None:
        import json

        token_list = json.loads(_client().subscription_payload())["params"]["tokenList"]
        exchange_type = token_list[0]["exchangeType"]
        assert exchange_type == 1
        assert type(exchange_type) is int, "must be a JSON number, never a quoted string"

    def test_tokens_are_json_strings_not_numbers(self) -> None:
        raw = _client().subscription_payload()
        assert '"tokens":["2885","1594"]' in raw.replace(" ", "")
        assert "2885," not in raw.replace('"2885"', ""), "tokens must never be bare numbers"

    def test_mode_three_is_requested_for_l2_depth(self) -> None:
        import json

        payload = json.loads(_client().subscription_payload())
        assert payload["params"]["mode"] == int(SubscriptionMode.SNAP_QUOTE) == 3

    def test_unsubscribe_uses_action_zero_and_the_same_shape(self) -> None:
        import json

        payload = json.loads(_client().subscription_payload(ACTION_UNSUBSCRIBE))
        assert payload["action"] == 0
        assert payload["params"]["tokenList"][0]["exchangeType"] == 1

    def test_integer_tokens_are_rejected_at_construction(self) -> None:
        with pytest.raises(TypeError, match="must be strings"):
            TokenSubscription(exchange_type=1, tokens=(2885, 1594))  # type: ignore[arg-type]

    def test_empty_token_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            TokenSubscription(exchange_type=1, tokens=())

    def test_blank_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            TokenSubscription(exchange_type=1, tokens=("2885", "  "))

    def test_non_integer_exchange_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="exchange_type must be an int"):
            TokenSubscription(exchange_type="1", tokens=("2885",))  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_the_frame_actually_sent_is_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression guard for the real defect: the builder returning ``str`` only helps
        if nothing re-encodes it before ``send``. Assert on what reaches the socket."""
        connection = _FakeConnection()
        monkeypatch.setattr(ws_client_module, "connect", lambda *a, **k: _FakeConnect(connection))

        client = _client()
        with pytest.raises(OSError, match="closed"):
            await client._session()

        assert connection.sent, "no subscription request was sent at all"
        request = connection.sent[0]
        assert isinstance(request, str), (
            f"sent {type(request).__name__}; bytes would become a Binary frame, which "
            "SmartStream discards without replying"
        )
        assert json.loads(request)["params"]["tokenList"][0]["exchangeType"] == 1

    def test_watchlist_grouping_produces_exchange_type_one_for_nse(self) -> None:
        """End to end from settings.yaml: the NSE code the payload carries is 1."""
        settings = Settings(
            watchlist=(
                WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),
                WatchlistItem(symbol="INFY", token="1594", exchange="NSE"),
            )
        )
        subscriptions = subscriptions_from_settings(settings)
        assert len(subscriptions) == 1
        assert subscriptions[0].exchange_type == 1
        assert subscriptions[0].tokens == ("2885", "1594")


class TestFeedClient:
    def test_backoff_schedule_is_exponential_then_flat(self) -> None:
        client = _client(backoff_seconds=(1.0, 2.0, 5.0, 10.0, 30.0))
        assert [client.backoff_for(i) for i in range(5)] == [1.0, 2.0, 5.0, 10.0, 30.0]
        assert client.backoff_for(50) == 30.0, "the final delay repeats forever"

    def test_credentials_headers(self) -> None:
        headers = _credentials().headers()
        assert headers["x-api-key"] == "k"
        assert headers["x-client-code"] == "C1"
        assert headers["Authorization"] == "tok"
        assert headers["x-feed-token"] == "tok"

    def test_incomplete_credentials_are_detected(self) -> None:
        assert not FeedCredentials(api_key="", client_code="C", feed_token="t").is_complete()
        assert _credentials().is_complete()

    async def test_run_refuses_incomplete_credentials(self) -> None:
        """A permanent rejection must not be retried against the rate limit."""
        client = SmartApiFeedClient(
            FeedCredentials(api_key="", client_code="", feed_token=""),
            [TokenSubscription(exchange_type=1, tokens=("2885",))],
            on_binary=lambda _payload: None,
        )
        with pytest.raises(FeedAuthenticationError, match="incomplete"):
            await client.run()

    async def test_session_rechecks_credentials_before_every_handshake(self) -> None:
        """The gate lives at dial time, not only at ``run`` entry: no DNS resolution on
        empty tokens, ever — that failure masquerades as [Errno 11001] getaddrinfo."""
        client = SmartApiFeedClient(
            FeedCredentials(api_key="k", client_code="C1", feed_token=""),
            [TokenSubscription(exchange_type=1, tokens=("2885",))],
            on_binary=lambda _payload: None,
        )
        with pytest.raises(FeedAuthenticationError, match="credentials incomplete"):
            await client._session()

    async def test_session_refuses_an_unpopulated_url(self) -> None:
        """An empty URL never reaches getaddrinfo: construction resolves it to the canonical
        endpoint, and the dial-time gate still trips if the URL is corrupted afterwards."""
        client = _client(url="")
        assert client._url == ws_client_module.SMART_STREAM_URL
        client._url = ""
        with pytest.raises(ValueError, match="URL not populated"):
            await client._session()

    def test_requires_at_least_one_subscription(self) -> None:
        with pytest.raises(ValueError, match="TokenSubscription"):
            SmartApiFeedClient(_credentials(), [], on_binary=lambda _p: None)

    def test_handler_failure_does_not_propagate(self) -> None:
        """A downstream bug must not cost us the subscription."""

        def boom(_payload: bytes) -> None:
            raise RuntimeError("decoder exploded")

        client = SmartApiFeedClient(
            _credentials(),
            [TokenSubscription(exchange_type=1, tokens=("2885",))],
            on_binary=boom,
        )
        client._dispatch(b"whatever")
        assert client.stats.handler_errors == 1

    def test_starts_unsubscribed(self) -> None:
        client = _client()
        assert not client.connected
        assert not client.subscribed

    def test_subscriptions_from_settings_groups_by_exchange(self) -> None:
        subscriptions = subscriptions_from_settings(Settings())
        assert subscriptions
        nse = next(s for s in subscriptions if s.exchange_type == ExchangeType.NSE_CM)
        assert "2885" in nse.tokens


# ──────────────────────────────────────────────────────────────────────────────
# service.py
# ──────────────────────────────────────────────────────────────────────────────


class TestDepthPadding:
    def test_pads_short_ladder_to_five(self) -> None:
        from tachyon.ingestion.decoder import DepthLevel

        prices, quantities = _pad_depth((DepthLevel(price=100.0, quantity=5, orders=1),))
        assert prices == (100.0, 0.0, 0.0, 0.0, 0.0)
        assert quantities == (5, 0, 0, 0, 0)

    def test_truncates_long_ladder(self) -> None:
        from tachyon.ingestion.decoder import DepthLevel

        levels = tuple(DepthLevel(price=float(i), quantity=i, orders=1) for i in range(8))
        prices, quantities = _pad_depth(levels)
        assert len(prices) == 5
        assert prices == (0.0, 1.0, 2.0, 3.0, 4.0), "index 0 must remain the touch"
        assert quantities == (0, 1, 2, 3, 4)

    def test_empty_ladder(self) -> None:
        prices, quantities = _pad_depth(())
        assert prices == (0.0,) * 5
        assert quantities == (0,) * 5


class TestIngestionService:
    @staticmethod
    def _service(publisher: Publisher, **kwargs: object) -> IngestionService:
        return IngestionService(_credentials(), publisher=publisher, **kwargs)  # type: ignore[arg-type]

    def test_publishes_tick_and_book_from_one_packet(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        publisher, subscriber = pubsub
        service = self._service(publisher)
        service.handle_binary(build_snap_quote(token="2885", ltp_paise=245_675))

        first = subscriber.recv(timeout_ms=2000)
        second = subscriber.recv(timeout_ms=2000)
        assert first is not None and second is not None

        by_topic = {first.topic: first.message, second.topic: second.message}
        tick = by_topic["TICK.RELIANCE"]
        book = by_topic["DEPTH.RELIANCE"]

        assert isinstance(tick, Tick)
        assert tick.ltp == pytest.approx(2456.75)
        assert tick.seq == 1
        assert isinstance(book, OrderBook)
        assert book.bid_qty == (100, 200, 300, 400, 500)
        assert book.ask_price[0] == pytest.approx(2400.05)

    def test_sequence_increments_across_packets(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        publisher, subscriber = pubsub
        service = self._service(publisher)
        for _ in range(5):
            service.handle_binary(build_snap_quote(token="2885"))

        seqs = []
        for _ in range(10):
            envelope = subscriber.recv(timeout_ms=2000)
            assert envelope is not None
            if isinstance(envelope.message, Tick):
                seqs.append(envelope.message.seq)
        assert seqs == [1, 2, 3, 4, 5]

    def test_unknown_token_is_dropped(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        """Publishing an instrument absent from the watchlist is forbidden (§8.1)."""
        publisher, subscriber = pubsub
        service = self._service(publisher)
        service.handle_binary(build_snap_quote(token="999999"))

        assert subscriber.recv(timeout_ms=300) is None
        assert service.stats.unknown_tokens == 1
        assert service.stats.ticks_published == 0

    def test_corrupt_frame_is_counted_not_raised(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        publisher, _ = pubsub
        service = self._service(publisher)
        service.handle_binary(b"\x09garbage")
        assert service.stats.decode_errors == 1

    def test_batched_frame_publishes_every_packet(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        publisher, subscriber = pubsub
        service = self._service(publisher)
        service.handle_binary(build_snap_quote(token="2885") + build_snap_quote(token="1594"))

        topics = set()
        for _ in range(4):
            envelope = subscriber.recv(timeout_ms=2000)
            assert envelope is not None
            topics.add(envelope.topic)
        assert topics == {"TICK.RELIANCE", "DEPTH.RELIANCE", "TICK.INFY", "DEPTH.INFY"}
        assert service.stats.packets == 2

    async def test_heartbeat_is_suppressed_while_disconnected(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        """The critical safety property: no heartbeat means the Brain's monitor goes stale.

        Heartbeating through an outage would hold the feed 'fresh' while no market data
        flowed, and the strategy would keep trading against a frozen book.
        """
        publisher, subscriber = pubsub
        service = self._service(publisher, heartbeat_interval=0.05)
        assert not service.client.subscribed

        task = asyncio.create_task(service._heartbeat_loop())
        await asyncio.sleep(0.3)
        service._stopping.set()
        await task

        assert service.stats.heartbeats == 0
        assert service.stats.heartbeats_suppressed >= 3
        assert subscriber.recv(timeout_ms=200) is None

    async def test_heartbeat_flows_once_subscribed(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        publisher, subscriber = pubsub
        service = self._service(publisher, heartbeat_interval=0.05)
        service.client._subscribed = True

        task = asyncio.create_task(service._heartbeat_loop())
        await asyncio.sleep(0.3)
        service._stopping.set()
        await task

        assert service.stats.heartbeats >= 3
        envelope = subscriber.recv(timeout_ms=1000)
        assert envelope is not None
        assert isinstance(envelope.message, Heartbeat)
        assert envelope.topic == "FEED.HEARTBEAT"


@pytest.fixture
def pubsub() -> Iterator[tuple[Publisher, Subscriber]]:
    """A connected PUB/SUB pair on a private port.

    The publisher binds before the subscriber connects — reversing that leaves the
    subscription waiting on ZeroMQ's reconnect interval and the first messages are lost to
    the slow-joiner window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"

    publisher = Publisher(endpoint, role="test-ingestor")
    subscriber = Subscriber((), role=SubscriberRole.STRATEGY, endpoint=endpoint)
    Publisher.settle(0.4)
    try:
        yield publisher, subscriber
    finally:
        subscriber.close()
        publisher.close()


# ─── SHMWriter + Angel adapter ────────────────────────────────────────────────
# Live-feed path: SNAP_QUOTE payloads → normalized 8-float slots → the C++
# sidecar's SPSC shared-memory ring. Every test uses an isolated segment name so
# a concurrently running sidecar's ring is never touched.

MOCK_SNAP_QUOTE: Final[dict[str, Any]] = {
    "exchangeSegment": "nse_cm",
    "token": 3045,
    "exchange_timestamp": 1_755_000_000_000,
    "last_price": 100.5,
    "depth": {
        "buy": [
            {"price": 100.5, "quantity": 500, "orders": 3},
            {"price": 100.4, "quantity": 1200, "orders": 7},
            {"price": 100.3, "quantity": 40, "orders": 1},
        ],
        "sell": [
            {"price": 100.7, "quantity": 300, "orders": 2},
            {"price": 100.8, "quantity": 900, "orders": 5},
        ],
    },
}

_SLOT_FLOATS_A: Final[tuple[float, ...]] = (1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0)


def _unique_segment() -> str:
    return f"tachyon_ut_shm_{os.getpid()}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def shm_name() -> Iterator[str]:
    """Isolated segment per test: never touch a live sidecar's ring."""
    yield _unique_segment()


def _binary_snap_quote() -> bytes:
    """Mode-3 packet whose top-two levels mirror MOCK_SNAP_QUOTE (prices in paise).

    The wire always carries a full best-five ladder, so all ten entries are built.
    """
    return build_snap_quote(
        token="3045",
        bids=[
            (500, 10_050, 3),
            (1200, 10_040, 7),
            (400, 10_030, 2),
            (300, 10_020, 1),
            (200, 10_010, 1),
        ],
        asks=[
            (300, 10_070, 2),
            (900, 10_080, 5),
            (800, 10_090, 3),
            (700, 10_100, 1),
            (600, 10_110, 2),
        ],
    )


class TestSnapQuoteParsing:
    def test_dict_payload_normalizes_top_two_levels(self) -> None:
        ticks = parse_payload(MOCK_SNAP_QUOTE)
        assert len(ticks) == 1
        tick = ticks[0]
        assert tick.instrument_id == 3045
        assert tick.timestamp_ns == 1_755_000_000_000 * 1_000_000
        assert tick.floats == (
            100.5,
            500.0,
            100.4,
            1200.0,  # top two bids
            100.7,
            300.0,
            100.8,
            900.0,  # third sell level dropped by design
        )

    def test_v2_best5_payload_keeps_sides_and_scales_paise(self) -> None:
        """SmartWebSocketV2 parsed dict: sides stay labelled, paise become rupees.

        Regression: the adapter once "honoured" the SDK's swapped assignment and read
        ``best_5_sell_data`` as bids — crossing every book (bid > ask) so the router's
        book lookups rejected every slot. The SDK's assignment swap cancels its own
        inverted flag classification, so the delivered fields are correctly labelled.
        """
        payload = {
            "token": 3045,
            "exchange_timestamp": 1_755_000_000_000,
            "best_5_buy_data": [
                {"flag": 1, "quantity": 500, "price": 10_050, "no of orders": 3},
                {"flag": 1, "quantity": 1200, "price": 10_040, "no of orders": 7},
            ],
            "best_5_sell_data": [
                {"flag": 0, "quantity": 300, "price": 10_070, "no of orders": 2},
                {"flag": 0, "quantity": 900, "price": 10_080, "no of orders": 5},
            ],
        }
        tick = parse_payload(payload)[0]
        assert tick.floats == (
            100.50,
            500.0,
            100.40,
            1200.0,  # bids, paise scaled to rupees
            100.70,
            300.0,
            100.80,
            900.0,  # asks, paise scaled to rupees
        )
        assert tick.floats[0] < tick.floats[4], "the book must not be crossed"

    def test_thin_book_pads_with_zeros(self) -> None:
        payload = {
            "token": 99,
            "exchange_timestamp": 1,
            "depth": {"buy": [{"price": 10.0, "quantity": 1}]},
            "depth_sell": [],
        }
        tick = parse_payload(payload)[0]
        assert tick.floats == (10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_feeds_wrapper_payload_expands(self) -> None:
        wrapped = {"data": {"feeds": [MOCK_SNAP_QUOTE, {**MOCK_SNAP_QUOTE, "token": 42}]}}
        ticks = parse_payload(wrapped)
        assert [t.instrument_id for t in ticks] == [3045, 42]

    def test_binary_frame_routes_through_decoder(self) -> None:
        ticks = parse_payload(_binary_snap_quote())
        assert len(ticks) == 1
        assert ticks[0].instrument_id == int("3045")
        assert len(ticks[0].floats) == 8
        # Prices arrive in rupees after the decoder's exchange-scale division.
        assert ticks[0].floats[0] == pytest.approx(100.50)
        assert ticks[0].floats[4] == pytest.approx(100.70)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            parse_payload(12345)  # type: ignore[arg-type]


class TestShmWriter:
    def test_slot_bytes_match_c_layout(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        try:
            writer.write_tick(3045, 111, _SLOT_FLOATS_A)
            cell_base = CELL_TABLE_OFFSET
            stamp = struct.unpack_from("<Q", writer._mv, cell_base)[0]
            payload = struct.unpack_from("<QQ8fI4x", writer._mv, cell_base + PAYLOAD_OFFSET)
            assert stamp == 2, "seqlock stamp must be even (complete) after publish"
            assert payload[0] == 111  # timestamp_ns
            assert payload[1] == 1  # producer sequence
            assert payload[2:10] == _SLOT_FLOATS_A  # lob_state[8], positional
            assert payload[10] == 3045  # flags word carries the token
            reserved_off = cell_base + PAYLOAD_OFFSET + 52
            assert struct.unpack_from("<I", writer._mv, reserved_off)[0] == 0
            tail = struct.unpack_from("<Q", writer._mv, 0)[0]
            assert tail == 1 and writer.tail == 1
        finally:
            writer.close()

    def test_sequence_stamps_increment_lock_free(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        try:
            for step in range(64):
                floats = tuple(float(step + i) for i in range(8))
                writer.write_tick(step + 1, 1_000 + step, floats)

            tail = struct.unpack_from("<Q", writer._mv, 0)[0]
            assert tail == 64 == writer.tail
            for slot in range(64):
                base = CELL_TABLE_OFFSET + slot * CELL_STRIDE
                stamp = struct.unpack_from("<Q", writer._mv, base)[0]
                seq = struct.unpack_from("<Q", writer._mv, base + PAYLOAD_OFFSET + 8)[0]
                # Each cell is visited once below capacity: stamp 0 -> 1 -> 2.
                assert stamp == 2, f"cell {slot} stamp drift"
                assert seq == slot + 1, f"cell {slot} sequence drift"
        finally:
            writer.close()

    def test_ring_wraps_and_reuses_cells(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        try:
            total = RING_CAPACITY + 3
            for step in range(total):
                writer.write_tick(1, step, tuple(float(step) for _ in range(8)))
            assert writer.tail == total
            # Oldest cells were overwritten by the newest three writes.
            for slot, expected_step in ((0, total - 3), (1, total - 2), (2, total - 1)):
                base = CELL_TABLE_OFFSET + slot * CELL_STRIDE
                ts = struct.unpack_from("<Q", writer._mv, base + PAYLOAD_OFFSET)[0]
                assert ts == expected_step
        finally:
            writer.close()

    def test_wrong_float_count_rejected_before_publish(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        try:
            with pytest.raises(struct.error):
                writer.write_tick(1, 1, (0.0,) * 7)
            assert writer.writes == 0
            assert struct.unpack_from("<Q", writer._mv, 0)[0] == 0
        finally:
            writer.close()


class TestCredentialHandling:
    @staticmethod
    def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
        """Scrub every candidate name and neutralize the host's real ``.env``."""
        for var in REQUIRED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("tachyon.ingestion.angel_adapter.load_dotenv", lambda **_: None)
        for var, value in values.items():
            monkeypatch.setenv(var, value)

    def test_missing_env_vars_raise_strictly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, {})
        with pytest.raises(RuntimeError) as excinfo:
            load_feed_secrets()
        message = str(excinfo.value)
        # Both names of each unresolved pair are listed so operators see every option.
        for var in REQUIRED_ENV_VARS:
            assert var in message

    def test_smartapi_primary_names_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "pri-key",
                "SMARTAPI_CLIENT_ID": "pri-client",
                "SMARTAPI_PIN": "1111",
                "SMARTAPI_TOTP_SECRET": "PRI-SECRET",
            },
        )
        secrets = load_feed_secrets()
        assert secrets.api_key == "pri-key"
        assert secrets.client_id == "pri-client"
        assert secrets.pin == "1111"
        assert secrets.totp_secret == "PRI-SECRET"

    def test_smartapi_takes_precedence_over_angel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "primary",
                "ANGEL_API_KEY": "legacy",
                "SMARTAPI_CLIENT_ID": "c1",
                "ANGEL_CLIENT_ID": "c2",
                "SMARTAPI_PIN": "1111",
                "ANGEL_PIN": "2222",
                "SMARTAPI_TOTP_SECRET": "s-pri",
                "ANGEL_TOTP_SECRET": "s-legacy",
            },
        )
        secrets = load_feed_secrets()
        assert secrets.api_key == "primary"
        assert secrets.client_id == "c1"
        assert secrets.pin == "1111"
        assert secrets.totp_secret == "s-pri"

    def test_mixed_conventions_are_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "k-smartapi",
                "ANGEL_CLIENT_ID": "c-angel",
                "SMARTAPI_PIN": "3333",
                "ANGEL_TOTP_SECRET": "s-angel",
            },
        )
        secrets = load_feed_secrets()
        assert secrets.api_key == "k-smartapi"
        assert secrets.client_id == "c-angel"
        assert secrets.pin == "3333"
        assert secrets.totp_secret == "s-angel"

    def test_repo_legacy_client_code_and_password_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``Settings`` has always used CLIENT_CODE/PASSWORD — they must resolve too."""
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "k",
                "SMARTAPI_CLIENT_CODE": "code-legacy",
                "SMARTAPI_PASSWORD": "pw-legacy",
                "SMARTAPI_TOTP_SECRET": "s",
            },
        )
        secrets = load_feed_secrets()
        assert secrets.client_id == "code-legacy"
        assert secrets.pin == "pw-legacy"

    def test_secrets_loaded_and_repr_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "k-123",
                "SMARTAPI_CLIENT_ID": "c-456",
                "SMARTAPI_PIN": "7890",
                "SMARTAPI_TOTP_SECRET": "JBSW",
            },
        )
        secrets = load_feed_secrets()
        assert secrets.api_key == "k-123"
        assert repr(secrets) == "FeedSecrets(api_key=***, client_id=***, pin=***, totp_secret=***)"
        assert "k-123" not in repr(secrets)

    def test_totp_generated_in_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "k",
                "SMARTAPI_CLIENT_ID": "c",
                "SMARTAPI_PIN": "pin",
                "SMARTAPI_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
            },
        )
        code = generate_totp(load_feed_secrets())
        assert len(code) == 6 and code.isdigit()

    def test_totp_normalizes_whitespace_and_missing_padding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secrets pasted from authenticator exports lose padding and gain spaces."""
        import pyotp

        self._set_env(
            monkeypatch,
            {
                "SMARTAPI_API_KEY": "k",
                "SMARTAPI_CLIENT_ID": "c",
                "SMARTAPI_PIN": "pin",
                "SMARTAPI_TOTP_SECRET": " jbsw y3dp ehpk 3px ",
            },
        )
        expected = str(pyotp.TOTP("JBSWY3DPEHPK3PX=").now())
        assert generate_totp(load_feed_secrets()) == expected

    def test_on_data_never_leaks_credentials(
        self, shm_name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed frame logs a stdlib warning without leaking secret material."""
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(
                writer,
                secrets=FeedSecrets(
                    api_key="SECRET-KEY",
                    client_id="CLIENT-ID",
                    pin="PIN-123",
                    totp_secret="TOTP-SECRET",
                ),
            )
            with caplog.at_level(logging.WARNING, logger="tachyon.ingestion.angel_adapter"):
                client.on_data(None, b"\xff\xff not-a-frame")
            assert client.parse_errors == 1
            assert client.ticks_written == 0
            events = [record.getMessage() for record in caplog.records]
            assert "angel_adapter.parse_failed" in events, events
            dumped = json.dumps(
                [str(m) for m in events] + [str(r.exc_text) for r in caplog.records if r.exc_text]
            )
            for needle in ("SECRET-KEY", "CLIENT-ID", "PIN-123", "TOTP-SECRET"):
                assert needle not in dumped
        finally:
            writer.close()


class TestAdapterAuthentication:
    """The adapter's login: cache-first, rate-limit backoff, WS sequencing guard."""

    SECRETS = FeedSecrets(
        api_key="SECRET-KEY", client_id="CLIENT-ID", pin="PIN-123", totp_secret="JBSWY3DPEHPK3PXP"
    )

    @staticmethod
    def _stub_smartapi(
        monkeypatch: pytest.MonkeyPatch, generate_session: Any, websocket: Any = None
    ) -> None:
        module = types.ModuleType("SmartApi")
        module.SmartConnect = generate_session  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "SmartApi", module)
        if websocket is not None:
            # The adapter imports the v2 transport from the submodule, which the real
            # package does not re-export at top level either.
            ws_module = types.ModuleType("SmartApi.smartWebSocketV2")
            ws_module.SmartWebSocketV2 = websocket  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, "SmartApi.smartWebSocketV2", ws_module)

    def test_authenticate_adopts_the_cache_without_a_network_login(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        token_cache.save_session_cache(
            api_key="SECRET-KEY",
            client_code="CLIENT-ID",
            jwt_token="cached-jwt",
            refresh_token="cached-refresh",
            feed_token="cached-feed",
        )

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a cached session must not import SmartApi or hit the network")

        self._stub_smartapi(monkeypatch, explode)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            assert client._authenticate() == ("cached-jwt", "cached-feed")
        finally:
            writer.close()

    def test_authenticate_backs_off_on_the_rate_limit(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        monkeypatch.setattr(token_cache, "LOGIN_RATE_LIMIT_BACKOFFS", (0.0, 0.0))
        calls = 0

        class FakeConnect:
            def __init__(self, api_key: str) -> None:
                pass

            def generateSession(self, *_args: Any) -> dict[str, Any]:  # noqa: N802 - SDK name
                nonlocal calls
                calls += 1
                if calls < 3:
                    return {
                        "status": False,
                        "message": "Access denied because of exceeding access rate",
                    }
                return {
                    "status": True,
                    "data": {"jwtToken": "j", "refreshToken": "r", "feedToken": "f"},
                }

        self._stub_smartapi(monkeypatch, FakeConnect)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            assert client._authenticate() == ("j", "f")
            assert calls == 3
            # ...and the successful login is cached for the next restart.
            cached = token_cache.load_session_cache("SECRET-KEY", "CLIENT-ID")
            assert cached is not None and cached.feed_token == "f"
        finally:
            writer.close()

    def test_authenticate_does_not_retry_a_genuine_rejection(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        monkeypatch.setattr(token_cache, "LOGIN_RATE_LIMIT_BACKOFFS", (0.0,))
        calls = 0

        class FakeConnect:
            def __init__(self, api_key: str) -> None:
                pass

            def generateSession(self, *_args: Any) -> dict[str, Any]:  # noqa: N802 - SDK name
                nonlocal calls
                calls += 1
                return {"status": False, "message": "Invalid PIN"}

        self._stub_smartapi(monkeypatch, FakeConnect)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            with pytest.raises(RuntimeError, match="Invalid PIN"):
                client._authenticate()
            assert calls == 1
        finally:
            writer.close()

    def test_connect_refuses_the_websocket_on_empty_tokens(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sequencing guard: no DNS/handshake until the tokens are confirmed non-empty."""
        constructed = 0

        class FakeWebSocket:
            def __init__(self, *_args: Any) -> None:
                nonlocal constructed
                constructed += 1

        class FakeConnect:
            def __init__(self, api_key: str) -> None:
                pass

            def generateSession(self, *_args: Any) -> dict[str, Any]:  # noqa: N802 - SDK name
                return {
                    "status": True,
                    "data": {"jwtToken": "j", "refreshToken": "r", "feedToken": ""},
                }

        self._stub_smartapi(monkeypatch, FakeConnect, websocket=FakeWebSocket)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            with pytest.raises(RuntimeError, match="no tokens returned"):
                client.connect()
            assert constructed == 0, "the socket must not be built on an empty feed token"
        finally:
            writer.close()

    def test_connect_guard_trips_if_authentication_yields_empty_tokens(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt-and-braces: even a faulty ``_authenticate`` cannot start the handshake."""
        constructed = 0

        class FakeWebSocket:
            def __init__(self, *_args: Any) -> None:
                nonlocal constructed
                constructed += 1

        class UnusedConnect:  # never called: _authenticate is replaced below
            def __init__(self, api_key: str) -> None:
                pass

        self._stub_smartapi(monkeypatch, UnusedConnect, websocket=FakeWebSocket)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            monkeypatch.setattr(client, "_authenticate", lambda: ("jwt", ""))
            with pytest.raises(RuntimeError, match="websocket not started"):
                client.connect()
            assert constructed == 0
        finally:
            writer.close()


class TestFeedUrlConfiguration:
    """The SmartStream URL: canonical v2 default, config override, nothing empty to getaddrinfo."""

    def test_the_canonical_default_is_the_smartstream_v2_endpoint(self) -> None:
        assert ws_client_module.SMART_STREAM_URL == "wss://smartapisocket.angelone.in/smart-stream"

    def test_an_empty_url_falls_back_to_the_canonical_endpoint(self) -> None:
        assert _client(url="")._url == ws_client_module.SMART_STREAM_URL

    def test_a_none_url_falls_back_to_the_canonical_endpoint(self) -> None:
        assert _client(url=None)._url == ws_client_module.SMART_STREAM_URL

    def test_a_whitespace_url_falls_back_to_the_canonical_endpoint(self) -> None:
        assert _client(url="   ")._url == ws_client_module.SMART_STREAM_URL

    def test_an_explicit_override_is_honoured(self) -> None:
        client = _client(url="wss://example.invalid/smart-stream")
        assert client._url == "wss://example.invalid/smart-stream"

    def test_a_non_websocket_scheme_is_rejected_at_construction(self) -> None:
        """A misconfigured URL must fail fast, not surface as a DNS fault mid-handshake."""
        with pytest.raises(ValueError, match="ws:// or wss://"):
            _client(url="https://smartapisocket.angelone.in/smart-stream")

    def test_the_settings_default_leaves_the_client_on_its_canonical_url(self) -> None:
        settings = Settings(
            watchlist=(WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),)
        )
        assert settings.feed.stream_url == ""


class TestAdapterWebSocketV2:
    """SmartWebSocketV2 wiring: all four credentials, subscribe target, resilient run loop."""

    SECRETS = FeedSecrets(
        api_key="SECRET-KEY", client_id="CLIENT-ID", pin="PIN-123", totp_secret="JBSWY3DPEHPK3PXP"
    )

    def test_connect_passes_all_four_credentials_and_retry_tuning(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v2 authenticates in the handshake headers: jwt, api key, client code, feed token."""
        captured: dict[str, Any] = {}

        class FakeV2:
            def __init__(
                self,
                auth_token: str,
                api_key: str,
                client_code: str,
                feed_token: str,
                **kwargs: Any,
            ) -> None:
                captured["args"] = (auth_token, api_key, client_code, feed_token)
                captured["kwargs"] = kwargs

        class FakeConnect:
            def __init__(self, api_key: str) -> None:
                pass

            def generateSession(self, *_args: Any) -> dict[str, Any]:  # noqa: N802 - SDK name
                return {
                    "status": True,
                    "data": {"jwtToken": "jwt-1", "refreshToken": "r", "feedToken": "feed-1"},
                }

        TestAdapterAuthentication._stub_smartapi(monkeypatch, FakeConnect, websocket=FakeV2)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            client.connect()
        finally:
            writer.close()
        assert captured["args"] == ("jwt-1", "SECRET-KEY", "CLIENT-ID", "feed-1")
        kwargs = captured["kwargs"]
        assert kwargs["max_retry_attempt"] == angel_adapter_module.SDK_MAX_RETRY_ATTEMPTS
        assert kwargs["retry_strategy"] == angel_adapter_module.SDK_RETRY_STRATEGY_EXPONENTIAL
        assert kwargs["retry_delay"] == angel_adapter_module.SDK_RETRY_DELAY_SECONDS

    def test_on_open_subscribes_via_the_v2_instance_not_the_websocketapp(
        self, shm_name: str
    ) -> None:
        """The SDK passes the WebSocketApp to on_open; subscribe() lives on the V2 instance.
        Calling it on the WebSocketApp raises AttributeError and the subscription never goes
        out — socket up, pongs flowing, zero ticks."""
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS, tokens=("2885",))
            calls: list[tuple[Any, ...]] = []

            class FakeWS:
                def subscribe(self, correlation_id: str, mode: int, token_list: Any) -> None:
                    calls.append((correlation_id, mode, token_list))

            client._ws = FakeWS()
            client._on_open(object())
            assert len(calls) == 1
            _cid, mode, token_list = calls[0]
            assert mode == 3
            assert token_list == [
                {"actiontype": "subscribe", "exchangeType": 1, "tokens": ["2885"]}
            ]
        finally:
            writer.close()

    def test_run_retries_a_dns_failure_and_recovers(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fatal path from the incident: getaddrinfo failure must not end the feed runner."""
        monkeypatch.setattr(angel_adapter_module, "RECONNECT_BACKOFFS", (0.0, 0.0))
        constructed: list[Any] = []

        class FakeV2:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                constructed.append(self)
                self.closed = 0

            def connect(self) -> None:
                if len(constructed) == 1:
                    raise OSError("[Errno 11001] getaddrinfo failed")
                client.stop()  # a clean close on the second cycle ends the loop

            def close_connection(self) -> None:
                self.closed += 1

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("re-auth must use the patched _authenticate, not the network")

        TestAdapterAuthentication._stub_smartapi(monkeypatch, explode, websocket=FakeV2)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            monkeypatch.setattr(client, "_authenticate", lambda: ("jwt", "feed"))
            client.run()  # must return normally, not raise or hang
        finally:
            writer.close()
        assert len(constructed) == 2, "the failed socket must be rebuilt, not abandoned"
        assert constructed[0].closed == 1, "the failed socket must be closed before the retry"

    def test_stop_interrupts_the_backoff(
        self, shm_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shutdown must not wait out a full reconnect delay."""
        monkeypatch.setattr(angel_adapter_module, "RECONNECT_BACKOFFS", (30.0,))
        constructed: list[Any] = []

        class FakeV2:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                constructed.append(self)

            def connect(self) -> None:
                raise OSError("[Errno 11001] getaddrinfo failed")

            def close_connection(self) -> None:
                pass

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("re-auth must use the patched _authenticate, not the network")

        TestAdapterAuthentication._stub_smartapi(monkeypatch, explode, websocket=FakeV2)
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, secrets=self.SECRETS)
            monkeypatch.setattr(client, "_authenticate", lambda: ("jwt", "feed"))
            thread = threading.Thread(target=client.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5.0
            while not constructed and time.monotonic() < deadline:
                time.sleep(0.02)
            assert constructed, "the first connect attempt never happened"
            client.stop()
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "stop() must interrupt the backoff and end the loop"
        finally:
            writer.close()

    def test_tick_from_dict_reads_sdk_best5_at_face_value(self) -> None:
        """smartapi-python's inverted flag classification and its assignment swap cancel
        out, so the delivered dict is correctly labelled: ``best_5_buy_data`` holds the buy
        side (lower prices) and ``best_5_sell_data`` the sell side (higher prices). Reading
        them at face value — and scaling paise to rupees — keeps the book uncrossed. The
        previous "honour the swap" flip crossed every book (bid > ask), which the router
        then rejected as unusable."""
        from tachyon.ingestion.angel_adapter import tick_from_dict

        buy_level = {"flag": 1, "price": 10_050, "quantity": 5, "no of orders": 1}
        sell_level = {"flag": 0, "price": 10_070, "quantity": 7, "no of orders": 1}
        payload = {
            "token": "2885",
            "exchange_timestamp": 1_700_000_000_000,
            "best_5_buy_data": [buy_level],
            "best_5_sell_data": [sell_level],
        }
        tick = tick_from_dict(payload)
        assert tick.floats[0] == 100.50, "best bid must be the buy side, scaled to rupees"
        assert tick.floats[1] == 5.0
        assert tick.floats[4] == 100.70, "best ask must be the sell side, scaled to rupees"
        assert tick.floats[5] == 7.0
        assert tick.floats[0] < tick.floats[4], "the book must not be crossed"


class TestFetchHistoricalCandlesHeaders:
    """Regression tests for the Angel One historical-candle pre-seed path.

    These pin the two failure modes that produced the live ban:

    1. ``AG8001`` (Invalid Token) from a historical fetch used to invalidate the
       session cache, which triggered a fresh ``generateSession`` on the next
       pre-seed call and produced 4 logins in 4 seconds → rate-limit ban.
    2. The ``Authorization`` header was sometimes emitted with a duplicated
       ``Bearer `` prefix or with the feed token instead of the JWT.

    Each test patches the network boundary (``requests.post``) and
    pre-populates ``client._rest_client`` with a stand-in whose
    ``access_token`` is the JWT we expect to see on the wire. This bypasses
    the lazy SDK import path and keeps the assertions focused on what
    ``fetch_historical_candles`` actually does.
    """

    SECRETS: Final[FeedSecrets] = FeedSecrets(
        api_key="SECRET-KEY",
        client_id="CLIENT-ID",
        pin="0000",
        totp_secret="JBSWY3DPEHPK3PX",
    )

    class _StubClient:
        """Stand-in for ``SmartConnect`` exposing the only attributes
        :meth:`fetch_historical_candles` reads.
        """

        def __init__(self, access_token: str = "JWT-RAW") -> None:
            self.access_token = access_token

    @staticmethod
    def _build_client(
        jwt: str = "JWT-RAW",
        secrets: FeedSecrets | None = None,
    ) -> AngelOneWebSocketClient:
        client = AngelOneWebSocketClient(secrets=secrets or TestFetchHistoricalCandlesHeaders.SECRETS)
        # Inject the stand-in directly — skips the lazy SDK import and the
        # authenticate-in-place path, leaving us only with the part of
        # ``fetch_historical_candles`` we want to exercise.
        client._rest_client = TestFetchHistoricalCandlesHeaders._StubClient(jwt)
        return client

    @staticmethod
    def _capture_post(return_payload: dict[str, Any]) -> tuple[Any, list[dict[str, str]]]:
        captured: list[dict[str, str]] = []

        def _post(*_args: Any, **kwargs: Any) -> Any:
            captured.append(dict(kwargs.get("headers") or {}))
            return types.SimpleNamespace(json=lambda: return_payload)

        return _post, captured

    async def test_authorization_header_is_bearer_with_jwt_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post, captured = self._capture_post({"success": True, "data": []})
        monkeypatch.setattr(angel_adapter_module.requests, "post", post)
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []
        assert len(captured) == 1
        auth = captured[0].get("Authorization", "")
        assert auth == "Bearer JWT-RAW", (
            f"Authorization must be exactly 'Bearer <jwt>', got {auth!r}"
        )
        assert not auth.lower().startswith("bearer bearer"), auth

    async def test_authorization_header_strips_a_pre_existing_bearer_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post, captured = self._capture_post({"success": True, "data": []})
        monkeypatch.setattr(angel_adapter_module.requests, "post", post)
        client = self._build_client(jwt="Bearer JWT-PASTED")
        try:
            await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert len(captured) == 1
        assert captured[0]["Authorization"] == "Bearer JWT-PASTED", (
            "a token that already carries 'Bearer ' must not produce 'Bearer Bearer <jwt>'"
        )

    async def test_angel_one_headers_are_attached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post, captured = self._capture_post({"success": True, "data": []})
        monkeypatch.setattr(angel_adapter_module.requests, "post", post)
        client = self._build_client()
        try:
            await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        headers = captured[0]
        for required in (
            "X-PrivateKey",
            "X-ClientLocalIP",
            "X-ClientPublicIP",
            "X-SourceID",
            "X-MACAddress",
            "X-UserType",
            "Content-Type",
            "Accept",
            "Authorization",
        ):
            assert required in headers, f"missing required header {required!r}"
        assert headers["X-PrivateKey"] == "SECRET-KEY"
        assert headers["X-ClientPublicIP"] == "87.76.191.175"
        assert headers["X-ClientLocalIP"] == "127.0.0.1"
        assert headers["X-UserType"] == "USER"
        assert headers["X-SourceID"] == "WEB"

    async def test_angel_public_ip_env_override_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post, captured = self._capture_post({"success": True, "data": []})
        monkeypatch.setattr(angel_adapter_module.requests, "post", post)
        monkeypatch.setenv("ANGEL_PUBLIC_IP", "203.0.113.42")
        client = self._build_client()
        try:
            await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
            monkeypatch.delenv("ANGEL_PUBLIC_IP", raising=False)
        assert captured[0]["X-ClientPublicIP"] == "203.0.113.42"


class TestFetchHistoricalCandlesNoCacheInvalidation:
    """A pre-seed failure must not invalidate the session cache.

    Background: the historical REST endpoint can return ``AG8001`` (Invalid Token)
    transiently — most often right after a fresh login before the gateway has
    propagated the new JWT. Invalidating the cache forces a fresh ``generateSession``
    on the next call, and four symbols × one invalidation each is four logins in
    four seconds — exactly the pattern that triggered ``Access denied because of
    exceeding access rate`` and the subsequent API-key ban.

    Contract: every pre-seed failure path returns ``[]`` and leaves
    ``token_cache`` untouched. The WebSocket login path owns the session lifecycle.
    """

    SECRETS: Final[FeedSecrets] = FeedSecrets(
        api_key="SECRET-KEY",
        client_id="CLIENT-ID",
        pin="0000",
        totp_secret="JBSWY3DPEHPK3PX",
    )

    @staticmethod
    def _build_client() -> AngelOneWebSocketClient:
        client = AngelOneWebSocketClient(secrets=TestFetchHistoricalCandlesNoCacheInvalidation.SECRETS)
        client._rest_client = TestFetchHistoricalCandlesHeaders._StubClient("JWT-XYZ")
        return client

    @staticmethod
    def _stub_response(payload: Any) -> Any:
        return types.SimpleNamespace(json=lambda: payload)

    @staticmethod
    def _prime_cache(monkeypatch: pytest.MonkeyPatch) -> Any:
        """Pre-populate the session cache and return its path so the test can assert
        the file survives. Cleans up after itself via the ``monkeypatch`` fixture's
        teardown.
        """
        from tachyon.core import token_cache

        with monkeypatch.context() as _:
            pass
        cache_path = token_cache._resolve_path(None)  # type: ignore[attr-defined]
        # Use the default path; monkeypatch teardown will unlink it via ``invalidate``.
        token_cache.save_session_cache(
            api_key="SECRET-KEY",
            client_code="CLIENT-ID",
            jwt_token="JWT-XYZ",
            refresh_token="REFRESH-XYZ",
            feed_token="FEED-XYZ",
        )
        return cache_path

    async def test_token_rejected_response_does_not_invalidate_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        cache_path = self._prime_cache(monkeypatch)
        monkeypatch.setattr(
            token_cache,
            "invalidate_session_cache",
            lambda **_: pytest.fail(
                "cache must not be invalidated on token-rejected pre-seed fetch; "
                "this is the bug that produced the 4-logins-in-4-seconds rate-limit ban"
            ),
        )
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response(
                {"success": False, "errorCode": "AG8001", "message": "Invalid Token"}
            ),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []
        assert cache_path.exists(), "session cache must survive a token-rejected pre-seed"

    async def test_generic_failure_does_not_invalidate_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        cache_path = self._prime_cache(monkeypatch)
        monkeypatch.setattr(
            token_cache,
            "invalidate_session_cache",
            lambda **_: pytest.fail("cache must not be invalidated on pre-seed failure"),
        )
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response(
                {"success": False, "errorCode": "AB9999", "message": "Internal server error"}
            ),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []
        assert cache_path.exists()

    async def test_request_exception_does_not_invalidate_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        cache_path = self._prime_cache(monkeypatch)
        monkeypatch.setattr(
            token_cache,
            "invalidate_session_cache",
            lambda **_: pytest.fail("cache must not be invalidated on transport failure"),
        )

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise OSError("connection reset by peer")

        monkeypatch.setattr(angel_adapter_module.requests, "post", _boom)
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []
        assert cache_path.exists()

    async def test_malformed_response_does_not_invalidate_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tachyon.core import token_cache

        cache_path = self._prime_cache(monkeypatch)
        monkeypatch.setattr(
            token_cache,
            "invalidate_session_cache",
            lambda **_: pytest.fail("cache must not be invalidated on malformed response"),
        )
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response(["unexpected", "shape"]),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []
        assert cache_path.exists()


class TestFetchHistoricalCandlesDefensiveResponseParsing:
    """Defensive parsing around the historical REST response.

    Background: Angel One's historical endpoint can return an empty body
    (HTTP 429 rate-limit response, or 400 with no body for an invalid
    symbol token) or a malformed JSON body. The previous implementation
    called ``response.json()`` unconditionally, which raised
    ``JSONDecodeError: Expecting value: line 1 column 1 (char 0)`` on an
    empty body and aborted the pre-seed loop. These tests pin the
    graceful-degradation contract: empty body / non-200 / malformed JSON
    each return ``[]`` and log a warning, with no exception bubbling.
    """

    SECRETS: Final[FeedSecrets] = FeedSecrets(
        api_key="SECRET-KEY",
        client_id="CLIENT-ID",
        pin="0000",
        totp_secret="JBSWY3DPEHPK3PX",
    )

    @staticmethod
    def _stub_response_with_text(status_code: int, body_text: str) -> Any:
        """Build a stub that exposes ``status_code`` + ``text`` like ``requests.Response``.

        Also exposes a ``.json()`` method so the happy path can parse a JSON body.
        """
        return types.SimpleNamespace(
            status_code=status_code,
            text=body_text,
            json=lambda: json.loads(body_text),
        )

    @staticmethod
    def _build_client() -> AngelOneWebSocketClient:
        client = AngelOneWebSocketClient(secrets=TestFetchHistoricalCandlesDefensiveResponseParsing.SECRETS)
        client._rest_client = TestFetchHistoricalCandlesHeaders._StubClient("JWT-XYZ")
        return client

    async def test_empty_body_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(200, ""),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []

    async def test_whitespace_only_body_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(200, "   \n\t  "),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []

    async def test_rate_limit_status_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(429, ""),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []

    async def test_unauthorized_status_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(401, ""),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []

    async def test_malformed_json_body_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 with non-JSON body (e.g. HTML error page) must NOT raise
        ``JSONDecodeError``; the previous behaviour aborted the pre-seed loop
        on every malformed response.
        """
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(200, "<html>Error</html>"),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert rows == []

    async def test_valid_200_with_bars_still_returns_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: the defensive path must not break the happy path."""
        monkeypatch.setattr(
            angel_adapter_module.requests,
            "post",
            lambda *a, **k: self._stub_response_with_text(
                200,
                '{"success": true, "data": [["2026-09-04T09:15:00+05:30", 100, 101, 99, 100.5, 1000]]}',
            ),
        )
        client = self._build_client()
        try:
            rows = await client.fetch_historical_candles("NSE", "2885", days=1)
        finally:
            client._rest_client = None
        assert len(rows) == 1
        assert rows[0][0] == "2026-09-04T09:15:00+05:30"
