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
import socket
import struct
from collections.abc import Iterator
from datetime import datetime

import pytest

from tachyon.core.clock import IST
from tachyon.core.config import Settings, WatchlistItem
from tachyon.ingestion import ws_client as ws_client_module
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
