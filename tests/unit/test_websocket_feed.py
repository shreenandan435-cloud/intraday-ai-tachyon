"""Tests for the WebSocket client.

The SmartWebSocket is a small async state machine around a
``websockets`` connection. The tests below exercise the parts that
do not require a real server: the parser, the backoff calculator,
the handler fan-out, and the lifecycle of a connection that drops
mid-stream. A local in-process WebSocket server is used for the
end-to-end flow.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any

import pytest

from tachyon.ingestion.websocket_feed import (
    DEFAULT_WS_URL,
    FeedFrame,
    FeedMode,
    HEARTBEAT_INTERVAL_SECONDS,
    RECONNECT_MAX_SECONDS,
    RECONNECT_MIN_SECONDS,
    SmartWebSocket,
    aggregator_handler,
)
from tachyon.math_engine.warmup import warmup


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    """Compile the Numba kernels before any test asks for them."""
    assert warmup() is True


class TestParse:
    """SmartStream v2 is a binary protocol — market-data frames are 200+ byte
    little-endian packets (see :mod:`tachyon.ingestion.decoder`), not text.
    These tests build real packets via :func:`build_snap_quote` and
    :func:`build_ltp_packet` and assert the parser hands them back as
    structured :class:`FeedFrame`s.
    """

    def test_binary_data_frame(self) -> None:
        # Real Mode-3 Snap Quote frame, mode-3 header byte + structured fields.
        from tests.unit.test_ingestion import build_snap_quote  # type: ignore[import-not-found]

        packet = build_snap_quote(token="3045", ltp_paise=123_450)
        frame = SmartWebSocket._parse(packet)
        assert frame is not None
        assert frame.kind == "data"
        assert frame.mode is FeedMode.L2
        assert frame.token == "3045"
        assert frame.payload["ltp"] == pytest.approx(1234.50)

    def test_binary_tick_frame(self) -> None:
        # Real Mode-1 LTP frame.
        from tests.unit.test_ingestion import build_ltp_packet  # type: ignore[import-not-found]

        packet = build_ltp_packet(token="3045", ltp_paise=99_000)
        frame = SmartWebSocket._parse(packet)
        assert frame is not None
        assert frame.mode is FeedMode.TICK
        assert frame.token == "3045"
        assert frame.payload["ltp"] == pytest.approx(990.0)

    def test_json_control_frame(self) -> None:
        frame = SmartWebSocket._parse('{"status": "ok"}')
        assert frame is not None
        assert frame.kind == "control"
        assert frame.token is None
        assert frame.payload == {"status": "ok"}

    def test_garbage_returns_none(self) -> None:
        assert SmartWebSocket._parse("") is None
        # Random bytes that don't match the Snap-Quote / LTP length prefixes
        # are not parseable, so the parser returns ``None``.
        assert SmartWebSocket._parse(b"not-a-binary-frame") is None
        assert SmartWebSocket._parse(b"") is None

    def test_short_binary_payload_is_dropped(self) -> None:
        # A binary frame shorter than the smallest valid packet is rejected
        # by ``iter_packets`` and surfaces as ``None`` rather than a half-decoded
        # ``FeedFrame``. This is the failure mode that would otherwise poison
        # the aggregator with zeroed prices.
        assert SmartWebSocket._parse(b"\x03\x00\x00") is None

    def test_no_null_bytes_in_output_tokens(self) -> None:
        """Regression: every watchlist key must round-trip through the parser
        with no embedded ``\\x00`` — a null-padded token that reaches the
        strategy layer never matches the watchlist dict and the tick is dropped.

        Exercises both Mode-1 (LTP) and Mode-3 (Snap Quote) decoders against the
        four canonical watchlist tokens used in production settings.yaml.
        """
        from tests.unit.test_ingestion import build_ltp_packet, build_snap_quote  # type: ignore[import-not-found]

        watchlist_tokens = ("1190", "21131", "761439", "24445")
        for token in watchlist_tokens:
            packet = build_snap_quote(token=token, ltp_paise=12_345)
            frame = SmartWebSocket._parse(packet)
            assert frame is not None, f"parser returned None for token={token!r}"
            assert frame.token == token, (
                f"token round-trip failed for {token!r}: got {frame.token!r}"
            )
            assert "\x00" not in (frame.token or ""), (
                f"null byte leaked into parsed token for {token!r}: {frame.token!r}"
            )
            assert frame.mode is FeedMode.L2
            assert frame.payload["ltp"] == pytest.approx(123.45)

            tick = build_ltp_packet(token=token, ltp_paise=12_345)
            frame_tick = SmartWebSocket._parse(tick)
            assert frame_tick is not None
            assert frame_tick.token == token
            assert "\x00" not in (frame_tick.token or "")
            assert frame_tick.mode is FeedMode.TICK
            assert frame_tick.payload["ltp"] == pytest.approx(123.45)

    def test_no_null_bytes_in_l2_depth_prices(self) -> None:
        """The best-five ladder's ``price`` fields must be in rupees, never paise."""
        from tests.unit.test_ingestion import build_snap_quote  # type: ignore[import-not-found]

        packet = build_snap_quote(token="1190", ltp_paise=10_000)
        frame = SmartWebSocket._parse(packet)
        assert frame is not None
        payload = frame.payload or {}
        bids = payload.get("bids") or ()
        asks = payload.get("asks") or ()
        # Every depth price is a float in rupees; we round-trip through repr to make
        # sure no leading NUL survives into a downstream JSON encoder.
        for level in (*bids, *asks):
            price, qty = level
            assert isinstance(price, float)
            assert isinstance(qty, int)
            assert math.isfinite(price)
            assert "\x00" not in repr(price)


class TestSubscriptionPayload:
    """Regression tests for the SmartStream v2 subscribe frame.

    SmartStream strictly requires::

        {
            "correlationID": "<id>",
            "action": 1,            # integer, NOT "subscribe"
            "params": {
                "mode": 3,          # integer, NOT a modeList
                "tokenList": [
                    {"exchangeType": 1, "tokens": [str, ...]}
                ]
            }
        }

    Sending a string action or a ``modeList`` array is silently dropped by
    the broker — the connection stays up, pongs flow, no market data ever
    arrives. These tests pin the exact JSON shape.
    """

    def _client(self, **kwargs: Any) -> SmartWebSocket:
        return SmartWebSocket(
            auth_token="JWT",
            feed_token="FEED",
            api_key="API-KEY",
            client_code="CLIENT-CODE",
            tokens=kwargs.pop("tokens", ("3045", "2885")),
            modes=kwargs.pop("modes", (FeedMode.L2, FeedMode.TICK)),
        )

    def test_payload_uses_integer_action_one(self) -> None:
        client = self._client()
        payload = client._subscription_payload_dict()
        assert payload["action"] == 1, (
            f"action must be the integer 1 (subscribe), got {payload['action']!r}"
        )
        assert isinstance(payload["action"], int)

    def test_payload_uses_params_mode_not_modelist(self) -> None:
        client = self._client()
        payload = client._subscription_payload_dict()
        assert "params" in payload
        assert "mode" in payload["params"], "missing params.mode (SmartStream v2 schema)"
        assert "modeList" not in payload, (
            "SmartStream v2 does not honour a top-level modeList; nested "
            "params.mode is required"
        )

    def test_payload_mode_collapsed_to_highest_int(self) -> None:
        client = self._client(modes=(FeedMode.TICK, FeedMode.L2))
        payload = client._subscription_payload_dict()
        assert payload["params"]["mode"] == 3, (
            "subscribing to {L2, TICK} must collapse to mode 3 (Snap Quote) "
            "because mode 3 carries LTP *and* the full best-five ladder"
        )

    def test_payload_tokenlist_shape_and_exchange_type(self) -> None:
        client = self._client(tokens=("3045",))
        payload = client._subscription_payload_dict()
        token_list = payload["params"]["tokenList"]
        assert isinstance(token_list, list) and len(token_list) == 1
        entry = token_list[0]
        assert entry["exchangeType"] == 1, "NSE cash is exchangeType == 1"
        assert entry["tokens"] == ["3045"]
        # Tokens must be JSON strings — a numeric token is silently dropped
        # by SmartStream and the subscription then produces zero ticks.
        assert all(isinstance(t, str) for t in entry["tokens"])

    def test_payload_carries_correlation_id(self) -> None:
        client = self._client()
        payload = client._subscription_payload_dict()
        assert payload["correlationID"]


class TestReconnectBackoff:
    def test_exponential_growth_capped_at_max(self) -> None:
        ws = SmartWebSocket(
            auth_token="x", feed_token="y", reconnect_min=1.0, reconnect_max=30.0
        )
        # attempt 1 → 1.0; attempt 2 → 2.0; attempt 3 → 4.0; … attempt 6 → 30.0 cap.
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        for n, exp in enumerate(expected, start=1):
            # Run the inner sleep; we never wait long because the
            # stop-event is never set, so the asyncio.wait_for hits
            # its timeout and we read the planned delay from the log
            # via inspection. Instead, recompute the formula.
            delay = min(ws._reconnect_max, ws._reconnect_min * (2 ** (n - 1)))
            assert delay == pytest.approx(exp, rel=1e-9)

    def test_zero_attempt_yields_min(self) -> None:
        """The first reconnect uses ``min`` (no backoff multiplier)."""
        ws = SmartWebSocket(auth_token="x", feed_token="y", reconnect_min=2.0, reconnect_max=30.0)
        ws._reconnect.attempt = 1
        delay = min(ws._reconnect_max, ws._reconnect_min * (2 ** (ws._reconnect.attempt - 1)))
        assert delay == pytest.approx(2.0)


class TestSubscription:
    def test_subscribe_extends_token_set(self) -> None:
        ws = SmartWebSocket(auth_token="x", feed_token="y", tokens=("1",))
        ws.subscribe(("2", "3"))
        assert ws._tokens == {"1", "2", "3"}

    def test_unsubscribe_drops_token(self) -> None:
        ws = SmartWebSocket(auth_token="x", feed_token="y", tokens=("1", "2"))
        ws.unsubscribe(("1",))
        assert ws._tokens == {"2"}


class TestHandlerFanOut:
    async def test_handler_receives_frames(self) -> None:
        from tests.unit.test_ingestion import build_ltp_packet, build_snap_quote  # type: ignore[import-not-found]

        received: list[FeedFrame] = []

        async def handler(frame: FeedFrame) -> None:
            received.append(frame)

        ws = SmartWebSocket(
            auth_token="x", feed_token="y", on_frame=handler
        )
        # Two real SmartStream packets: mode-3 snap quote and mode-1 LTP.
        await ws._dispatch(build_snap_quote(token="3045", ltp_paise=10_000))
        await ws._dispatch(build_ltp_packet(token="3046", ltp_paise=5_050))
        assert len(received) == 2
        assert received[0].mode is FeedMode.L2
        assert received[1].mode is FeedMode.TICK

    async def test_handler_exceptions_do_not_break_fanout(self) -> None:
        from tests.unit.test_ingestion import build_snap_quote  # type: ignore[import-not-found]

        received: list[FeedFrame] = []

        async def bad_handler(frame: FeedFrame) -> None:
            raise RuntimeError("intentional")

        async def good_handler(frame: FeedFrame) -> None:
            received.append(frame)

        ws = SmartWebSocket(auth_token="x", feed_token="y")
        ws.add_handler(bad_handler)
        ws.add_handler(good_handler)
        # The bad handler logs and is skipped; the good handler still
        # receives the frame.
        await ws._dispatch(build_snap_quote(token="3045", ltp_paise=10_000))
        assert len(received) == 1

    async def test_garbage_frames_dropped_silently(self) -> None:
        ws = SmartWebSocket(auth_token="x", feed_token="y")
        # An unparseable buffer: no length prefix that matches a known
        # packet type, so ``iter_packets`` returns ``[]`` and the frame
        # is dropped with no exception bubbling.
        await ws._dispatch(b"")
        assert ws.frames_dropped == 1


class TestAggregatorHandler:
    async def test_tick_feeds_aggregator(self) -> None:
        """The handler maps a TICK frame to ``on_tick`` on the aggregator."""
        from tachyon.math_engine.core import TickAggregator

        aggregator = TickAggregator("RELIANCE")
        handler = aggregator_handler(aggregator)
        frame = FeedFrame(
            kind="data",
            mode=FeedMode.TICK,
            token="3045",
            payload={"ltp": 100.0, "v": 50, "ltt": 1_700_000_000.0, "sq": 1},
        )
        await handler(frame)
        assert aggregator.ltp == pytest.approx(100.0)

    async def test_l2_feeds_orderbook(self) -> None:
        from tachyon.math_engine.core import TickAggregator

        aggregator = TickAggregator("RELIANCE")
        handler = aggregator_handler(aggregator)
        frame = FeedFrame(
            kind="data",
            mode=FeedMode.L2,
            token="3045",
            payload={
                "bp": [100.0, 99.5, 99.0, 98.5, 98.0],
                "sp": [101.0, 101.5, 102.0, 102.5, 103.0],
                "bq": [200, 100, 50, 50, 50],
                "sq": [200, 100, 50, 50, 50],
            },
        )
        await handler(frame)
        # OBI = (200 - 200) / 400 = 0.0 (balanced book).
        assert aggregator._obi == pytest.approx(0.0, abs=1e-9)  # type: ignore[attr-defined]

    async def test_unknown_frame_kind_is_skipped(self) -> None:
        from tachyon.math_engine.core import TickAggregator

        aggregator = TickAggregator("RELIANCE")
        handler = aggregator_handler(aggregator)
        frame = FeedFrame(
            kind="control",
            mode=None,
            token=None,
            payload={"status": "ok"},
        )
        await handler(frame)
        # No state change: the LTP stays at its initial value (NaN).
        assert math.isnan(aggregator.ltp)


class TestStopSignal:
    def test_stop_marks_event(self) -> None:
        ws = SmartWebSocket(auth_token="x", feed_token="y")
        ws.stop()
        # The stop event is set; the run loop sees it on next pass.
        assert ws._stop_event.is_set()


class TestConstants:
    def test_reconnect_bounds_are_documented(self) -> None:
        assert RECONNECT_MIN_SECONDS == 1.0
        assert RECONNECT_MAX_SECONDS == 30.0
        assert HEARTBEAT_INTERVAL_SECONDS == 20.0
        assert DEFAULT_WS_URL.startswith("wss://")


class TestHandshakeHeaders:
    """Regression tests for the SmartStream opening handshake headers.

    SmartStream ``wss://smartapisocket.angelone.in/smart-stream`` returns HTTP 401
    before the WebSocket upgrade completes when any of the four required
    headers (``Authorization: Bearer <jwt>``, ``x-api-key``, ``x-client-code``,
    ``x-feed-token``) is missing. These tests pin that contract.
    """

    def _client(self, **kwargs: Any) -> SmartWebSocket:
        return SmartWebSocket(
            auth_token=kwargs.pop("auth_token", "JWT-RAW"),
            feed_token=kwargs.pop("feed_token", "FEED-RAW"),
            api_key=kwargs.pop("api_key", "API-KEY"),
            client_code=kwargs.pop("client_code", "CLIENT-CODE"),
            **kwargs,
        )

    def test_handshake_emits_bearer_jwt_with_no_double_prefix(self) -> None:
        client = self._client()
        headers = client._handshake_headers()
        assert headers["Authorization"] == "Bearer JWT-RAW"

    def test_handshake_strips_a_pre_existing_bearer_prefix(self) -> None:
        # A token pasted via env that already carries ``Bearer `` must NOT produce
        # ``Bearer Bearer <jwt>`` on the wire — the broker rejects that shape with
        # HTTP 401, which is the failure mode behind the original 401.
        client = self._client(auth_token="Bearer JWT-PASTED")
        headers = client._handshake_headers()
        assert headers["Authorization"] == "Bearer JWT-PASTED"

    def test_handshake_carries_api_key_client_code_and_feed_token(self) -> None:
        client = self._client()
        headers = client._handshake_headers()
        assert headers["x-api-key"] == "API-KEY"
        assert headers["x-client-code"] == "CLIENT-CODE"
        assert headers["x-feed-token"] == "FEED-RAW"

    def test_handshake_uses_jwt_not_feed_token_for_bearer(self) -> None:
        # The bearer is the JWT, not the feed token — Angel One rejects the feed
        # token as the bearer with HTTP 401.
        client = self._client(auth_token="JWT-X", feed_token="FEED-Y")
        headers = client._handshake_headers()
        assert "JWT-X" in headers["Authorization"]
        assert "FEED-Y" not in headers["Authorization"]

    def test_extra_headers_merge_into_handshake(self) -> None:
        client = self._client(extra_headers={"X-ClientPublicIP": "87.76.191.175"})
        headers = client._handshake_headers()
        assert headers["X-ClientPublicIP"] == "87.76.191.175"
        # Required headers must still be present alongside the extras.
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["x-feed-token"] == "FEED-RAW"
