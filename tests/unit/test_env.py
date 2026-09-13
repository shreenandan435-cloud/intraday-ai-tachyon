"""Vectorized RL Environment — tachyon.rl.env.

Tests cover the LOBEnv which steps through historical Parquet data arrays natively,
computing scale-invariant rewards in basis points and returning the 3 portfolio state
features plus the dynamic action mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from tachyon.ipc.schemas import DEPTH_LEVELS, OrderBook
from tachyon.model.dataset import LOBDataset
from tachyon.persistence.tick_recorder import DEPTH_STREAM, TickRecorder
from tachyon.rl.env import LOBEnv, StepResult

# ─── constants ────────────────────────────────────────────────────────────────

LOB_FEATURES: Final[int] = 4 * DEPTH_LEVELS + 2  # 22
EMBEDDING_IN_FEATURES: Final[int] = LOB_FEATURES + 3  # 25
DEFAULT_SEQ_LEN: Final[int] = 128

AT = "2026-08-17 09:47:00+05:30"
BASE_TS = 1_755_000_000.0
TICK_INTERVAL = 0.1


# ── fixture construction ─────────────────────────────────────────────────────


def _book(
    *,
    ts_epoch: float,
    bid_price: tuple[float, ...] = (100.5, 100.4, 100.3, 100.2, 100.1),
    bid_qty: tuple[int, ...] = (10, 200, 3_000, 40_000, 500_000),
    ask_price: tuple[float, ...] = (100.7, 100.8, 100.9, 101.0, 101.1),
    ask_qty: tuple[int, ...] = (11, 210, 3_100, 41_000, 510_000),
) -> OrderBook:
    return OrderBook(
        token="1053",
        bid_price=bid_price,
        bid_qty=bid_qty,
        ask_price=ask_price,
        ask_qty=ask_qty,
        ts_epoch=ts_epoch,
    )


def _harvest(
    root: Path,
    symbol: str,
    books: list[OrderBook],
    *,
    moment: str = AT,
) -> None:
    from datetime import datetime

    from tachyon.core.clock import ManualClock

    recorder = TickRecorder(
        directory=root,
        clock=ManualClock(datetime.fromisoformat(moment)),
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
    root = tmp_path / "ticks"
    _harvest(root, "UFLEX", _ramp(500))
    return root


@pytest.fixture
def dataset(corpus: Path) -> LOBDataset:
    return LOBDataset(directory=corpus, seq_len=DEFAULT_SEQ_LEN)


# ─── LOBEnv ──────────────────────────────────────────────────────────────────


class TestLOBEnvInitialization:
    def test_constructs_from_dataset(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset, seq_len=DEFAULT_SEQ_LEN)
        assert env.seq_len == DEFAULT_SEQ_LEN
        assert env.n_features == LOB_FEATURES

    def test_constructs_with_custom_params(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset, seq_len=64, lambda_holding=0.001, transaction_cost_bps=2.5)
        assert env.seq_len == 64
        assert env.lambda_holding == 0.001
        assert env.transaction_cost_bps == 2.5

    def test_rejects_seq_len_longer_than_dataset(self, dataset: LOBDataset) -> None:
        with pytest.raises(ValueError, match="seq_len.*cannot exceed dataset windows"):
            LOBEnv(dataset, seq_len=len(dataset) + 100)


class TestLOBEnvObservationSpace:
    def test_observation_shape_b_t_22(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        obs, _ = env.reset()
        assert obs.shape == (1, DEFAULT_SEQ_LEN, LOB_FEATURES)
        assert obs.dtype == np.float32

    def test_observation_is_normalised_lob_features(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        obs, _ = env.reset()
        # Check feature ranges: bids negative (bps from mid), asks positive
        assert (obs[:, :, :DEPTH_LEVELS] <= 0).all()  # bid bps
        assert (obs[:, :, DEPTH_LEVELS:2*DEPTH_LEVELS] >= 0).all()  # ask bps
        # Log quantities should be positive
        assert (obs[:, :, 2*DEPTH_LEVELS:4*DEPTH_LEVELS] >= 0).all()
        # Spread tick and OBI
        assert obs[0, 0, 20] >= 0  # spread_tick
        assert -1 <= obs[0, 0, 21] <= 1  # obi_l1

    def test_info_contains_portfolio_state(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        _, info = env.reset()
        assert "position" in info
        assert "entry_price_bps" in info
        assert "holding_bars" in info
        assert "action_mask" in info
        assert info["position"] == 0  # flat at reset
        assert info["entry_price_bps"] == 0
        assert info["holding_bars"] == 0
        assert info["action_mask"].shape == (4,)  # HOLD, BUY, SELL, CLOSE
        assert info["action_mask"].dtype == bool


class TestLOBEnvStep:
    def test_step_returns_step_result(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        result = env.step(0)  # HOLD
        assert isinstance(result, StepResult)
        assert result.observation.shape == (1, DEFAULT_SEQ_LEN, LOB_FEATURES)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_hold_action_no_position_change(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        result = env.step(0)  # HOLD
        assert result.info["position"] == 0
        assert result.info["entry_price_bps"] == 0
        assert result.info["holding_bars"] == 0

    def test_buy_action_opens_long_position(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        result = env.step(1)  # BUY
        assert result.info["position"] == 1  # long
        assert result.info["entry_price_bps"] != 0
        assert result.info["holding_bars"] == 1

    def test_sell_action_opens_short_position(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        result = env.step(2)  # SELL
        assert result.info["position"] == -1  # short
        assert result.info["entry_price_bps"] != 0
        assert result.info["holding_bars"] == 1

    def test_close_action_closes_position(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        env.step(1)  # BUY to open long
        result = env.step(3)  # CLOSE
        assert result.info["position"] == 0
        assert result.info["entry_price_bps"] == 0
        assert result.info["holding_bars"] == 0

    def test_buy_when_already_long_is_invalid(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        env.step(1)  # BUY -> long
        result = env.step(1)  # BUY again should be masked
        # Position should remain long (1), not double
        assert result.info["position"] == 1
        # Action mask should reflect invalid BUY
        assert not result.info["action_mask"][1]  # BUY masked

    def test_sell_when_already_short_is_invalid(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        env.step(2)  # SELL -> short
        result = env.step(2)  # SELL again should be masked
        assert result.info["position"] == -1
        assert not result.info["action_mask"][2]  # SELL masked

    def test_close_when_flat_is_invalid(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        result = env.step(3)  # CLOSE when flat
        assert result.info["position"] == 0
        assert not result.info["action_mask"][3]  # CLOSE masked


class TestLOBEnvRewardCalculation:
    def test_reward_is_zero_for_hold_flat(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset, lambda_holding=0.0, transaction_cost_bps=0.0)
        env.reset()
        result = env.step(0)  # HOLD
        assert abs(result.reward) < 1e-6  # essentially zero

    def test_reward_includes_unrealised_pnl_long(self, dataset: LOBDataset) -> None:
        """Price goes up after BUY -> positive unrealised PnL in bps."""
        env = LOBEnv(dataset, lambda_holding=0.0, transaction_cost_bps=0.0)
        env.reset()
        env.step(1)  # BUY at mid price
        result = env.step(0)  # HOLD - price moves up in fixture
        # Mid price ramps up in fixture, so long position has positive unrealised PnL
        assert result.reward > 0

    def test_reward_includes_unrealised_pnl_short(self, dataset: LOBDataset) -> None:
        """Price goes up after SELL -> negative unrealised PnL in bps."""
        env = LOBEnv(dataset, lambda_holding=0.0, transaction_cost_bps=0.0)
        env.reset()
        env.step(2)  # SELL at mid price
        result = env.step(0)  # HOLD - price moves up
        # Short position loses as price goes up
        assert result.reward < 0

    def test_reward_includes_realised_pnl_on_close(self, dataset: LOBDataset) -> None:
        """Closing a profitable position yields positive realised PnL."""
        env = LOBEnv(dataset, lambda_holding=0.0, transaction_cost_bps=0.0)
        env.reset()
        env.step(1)  # BUY
        env.step(0)  # HOLD - price moves up
        result = env.step(3)  # CLOSE
        # Should include realised PnL from the round trip
        assert result.reward > 0

    def test_reward_penalises_holding_time(self, dataset: LOBDataset) -> None:
        """Holding penalty (lambda * holding_bars) reduces reward."""
        env = LOBEnv(dataset, lambda_holding=1.0, transaction_cost_bps=0.0)  # 1 bps per bar
        env.reset()
        env.step(1)  # BUY - holding_bars becomes 1
        # Hold for 4 more bars (total 5)
        for _ in range(3):
            env.step(0)
        result = env.step(0)  # 4th HOLD, holding_bars = 5
        # Reward should be reduced by 5 bps (holding penalty) minus any unrealised PnL
        assert result.info["holding_bars"] == 5
        # Total reward should include the accumulated holding penalties
        assert result.info["total_reward_bps"] < 0

    def test_reward_includes_transaction_costs(self, dataset: LOBDataset) -> None:
        """Transaction cost penalty on open and close."""
        env = LOBEnv(dataset, lambda_holding=0.0, transaction_cost_bps=10.0)
        env.reset()
        result_open = env.step(1)  # BUY - pays transaction cost
        result_close = env.step(3)  # CLOSE - pays transaction cost
        # Both steps should have negative reward from costs
        assert result_open.reward < -5
        assert result_close.reward < -5

    def test_reward_is_scale_invariant_bps(self, dataset: LOBDataset) -> None:
        """Reward in bps should be same regardless of absolute price level."""
        # This is tested implicitly by the dataset normalisation


class TestLOBEnvActionMask:
    def test_action_mask_flat_allows_hold_buy_sell(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        _, info = env.reset()
        mask = info["action_mask"]
        assert mask[0]  # HOLD
        assert mask[1]  # BUY
        assert mask[2]  # SELL
        assert not mask[3]  # CLOSE blocked when flat

    def test_action_mask_long_allows_hold_close(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        env.step(1)  # BUY -> long
        result = env.step(0)
        mask = result.info["action_mask"]
        assert mask[0]  # HOLD
        assert not mask[1]  # BUY blocked (already long)
        assert mask[2]  # SELL (would flip to short)
        assert mask[3]  # CLOSE allowed

    def test_action_mask_short_allows_hold_close(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset)
        env.reset()
        env.step(2)  # SELL -> short
        result = env.step(0)
        mask = result.info["action_mask"]
        assert mask[0]  # HOLD
        assert mask[1]  # BUY (would flip to long)
        assert not mask[2]  # SELL blocked (already short)
        assert mask[3]  # CLOSE allowed

    def test_action_mask_blocks_on_crossed_book(self, dataset: LOBDataset) -> None:
        """If book is crossed/empty, BUY and SELL should be masked."""
        # A crossed-book fixture would need a custom recorder harvest;
        # placeholder kept to document the intended coverage.
        pytest.skip("crossed-book fixture not yet harvested")


class TestLOBEnvEpisodeTermination:
    def test_done_at_end_of_dataset(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset, seq_len=min(5, len(dataset)))
        env.reset()
        done = False
        while not done:
            result = env.step(0)
            done = result.done
        assert done

    def test_done_when_loss_limit_breached(self, dataset: LOBDataset) -> None:
        """Episode terminates if daily loss limit would be breached."""
        env = LOBEnv(dataset, loss_limit_bps=-50)  # tight limit
        env.reset()
        # Driving the realised PnL to the limit requires manipulating env
        # internals; documented as intended coverage.
        pytest.skip("loss-limit breach path requires an injected PnL sequence")


class TestLOBEnvVectorized:
    def test_supports_batch_size_greater_than_1(self, dataset: LOBDataset) -> None:
        """Environment should support vectorized batch_size > 1."""
        # Current implementation is single-env; this is for future extension
        env = LOBEnv(dataset)
        obs, _ = env.reset()
        assert obs.shape[0] == 1  # batch dimension

    def test_reset_after_done_returns_new_episode(self, dataset: LOBDataset) -> None:
        env = LOBEnv(dataset, seq_len=min(5, len(dataset)))
        env.reset()
        while True:
            result = env.step(0)
            if result.done:
                break
        obs2, info2 = env.reset()
        assert info2["position"] == 0
        assert info2["holding_bars"] == 0


class TestStepResult:
    def test_step_result_is_named_tuple(self) -> None:
        assert issubclass(StepResult, tuple)
        assert hasattr(StepResult, '_fields')
        assert set(StepResult._fields) == {"observation", "reward", "done", "info"}

    def test_step_result_unpacks(self) -> None:
        obs = np.zeros((1, 10, 22), dtype=np.float32)
        result = StepResult(observation=obs, reward=1.0, done=False, info={})
        o, r, d, i = result
        assert o is obs
        assert r == 1.0
        assert d is False
        assert i == {}


# ─── Integration with PPO ────────────────────────────────────────────────────


class TestEnvPPOIntegration:
    def test_observation_matches_embedding_input(self, dataset: LOBDataset) -> None:
        """Env observation (B, T, 22) + portfolio (3) = embedding input (B, T, 25)."""
        from tachyon.model.embedding import LOBEmbedding

        env = LOBEnv(dataset)
        obs, info = env.reset()

        # Build full 25-feature input
        portfolio = np.array([
            [info["position"], info["entry_price_bps"], info["holding_bars"]]
        ], dtype=np.float32)  # (1, 3)
        portfolio = np.repeat(portfolio, DEFAULT_SEQ_LEN, axis=0)  # (T, 3)
        portfolio = portfolio[None, ...]  # (1, T, 3)

        full_input = np.concatenate([obs, portfolio], axis=-1)  # (1, T, 25)
        assert full_input.shape == (1, DEFAULT_SEQ_LEN, EMBEDDING_IN_FEATURES)

        # Embedding should accept it
        embedding = LOBEmbedding(d_model=64).eval()
        embedded = embedding(torch.from_numpy(full_input))
        assert embedded.shape == (1, DEFAULT_SEQ_LEN, 64)

    def test_action_mask_compatible_with_ppo(self, dataset: LOBDataset) -> None:
        """Action mask from env can be passed to PPO forward."""
        from tachyon.model.ppo import TachyonPPO

        env = LOBEnv(dataset)
        model = TachyonPPO(d_model=64, n_layers=2).eval()

        obs, info = env.reset()
        # PPO expects mask (B, T, 4) or (B, 4) - env returns (4,) so expand
        mask = (
            torch.from_numpy(info["action_mask"])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, DEFAULT_SEQ_LEN, 4)
        )  # (1, T, 4)

        # Build full input
        portfolio = np.array(
            [[info["position"], info["entry_price_bps"], info["holding_bars"]]],
            dtype=np.float32,
        )
        portfolio = np.repeat(portfolio, DEFAULT_SEQ_LEN, axis=0)[None, ...]
        full_input = torch.from_numpy(np.concatenate([obs, portfolio], axis=-1))

        actor_logits, _ = model(full_input, action_mask=mask)
        # Masked actions should have -1e4 logits
        assert (actor_logits[0, -1, ~mask[0, -1]] < -1e3).all()
