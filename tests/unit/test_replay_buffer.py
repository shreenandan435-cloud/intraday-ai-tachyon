"""AsyncReplayBuffer + reward shaping — unit tests for the Track 2 RL data path.

Covers the delivery contract (zero dropped transitions under high-throughput,
multi-producer push), ring eviction, batch shape/contiguity/device semantics of
sample(), and the Differential Sharpe Ratio reward math.
"""

from __future__ import annotations

import threading

import pytest
import torch

from tachyon.rl.replay_buffer import AsyncReplayBuffer, ReplayBatch, Transition
from tachyon.rl.reward import DifferentialSharpeRatio, calculate_reward


def _make_transition(seq: int) -> Transition:
    """Deterministic transition whose payload encodes ``seq`` for integrity checks."""
    state = [float(seq % 1000) + i * 1e-3 for i in range(4)]
    return Transition(
        state=state,
        action=seq % 4,
        reward=float(seq) * 0.5,
        next_state=[v + 1.0 for v in state],
        done=seq % 100 == 99,
    )


# ─── Delivery: no dropped transitions ────────────────────────────────────────


class TestNoDroppedTransitions:
    def test_single_producer_high_throughput(self) -> None:
        total = 20_000
        buf = AsyncReplayBuffer(capacity=32_768, state_dim=4, pin_memory=False)
        try:
            for seq in range(total):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()

            assert buf.writer_error is None
            assert buf.pushed == total
            assert len(buf) == total
            assert buf.dropped == 0
            # Rows [0, total) were filled in push order; the rest of the ring is untouched.
            assert torch.equal(
                buf.states[:total, 0],
                torch.arange(total, dtype=torch.float32).fmod(1000),
            )
            assert torch.equal(buf.rewards[:total], torch.arange(total, dtype=torch.float32) * 0.5)
        finally:
            buf.close()

    def test_multi_producer_no_loss_and_full_integrity(self) -> None:
        total_per_producer = 12_500
        producers = 4
        total = producers * total_per_producer
        buf = AsyncReplayBuffer(capacity=65_536, state_dim=4, pin_memory=False, seed=1234)
        try:

            def produce(offset: int) -> None:
                for seq in range(offset * total_per_producer, (offset + 1) * total_per_producer):
                    t = _make_transition(seq)
                    buf.push(t.state, t.action, t.reward, t.next_state, t.done)

            threads = [threading.Thread(target=produce, args=(i,)) for i in range(producers)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            buf.close()

            assert buf.writer_error is None
            assert buf.pushed == total
            assert buf.dropped == 0
            assert len(buf) == total

            # Every encoded sequence landed exactly once — sort the first column and
            # compare against the expected multiset {seq % 1000}.
            got = buf.states[:total, 0].sort().values
            want = torch.arange(total, dtype=torch.float32).fmod(1000).sort().values
            assert torch.equal(got, want)
        finally:
            buf.close()


# ── Ring eviction ───────────────────────────────────────────────────────────


class TestRingEviction:
    def test_oldest_evicted_once_full(self) -> None:
        capacity = 256
        overrun = 64
        buf = AsyncReplayBuffer(capacity=capacity, state_dim=4, pin_memory=False)
        try:
            for seq in range(capacity + overrun):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()

            assert len(buf) == capacity
            # Survivors are sequences [overrun, capacity + overrun); rewards are exact.
            assert buf.rewards.min().item() == pytest.approx(overrun * 0.5)
            assert buf.rewards.max().item() == pytest.approx((capacity + overrun - 1) * 0.5)
        finally:
            buf.close()


# ─── Sampling contract ────────────────────────────────────────────────────────


class TestSample:
    def test_shapes_dtypes_contiguity(self) -> None:
        n = 512
        buf = AsyncReplayBuffer(capacity=1024, state_dim=4, pin_memory=False, seed=42)
        try:
            for seq in range(n):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()

            batch = buf.sample(128, device="cpu")
            assert isinstance(batch, ReplayBatch)
            expected = {
                "states": ((128, 4), torch.float32),
                "actions": ((128, 1), torch.int64),
                "rewards": ((128,), torch.float32),
                "next_states": ((128, 4), torch.float32),
                "dones": ((128,), torch.bool),
            }
            for name, (shape, dtype) in expected.items():
                tensor = getattr(batch, name)
                assert tensor.shape == shape, name
                assert tensor.dtype == dtype, name
                assert tensor.device.type == "cpu", name
                assert tensor.is_contiguous(), name

            dones_frac = batch.dones.float().mean().item()
            assert 0.0 <= dones_frac <= 0.05
        finally:
            buf.close()

    def test_sampled_rows_match_pool(self) -> None:
        n = 300
        buf = AsyncReplayBuffer(capacity=1024, state_dim=4, pin_memory=False, seed=7)
        try:
            for seq in range(n):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()

            batch = buf.sample(64, device="cpu")
            for row in range(64):
                idx = int(batch.rewards[row].item() / 0.5)
                assert torch.equal(batch.states[row], buf.states[idx])
                assert torch.equal(batch.actions[row], buf.actions[idx])
                assert bool(batch.dones[row]) == (idx % 100 == 99)
        finally:
            buf.close()

    def test_non_blocking_transfer_accepted(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable on this build; non_blocking H2D needs an accelerator")
        buf = AsyncReplayBuffer(capacity=64, state_dim=4, pin_memory=True)
        try:
            for seq in range(64):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()
            assert buf.is_pinned
            batch = buf.sample(16, device="cuda")
            assert batch.states.device.type == "cuda"
            assert batch.states.is_pinned() is False
        finally:
            buf.close()

    def test_insufficient_samples_raises(self) -> None:
        buf = AsyncReplayBuffer(capacity=16, state_dim=4, pin_memory=False)
        try:
            with pytest.raises(ValueError, match="batch_size"):
                buf.sample(8)
            for seq in range(4):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()
            with pytest.raises(ValueError, match="holds only 4"):
                buf.sample(5)
        finally:
            buf.close()

    def test_default_device_resolution(self) -> None:
        buf = AsyncReplayBuffer(capacity=8, state_dim=4, pin_memory=False)
        try:
            for seq in range(8):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
            buf.close()

            batch = buf.sample(4)
            expected = "cuda" if torch.cuda.is_available() else "cpu"
            assert batch.states.device.type == expected
        finally:
            buf.close()


# ─── Lifecycle / malformed input ──────────────────────────────────────────────


class TestLifecycle:
    def test_context_manager_drains(self) -> None:
        with AsyncReplayBuffer(capacity=64, state_dim=4, pin_memory=False) as buf:
            for seq in range(30):
                t = _make_transition(seq)
                buf.push(t.state, t.action, t.reward, t.next_state, t.done)
        assert len(buf) == 30

    def test_malformed_state_counted_as_dropped(self) -> None:
        buf = AsyncReplayBuffer(capacity=16, state_dim=4, pin_memory=False)
        try:
            good = _make_transition(0)
            buf.push(good.state, 0, 0.0, good.next_state)
            buf.push([1.0, 2.0], 0, 0.0, [1.0, 2.0, 3.0, 4.0])  # wrong state_dim
            buf.close()
            assert buf.dropped == 1
            assert buf.pushed == 2
            assert len(buf) == 1
            assert isinstance(buf.writer_error, ValueError)
        finally:
            buf.close()

    def test_invalid_constructor_args(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            AsyncReplayBuffer(capacity=0, state_dim=4)
        with pytest.raises(ValueError, match="state_dim"):
            AsyncReplayBuffer(capacity=8, state_dim=0)


# ─── Reward shaping ───────────────────────────────────────────────────────────


class TestCalculateReward:
    def test_matches_closed_form(self) -> None:
        eta, lam = 0.05, 1e-4
        got = calculate_reward(pnl=250.0, inventory_size=3.0, rolling_variance=4e-4)
        want = eta * 250.0 / (4e-4**0.5) - lam * 9.0
        assert got == pytest.approx(want)

    def test_inventory_penalty_is_quadratic_and_symmetric(self) -> None:
        flat = calculate_reward(0.0, 0.0, 1e-4)
        long5 = calculate_reward(0.0, 5.0, 1e-4)
        short5 = calculate_reward(0.0, -5.0, 1e-4)
        long10 = calculate_reward(0.0, 10.0, 1e-4)
        assert flat == 0.0
        assert long5 == short5
        assert long10 == pytest.approx(4.0 * long5)

    def test_volatility_scaling_damps_reward(self) -> None:
        calm = calculate_reward(100.0, 0.0, 1e-6)
        volatile = calculate_reward(100.0, 0.0, 1e-2)
        assert abs(calm) > abs(volatile)

    def test_zero_variance_clamped_not_infinite(self) -> None:
        r = calculate_reward(50.0, 1.0, 0.0)
        assert r != float("inf") and r == r  # finite, not NaN

    def test_positive_pnl_rewards_more_than_negative_at_equal_risk(self) -> None:
        gain = calculate_reward(+10.0, 0.0, 1e-4)
        loss = calculate_reward(-10.0, 0.0, 1e-4)
        assert gain > 0.0 > loss


class TestDifferentialSharpeRatio:
    def test_recursion_tracks_sharpe_direction(self) -> None:
        dsr = DifferentialSharpeRatio()
        steady_gains = [dsr.update(x) for x in [1.0] * 200]
        assert steady_gains[-1] > 0.0

    def test_alternating_returns_yield_stable_small_dsr(self) -> None:
        dsr = DifferentialSharpeRatio()
        first = dsr.update(0.5)
        last = first
        for k in range(1, 1000):
            last = dsr.update(0.5 if k % 2 == 0 else -0.5)
        # Alternating +/- returns keep A_t near zero and B_t near E[R^2]; the DSR
        # increments stay bounded well below the cold-start magnitude.
        assert abs(first) > abs(last)
        assert last == last  # finite, not NaN

    def test_invalid_eta_rejected(self) -> None:
        with pytest.raises(ValueError, match="eta_a"):
            DifferentialSharpeRatio(eta_a=0.0)
