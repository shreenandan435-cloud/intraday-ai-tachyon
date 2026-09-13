"""Recurrent PPO integration — GTrXL x AsyncReplayBuffer end-to-end.

Drives the full training path with production components: transitions (raw LOB
states, discrete actions, DSR-shaped rewards) flow through the AsyncReplayBuffer's
writer thread, are chunked into sequences, burn in the GTrXL memory cache, and
come back out as clipped-surrogate gradient updates. The assertions target the
two failure modes that matter for recurrent PPO: silent shape mismatches across
the temporal reshape, and gradients that never reach the network weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tachyon.rl.models.gtrxl import TachyonActorCriticGTrXL
from tachyon.rl.ppo import TachyonPPOTrainer
from tachyon.rl.replay_buffer import AsyncReplayBuffer
from tachyon.rl.reward import calculate_reward

OBS_DIM = 25
NUM_ACTIONS = 4


# ── fixtures ────────────────────────────────────────────────────────────────


def _build_trainer() -> tuple[TachyonActorCriticGTrXL, TachyonPPOTrainer, dict[str, torch.Tensor]]:
    """Small deterministic GTrXL + trainer; returns snapshot of initial weights."""
    torch.manual_seed(42)
    model = TachyonActorCriticGTrXL(
        obs_dim=OBS_DIM,
        d_model=32,
        n_heads=4,
        n_layers=2,
        mem_len=8,
        num_actions=NUM_ACTIONS,
        mlp_ratio=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = TachyonPPOTrainer(
        model,
        optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        seq_len=16,
        burn_in_steps=4,
        epochs=2,
        minibatch_sequences=2,
    )
    snapshot = {name: param.detach().clone() for name, param in model.named_parameters()}
    return model, trainer, snapshot


def _populate_buffer(buffer: AsyncReplayBuffer, count: int, seed: int) -> None:
    """Push simulated LOB transitions whose rewards come from calculate_reward."""
    rng = np.random.default_rng(seed)
    variance = 1e-4  # rolling per-bar variance estimate feeding the DSR shaping
    for step in range(count):
        state = rng.standard_normal(OBS_DIM).astype(np.float32)
        action = int(rng.integers(0, NUM_ACTIONS))
        pnl = float(rng.standard_normal())
        inventory = float(action - 1)  # -1, 0, 1 signed position proxy
        reward = calculate_reward(
            pnl=pnl,
            inventory_size=inventory,
            rolling_variance=variance,
            inventory_lambda=1e-4,
        )
        next_state = state + rng.standard_normal(OBS_DIM).astype(np.float32) * 0.01
        buffer.push(state, action, reward, next_state, done=(step % 97 == 96))


@pytest.fixture
def populated_buffer() -> AsyncReplayBuffer:
    """Buffer with 1024 fully-drained DSR-shaped transitions ready for sampling."""
    buffer = AsyncReplayBuffer(capacity=4096, state_dim=OBS_DIM, pin_memory=False)
    _populate_buffer(buffer, 1024, seed=1234)
    # Draining is deterministic: writes are async by design, so join the writer
    # thread before any trainer.update() samples from the pool.
    buffer.close()
    assert len(buffer) == 1024
    return buffer


# ─── update contract ─────────────────────────────────────────────────────────


class TestUpdateContract:
    def test_losses_computed_and_finite(self, populated_buffer: AsyncReplayBuffer) -> None:
        _, trainer, _ = _build_trainer()
        stats = trainer.update(populated_buffer, batch_size=64)

        for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"):
            assert key in stats, f"missing stat {key}"
            value = stats[key]
            assert isinstance(value, float)
            assert np.isfinite(value), f"{key} is not finite: {value}"

        # Entropy of a 4-way distribution starts near log(4); a wildly different
        # value means the actor logits path is broken.
        assert -0.5 < stats["entropy"] < np.log(4.0) + 0.5

    def test_weights_receive_non_zero_gradients(self, populated_buffer: AsyncReplayBuffer) -> None:
        model, trainer, _ = _build_trainer()
        trainer.update(populated_buffer, batch_size=64)

        checked = {
            "encoder.0.weight": None,
            "actor_head.weight": None,
            "critic_head.weight": None,
            "blocks.0.attn.w_q.weight": None,
            "blocks.1.ffn_gate.x_proj.weight": None,
        }
        for name, param in model.named_parameters():
            if name in checked:
                assert param.grad is not None, f"no gradient reached {name}"
                assert float(param.grad.abs().sum()) > 0.0, f"zero gradient at {name}"
                checked[name] = param.grad

        assert all(grad is not None for grad in checked.values())

    def test_update_changes_model_weights(self, populated_buffer: AsyncReplayBuffer) -> None:
        model, trainer, snapshot = _build_trainer()
        trainer.update(populated_buffer, batch_size=64)

        changed = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.detach(), snapshot[name]):
                changed += 1
        assert changed > 10, f"only {changed} parameter tensors moved — update is a no-op"


class TestGradientThroughTemporalBoundary:
    def test_burn_in_cache_feeds_graduated_forward(self) -> None:
        """Gradients must reach the encoder through a memory-warmed forward.

        Simulates exactly what the trainer does: burn-in under no_grad populates
        ``mems``, then a graduated forward over the *later* half of a sequence
        must still backprop into the weights via attention over [mem | x].
        """
        torch.manual_seed(0)
        model = TachyonActorCriticGTrXL(obs_dim=25, d_model=32, n_heads=4, n_layers=2, mem_len=8)
        seq = torch.randn(2, 20, 25)

        with torch.no_grad():
            _, _, mems = model(seq[:, :4])

        logits, values, _ = model(seq[:, 4:], mems=mems)
        (logits.sum() + values.sum()).backward()

        encoder_grad = model.encoder[0].weight.grad
        assert encoder_grad is not None
        assert float(encoder_grad.abs().sum()) > 0.0

    def test_recurrent_loss_differs_from_memoryless_loss(self) -> None:
        """The same trainable segment yields different losses with vs without memory."""
        torch.manual_seed(3)
        model = TachyonActorCriticGTrXL(obs_dim=25, d_model=32, n_heads=4, n_layers=2, mem_len=8)
        model.eval()
        seg_a = torch.randn(2, 6, 25)
        seg_b = torch.randn(2, 10, 25)

        with torch.no_grad():
            _, _, mems = model(seg_a)
            loss_warm = model(seg_b, mems=mems)[0].abs().mean()
            loss_cold = model(seg_b, mems=None)[0].abs().mean()

        assert not torch.isclose(loss_warm, loss_cold), (
            "memory burn-in has no effect on the trainable segment"
        )


class TestEdgeCases:
    def test_insufficient_samples_raises(self) -> None:
        buffer = AsyncReplayBuffer(capacity=128, state_dim=OBS_DIM, pin_memory=False)
        _populate_buffer(buffer, 8, seed=7)
        _, trainer, _ = _build_trainer()
        with pytest.raises(ValueError):
            trainer.update(buffer, batch_size=64)

    def test_consecutive_updates_stay_finite(self, populated_buffer: AsyncReplayBuffer) -> None:
        _, trainer, _ = _build_trainer()
        for _round in range(3):
            stats = trainer.update(populated_buffer, batch_size=64)
            assert np.isfinite(stats["actor_loss"])
            assert np.isfinite(stats["critic_loss"])

    def test_batch_not_divisible_by_seq_len_is_trimmed(
        self, populated_buffer: AsyncReplayBuffer
    ) -> None:
        """A batch_size between sequence multiples trains on floor(batch/seq_len)."""
        _, trainer, snapshot = _build_trainer()
        stats = trainer.update(populated_buffer, batch_size=70)  # 70 // 16 = 4 sequences
        assert np.isfinite(stats["actor_loss"])
        changed = sum(
            not torch.equal(param.detach(), snapshot[name])
            for name, param in trainer.model.named_parameters()
        )
        assert changed > 0

    @pytest.mark.parametrize("burn_in", [0, 15])
    def test_burn_in_bounds_accepted(self, burn_in: int) -> None:
        model = TachyonActorCriticGTrXL(obs_dim=25, d_model=32, n_heads=4, n_layers=1, mem_len=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = TachyonPPOTrainer(model, optimizer, seq_len=16, burn_in_steps=burn_in)
        assert trainer.burn_in_steps == burn_in

    @pytest.mark.parametrize("bad_burn_in", [-1, 16])
    def test_burn_in_out_of_bounds_rejected(self, bad_burn_in: int) -> None:
        model = TachyonActorCriticGTrXL(obs_dim=25, d_model=32, n_heads=4, n_layers=1, mem_len=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="burn_in_steps"):
            TachyonPPOTrainer(model, optimizer, seq_len=16, burn_in_steps=bad_burn_in)


class TestRewardProvenance:
    def test_buffer_rewards_are_dsr_shaped(self, populated_buffer: AsyncReplayBuffer) -> None:
        """Sanity on the fixture: pushed rewards really carry the DSR signature."""
        batch = populated_buffer.sample(256, device="cpu")
        rewards = batch.rewards
        assert torch.isfinite(rewards).all()
        # calculate_reward(pnl ~ N(0,1), inv in {-1,0,1}, var=1e-4) scales pnl by
        # eta/sigma = 0.05/0.01 = 5x minus a tiny penalty — so std >> raw pnl scale.
        assert float(rewards.std()) > 1.0
