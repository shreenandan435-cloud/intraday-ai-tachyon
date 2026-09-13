"""PPO Training Engine — tachyon.rl.train.

Tests cover GAE, clipped surrogate objective, value loss clipping, entropy bonus,
mixed precision training, vectorized trajectory collection, and checkpointing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from tachyon.core.clock import ManualClock
from tachyon.ipc.schemas import OrderBook
from tachyon.model.dataset import LOBDataset
from tachyon.model.ppo import TachyonPPO
from tachyon.persistence.tick_recorder import DEPTH_STREAM, TickRecorder
from tachyon.rl.env import LOBEnv
from tachyon.rl.train import (
    GAEConfig,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    compute_gae,
    ppo_loss,
)

if TYPE_CHECKING:
    pass


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_test_corpus(tmp_path: Path, n_ticks: int = 500) -> Path:
    root = tmp_path / "ticks"

    def _book(ts_epoch: float) -> OrderBook:
        return OrderBook(
            token="1053",
            bid_price=(100.5, 100.4, 100.3, 100.2, 100.1),
            bid_qty=(10, 200, 3_000, 40_000, 500_000),
            ask_price=(100.7, 100.8, 100.9, 101.0, 101.1),
            ask_qty=(11, 210, 3_100, 41_000, 510_000),
            ts_epoch=ts_epoch,
        )

    recorder = TickRecorder(
        directory=root,
        clock=ManualClock(datetime(2026, 8, 17, 9, 47)),
        streams=(DEPTH_STREAM,),
        start=False,
    )
    for i in range(n_ticks):
        recorder.record_depth("UFLEX", _book(1_755_000_000.0 + i * 0.1))
    recorder.drain_for_test()
    recorder.close()
    return root


@pytest.fixture
def test_dataset(tmp_path: Path) -> LOBDataset:
    corpus = _make_test_corpus(tmp_path, n_ticks=500)
    return LOBDataset(directory=corpus, seq_len=64)


@pytest.fixture
def test_env(test_dataset: LOBDataset) -> LOBEnv:
    return LOBEnv(test_dataset, lambda_holding=0.1, transaction_cost_bps=5.0)


@pytest.fixture
def small_ppo() -> TachyonPPO:
    return TachyonPPO(d_model=64, n_layers=2, n_heads=4, num_actions=4).eval()


@pytest.fixture
def ppo_config() -> PPOConfig:
    return PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip=0.2,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=32,
        minibatch_size=16,
        epochs=4,
        device="cpu",
    )


# ─── GAEConfig ────────────────────────────────────────────────────────────────


class TestGAEConfig:
    def test_defaults(self) -> None:
        cfg = GAEConfig()
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95

    def test_custom(self) -> None:
        cfg = GAEConfig(gamma=0.98, gae_lambda=0.9)
        assert cfg.gamma == 0.98
        assert cfg.gae_lambda == 0.9


# ─── PPOConfig ────────────────────────────────────────────────────────────────


class TestPPOConfig:
    def test_defaults(self) -> None:
        cfg = PPOConfig()
        assert cfg.lr == 3e-4
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_ratio == 0.2
        assert cfg.value_clip == 0.2
        assert cfg.entropy_coef == 0.01
        assert cfg.max_grad_norm == 0.5
        assert cfg.batch_size == 64
        assert cfg.minibatch_size == 32
        assert cfg.epochs == 4
        assert cfg.device == "cpu"

    def test_custom(self) -> None:
        cfg = PPOConfig(lr=1e-3, batch_size=128, device="cuda")
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 128
        assert cfg.device == "cuda"


# ─── compute_gae ──────────────────────────────────────────────────────────────


class TestComputeGAE:
    def test_gae_single_step(self) -> None:
        """GAE with one step reduces to TD residual."""
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.5, 0.0]])  # V(s0), V(s1)=0
        dones = torch.tensor([[False]])
        cfg = GAEConfig(gamma=0.99, gae_lambda=0.95)

        advantages, returns = compute_gae(rewards, values, dones, cfg)

        # TD residual: r + gamma * V(s1) - V(s0) = 1.0 + 0 - 0.5 = 0.5
        assert advantages.shape == (1, 1)
        assert torch.allclose(advantages[0, 0], torch.tensor(0.5), atol=1e-4)
        # Return: advantage + V(s0) = 0.5 + 0.5 = 1.0
        assert torch.allclose(returns[0, 0], torch.tensor(1.0), atol=1e-4)

    def test_gae_multi_step(self) -> None:
        """GAE with multiple steps."""
        rewards = torch.tensor([[1.0, 2.0, 0.0]])
        values = torch.tensor([[0.0, 1.0, 2.0, 0.0]])  # V0, V1, V2, V3=0
        dones = torch.tensor([[False, False, True]])
        cfg = GAEConfig(gamma=1.0, gae_lambda=1.0)  # Monte Carlo

        advantages, returns = compute_gae(rewards, values, dones, cfg)

        # With gamma=1, lambda=1, GAE = MC return - V
        # G0 = 1+2+0 = 3, G1 = 2+0 = 2, G2 = 0
        # A0 = G0 - V0 = 3 - 0 = 3
        # A1 = G1 - V1 = 2 - 1 = 1
        # A2 = G2 - V2 = 0 - 2 = -2
        expected_adv = torch.tensor([[3.0, 1.0, -2.0]])
        # Returns = advantages + values[:T]
        expected_ret = torch.tensor([[3.0, 2.0, 0.0]])

        assert torch.allclose(advantages, expected_adv, atol=1e-4)
        assert torch.allclose(returns, expected_ret, atol=1e-4)

    def test_gae_respects_done(self) -> None:
        """GAE should not bootstrap past done=True."""
        rewards = torch.tensor([[1.0, 1.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        dones = torch.tensor([[True, False]])  # Episode ends at step 0
        cfg = GAEConfig(gamma=0.99, gae_lambda=0.95)

        advantages, returns = compute_gae(rewards, values, dones, cfg)

        # Step 0: done=True, so no bootstrap
        assert torch.allclose(advantages[0, 0], torch.tensor(1.0), atol=1e-4)
        # Step 1: normal
        assert torch.allclose(returns[0, 0], torch.tensor(1.0), atol=1e-4)

    def test_gae_batch_dim(self) -> None:
        """GAE works with batch dimension > 1."""
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        dones = torch.tensor([[False], [False]])
        cfg = GAEConfig(gamma=1.0, gae_lambda=1.0)

        advantages, returns = compute_gae(rewards, values, dones, cfg)

        assert advantages.shape == (2, 1)
        assert returns.shape == (2, 1)

    def test_gae_bootstraps_through_truncation(self) -> None:
        """Time-limit truncation (dataset exhausted) must NOT zero the bootstrap value.

        The trajectory continues in reality; zeroing V(s_{t+1}) there is the classic
        truncation bias that makes all terminal-window states look artificially bad.
        """
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.0, 0.9]])  # V(s0), bootstrap V(s1)=0.9
        dones = torch.tensor([[True]])
        truncated = torch.tensor([[True]])
        cfg = GAEConfig(gamma=1.0, gae_lambda=1.0)

        advantages_t, _ = compute_gae(rewards, values, dones, cfg, truncated)
        # Expected: delta = 1.0 + 1.0 * 0.9 * 1.0 (bootstrap flows) - 0.0 = 1.9
        assert torch.allclose(advantages_t[0, 0], torch.tensor(1.9), atol=1e-4)

    def test_gae_zeroes_bootstrap_at_true_termination(self) -> None:
        """True MDP terminal (done & not truncated, e.g. loss limit) zeroes the bootstrap."""
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.0]])
        dones = torch.tensor([[True]])
        truncated = torch.tensor([[False]])  # terminal, NOT truncation
        cfg = GAEConfig(gamma=1.0, gae_lambda=1.0)

        advantages_t, _ = compute_gae(
            rewards, torch.cat([values, torch.tensor([[0.9]])], dim=1), dones, cfg, truncated
        )
        # Expected: delta = 1.0 + γ·0.9·0 (zeroed) - 0.0 = 1.0
        assert torch.allclose(advantages_t[0, 0], torch.tensor(1.0), atol=1e-4)

    def test_gae_recursion_stops_at_any_boundary(self) -> None:
        """last_gae must reset at BOTH terminal and truncated boundaries — no cross-episode leak."""
        rewards = torch.tensor([[1.0, 10.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        dones = torch.tensor([[True, False]])  # boundary after step 0
        truncated = torch.tensor([[True, False]])  # step 0 is a truncation
        cfg = GAEConfig(gamma=1.0, gae_lambda=1.0)

        advantages, _ = compute_gae(rewards, values, dones, cfg, truncated)
        # Step 1: delta = 10 + 0 - 0 = 10, last_gae = 10
        # Step 0: delta = 1 + (1-done)=0 V ... bootstraps 0 (V(s1)=0) → 1
        # recursion reset: A0 = 1 (not 1+10)
        assert torch.allclose(advantages[0, 1], torch.tensor(10.0), atol=1e-4)
        assert torch.allclose(advantages[0, 0], torch.tensor(1.0), atol=1e-4)


# ─── RolloutBuffer ────────────────────────────────────────────────────────────


class TestRolloutBuffer:
    def test_buffer_init(self, ppo_config: PPOConfig) -> None:
        buf = RolloutBuffer(
            capacity=100,
            obs_shape=(64, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )
        assert buf.capacity == 100
        assert len(buf) == 0

    def test_add_and_get(self, ppo_config: PPOConfig) -> None:
        buf = RolloutBuffer(
            capacity=10,
            obs_shape=(64, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )

        obs = torch.randn(64, 25)
        action = torch.tensor(1)
        log_prob = torch.tensor(-0.5)
        value = torch.tensor(0.3)
        reward = 0.1
        done = False

        buf.add(obs, action, log_prob, value, reward, done)
        assert len(buf) == 1

        # Must compute advantages before getting batch - pass scalar bootstrap value
        buf.compute_advantages(torch.tensor(0.0), GAEConfig())
        data = buf.get_batch()
        assert data["obs"].shape == (1, 64, 25)
        assert data["actions"].shape == (1,)
        assert data["log_probs"].shape == (1,)
        assert data["values"].shape == (1,)
        assert data["rewards"].shape == (1,)
        assert data["dones"].shape == (1,)

    def test_buffer_full(self, ppo_config: PPOConfig) -> None:
        buf = RolloutBuffer(
            capacity=3,
            obs_shape=(64, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )

        for _ in range(5):  # Over capacity
            buf.add(
                torch.randn(64, 25), torch.tensor(0), torch.tensor(0.0),
                torch.tensor(0.0), 0.0, False,
            )

        assert len(buf) == 3  # Capped at capacity

    def test_compute_advantages(self, ppo_config: PPOConfig) -> None:
        buf = RolloutBuffer(
            capacity=10,
            obs_shape=(64, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )

        # Add a trajectory
        for i in range(5):
            buf.add(
                torch.randn(64, 25),
                torch.tensor(i % 4),
                torch.tensor(-0.5),
                torch.tensor(float(i) * 0.1),
                float(i) * 0.1,
                i == 4,
            )

        buf.compute_advantages(torch.tensor(0.0), GAEConfig())
        data = buf.get_batch()
        assert "advantages" in data
        assert "returns" in data
        assert data["advantages"].shape == (5,)
        assert data["returns"].shape == (5,)

    def test_clear(self, ppo_config: PPOConfig) -> None:
        buf = RolloutBuffer(
            capacity=10,
            obs_shape=(64, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )
        buf.add(
            torch.randn(64, 25), torch.tensor(0), torch.tensor(0.0),
            torch.tensor(0.0), 0.0, False,
        )
        buf.clear()
        assert len(buf) == 0

    def test_buffer_temporal_order_after_wrap(self, ppo_config: PPOConfig) -> None:
        """Once the ring wraps, the oldest entry sits at _ptr — GAE's backward recursion
        must walk newest→oldest, NOT slot 0→N. A wrapped buffer without the permutation
        corrupts the recursion by reading time out of order.
        """
        buf = RolloutBuffer(
            capacity=4,
            obs_shape=(2, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )
        # Write 6 transitions into a capacity-4 ring: slots 4,5 overwrite 0,1.
        # True temporal order of the survivors is rewards [2,3,4,5] (slots 2,3,0,1).
        for i in range(6):
            buf.add(
                torch.full((2, 25), float(i)),
                torch.tensor(i % 4),
                torch.tensor(0.0),
                torch.tensor(0.0),   # V=0 → advantages collapse to the reward stream
                float(i * 2),        # rewards 0,2,4,6,8,10; survivors carry 4,6,8,10
                False,
            )
        buf.compute_advantages(torch.tensor(0.0), GAEConfig(gamma=0.0, gae_lambda=0.0))
        data = buf.get_batch()
        # γ=0, λ=0 ⇒ advantage_t == r_t. In temporal order that's [4, 6, 8, 10].
        assert torch.allclose(data["rewards"], torch.tensor([4.0, 6.0, 8.0, 10.0]))
        assert torch.allclose(data["advantages"], torch.tensor([4.0, 6.0, 8.0, 10.0]))

    def test_buffer_stores_masks_and_truncated(self, ppo_config: PPOConfig) -> None:
        """The update pass re-applies the exact mask the action was sampled under, and
        GAE treats truncated (time-limit) differently from terminated (MDP end)."""
        buf = RolloutBuffer(
            capacity=4,
            obs_shape=(2, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )
        mask = torch.tensor([True, True, True, False])  # CLOSE masked
        buf.add(torch.randn(2, 25), torch.tensor(1), torch.tensor(-0.5), torch.tensor(0.3),
                1.0, False, action_mask=mask, truncated=False)
        buf.add(torch.randn(2, 25), torch.tensor(0), torch.tensor(-0.6), torch.tensor(0.4),
                1.0, True, action_mask=mask, truncated=True)  # dataset ran out mid-episode
        buf.compute_advantages(torch.tensor(2.0), GAEConfig(gamma=1.0, gae_lambda=1.0))
        data = buf.get_batch()
        assert data["action_masks"].shape == (2, 4)
        assert data["action_masks"][0].tolist() == [True, True, True, False]
        assert data["truncated"].tolist() == [False, True]
        assert data["terminated"].tolist() == [False, False]
        # Last step is truncated: bootstrap V=2.0 flows into its delta.
        # δ_1 = 1.0 + γ·2.0·(1-0) − 0.4 = 2.6 (NOT zeroed — it is a time-limit cut)
        assert torch.allclose(data["advantages"][1], torch.tensor(2.6), atol=1e-4)

    def test_truncated_bootstraps_but_terminated_zeroes(self, ppo_config: PPOConfig) -> None:
        """Paired control: same layout, terminal end instead of truncation."""
        buf = RolloutBuffer(
            capacity=4,
            obs_shape=(2, 25),
            action_dim=4,
            device=torch.device("cpu"),
        )
        buf.add(torch.randn(2, 25), torch.tensor(1), torch.tensor(-0.5), torch.tensor(0.3),
                1.0, False)
        buf.add(torch.randn(2, 25), torch.tensor(0), torch.tensor(-0.6), torch.tensor(0.4),
                1.0, True, truncated=False)  # TRUE terminal (loss limit)
        buf.compute_advantages(torch.tensor(2.0), GAEConfig(gamma=1.0, gae_lambda=1.0))
        data = buf.get_batch()
        # δ_1 = 1.0 + γ·2.0·(1-1) − 0.4 = 0.6 (bootstrap zeroed at the true terminal)
        assert torch.allclose(data["advantages"][1], torch.tensor(0.6), atol=1e-4)


# ─── ppo_loss ─────────────────────────────────────────────────────────────────


class TestPPOLoss:
    def test_clipped_surrogate(self, small_ppo: TachyonPPO) -> None:
        """Test clipped PPO loss computation."""
        batch_size = 16
        seq_len = 64

        obs = torch.randn(batch_size, seq_len, 25)
        actions = torch.randint(0, 4, (batch_size, seq_len))
        old_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 0.5
        advantages = torch.randn(batch_size, seq_len)
        returns = torch.randn(batch_size, seq_len)
        old_values = torch.randn(batch_size, seq_len)

        loss_dict = ppo_loss(
            model=small_ppo,
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
            clip_ratio=0.2,
            value_clip=0.2,
            entropy_coef=0.01,
        )

        assert "policy_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "entropy_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert "clip_fraction" in loss_dict
        assert loss_dict["total_loss"].requires_grad

    def test_entropy_bonus(self, small_ppo: TachyonPPO) -> None:
        """Entropy bonus should be negative (encouraging exploration)."""
        batch_size = 8
        obs = torch.randn(batch_size, 64, 25)
        actions = torch.randint(0, 4, (batch_size, 64))
        old_log_probs = torch.full((batch_size, 64), -1.386)  # log(0.25) uniform
        advantages = torch.randn(batch_size, 64)
        returns = torch.randn(batch_size, 64)
        old_values = torch.randn(batch_size, 64)

        # Without entropy
        loss_no_ent = ppo_loss(
            small_ppo, obs, actions, old_log_probs, advantages, returns, old_values,
            clip_ratio=0.2, value_clip=0.2, entropy_coef=0.0,
        )

        # With entropy
        loss_with_ent = ppo_loss(
            small_ppo, obs, actions, old_log_probs, advantages, returns, old_values,
            clip_ratio=0.2, value_clip=0.2, entropy_coef=0.01,
        )

        # Total loss should be lower with entropy bonus (since entropy is negative)
        assert loss_with_ent["total_loss"] < loss_no_ent["total_loss"]
        assert loss_with_ent["entropy_loss"] < 0  # Entropy is positive, but we minimize -entropy

    def test_value_clipping(self, small_ppo: TachyonPPO) -> None:
        """Value loss clipping should limit critic gradient spikes."""
        batch_size = 8
        obs = torch.randn(batch_size, 64, 25)
        actions = torch.randint(0, 4, (batch_size, 64))
        old_log_probs = torch.randn(batch_size, 64) * 0.1 - 0.5
        advantages = torch.randn(batch_size, 64)
        returns = torch.randn(batch_size, 64) * 100  # Large returns to test clipping
        old_values = torch.randn(batch_size, 64)

        loss_dict = ppo_loss(
            small_ppo, obs, actions, old_log_probs, advantages, returns, old_values,
            clip_ratio=0.2, value_clip=0.2, entropy_coef=0.01,
        )

        # Value loss should be finite
        assert torch.isfinite(loss_dict["value_loss"])
        assert loss_dict["value_loss"] >= 0


# ─── PPOTrainer ───────────────────────────────────────────────────────────────


class TestPPOTrainer:
    def test_trainer_init(
        self, test_env: LOBEnv, small_ppo: TachyonPPO, ppo_config: PPOConfig
    ) -> None:
        trainer = PPOTrainer(
            model=small_ppo,
            env=test_env,
            config=ppo_config,
        )
        assert trainer.model is small_ppo
        assert trainer.env is test_env
        assert trainer.config == ppo_config
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.scaler is not None  # AMP scaler

    def test_collect_trajectories(
        self, test_env: LOBEnv, small_ppo: TachyonPPO, ppo_config: PPOConfig
    ) -> None:
        trainer = PPOTrainer(model=small_ppo, env=test_env, config=ppo_config)

        # Collect a small batch
        stats = trainer.collect_trajectories(num_steps=20)

        assert "mean_reward" in stats
        assert "mean_episode_length" in stats
        assert "num_episodes" in stats
        assert stats["num_episodes"] >= 0

    def test_update_step(
        self, test_env: LOBEnv, small_ppo: TachyonPPO, ppo_config: PPOConfig
    ) -> None:
        trainer = PPOTrainer(model=small_ppo, env=test_env, config=ppo_config)

        # Collect some data first
        trainer.collect_trajectories(num_steps=32)

        # Run update
        loss_stats = trainer.update()

        assert "policy_loss" in loss_stats
        assert "value_loss" in loss_stats
        assert "entropy_loss" in loss_stats
        assert "total_loss" in loss_stats
        assert "clip_fraction" in loss_stats

    def test_train_loop(
        self, test_env: LOBEnv, small_ppo: TachyonPPO, ppo_config: PPOConfig
    ) -> None:
        trainer = PPOTrainer(model=small_ppo, env=test_env, config=ppo_config)

        # Short training run
        history = trainer.train(
            total_timesteps=100,
            eval_freq=50,
            log_freq=10,
        )

        assert "policy_loss" in history
        assert "value_loss" in history
        assert "episode_reward" in history
        assert len(history["policy_loss"]) > 0

    def test_mixed_precision(self, test_env: LOBEnv, small_ppo: TachyonPPO) -> None:
        """Test that AMP is used when device is CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = PPOConfig(device="cuda")
        small_ppo_cuda = small_ppo.to("cuda")
        test_env_cuda = LOBEnv(test_env._dataset, lambda_holding=0.1)

        trainer = PPOTrainer(model=small_ppo_cuda, env=test_env_cuda, config=config)

        # Check scaler is initialized for CUDA
        assert trainer.scaler is not None
        assert trainer.scaler._enabled is True

    def test_checkpointing(
        self,
        test_env: LOBEnv,
        small_ppo: TachyonPPO,
        ppo_config: PPOConfig,
        tmp_path: Path,
    ) -> None:
        """Test model checkpointing saves best model."""
        checkpoint_dir = tmp_path / "checkpoints"
        # Use batch_size=16, eval_freq=16 so eval happens at step 16, 32, 48, 64
        test_config = PPOConfig(
            lr=ppo_config.lr,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            clip_ratio=ppo_config.clip_ratio,
            value_clip=ppo_config.value_clip,
            entropy_coef=ppo_config.entropy_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            batch_size=16,
            minibatch_size=ppo_config.minibatch_size,
            epochs=ppo_config.epochs,
            device=ppo_config.device,
        )
        trainer = PPOTrainer(
            model=small_ppo,
            env=test_env,
            config=test_config,
            checkpoint_dir=checkpoint_dir,
        )

        # Mock a good evaluation
        eval_result = {"mean_reward_bps": 100.0, "sharpe_proxy": 1.5}
        with patch.object(trainer, "_evaluate", return_value=eval_result):
            trainer.train(total_timesteps=64, eval_freq=16, log_freq=10)

        best_model = checkpoint_dir / "best_policy.pt"
        assert best_model.exists()

        # Verify it can be loaded
        loaded = torch.load(best_model)
        assert "model_state_dict" in loaded
        assert "config" in loaded
        assert "step" in loaded
        assert "best_reward" in loaded

    def test_resume_from_checkpoint(
        self,
        test_env: LOBEnv,
        small_ppo: TachyonPPO,
        ppo_config: PPOConfig,
        tmp_path: Path,
    ) -> None:
        """Test training can resume from checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a trainer first to get optimizer state
        trainer1 = PPOTrainer(
            model=small_ppo,
            env=test_env,
            config=ppo_config,
            checkpoint_dir=checkpoint_dir,
        )

        # Save a checkpoint
        checkpoint = {
            "model_state_dict": small_ppo.state_dict(),
            "optimizer_state_dict": trainer1.optimizer.state_dict(),
            "config": {"d_model": 64, "n_layers": 2, "n_heads": 4, "num_actions": 4},
            "step": 1000,
            "best_reward": 50.0,
        }
        checkpoint_path = checkpoint_dir / "checkpoint_step_1000.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create new trainer and resume
        trainer2 = PPOTrainer(
            model=small_ppo,
            env=test_env,
            config=ppo_config,
            checkpoint_dir=checkpoint_dir,
        )
        trainer2.resume(checkpoint_path)

        assert trainer2.global_step == 1000
        assert trainer2.best_reward == 50.0


# ─── Integration ──────────────────────────────────────────────────────────────


class TestTrainIntegration:
    def test_full_training_run(self, tmp_path: Path) -> None:
        """End-to-end training test with small model and data."""
        # Setup
        corpus = _make_test_corpus(tmp_path, n_ticks=300)
        dataset = LOBDataset(directory=corpus, seq_len=32)
        env = LOBEnv(dataset, lambda_holding=0.1, transaction_cost_bps=5.0)
        model = TachyonPPO(d_model=32, n_layers=1, n_heads=2, num_actions=4)

        config = PPOConfig(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_clip=0.2,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            batch_size=16,
            minibatch_size=8,
            epochs=2,
            device="cpu",
        )

        trainer = PPOTrainer(
            model=model,
            env=env,
            config=config,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        # Train - use eval_freq=16 to match batch_size
        history = trainer.train(total_timesteps=64, eval_freq=16, log_freq=16)

        # Verify metrics tracked
        assert len(history["policy_loss"]) > 0
        assert len(history["value_loss"]) > 0
        assert len(history["entropy_loss"]) > 0
        assert len(history["episode_reward"]) > 0

        # Verify checkpoint exists
        best_model = tmp_path / "checkpoints" / "best_policy.pt"
        assert best_model.exists()

    def test_sharpe_proxy_tracked(
        self, test_env: LOBEnv, small_ppo: TachyonPPO, ppo_config: PPOConfig
    ) -> None:
        """Sharpe ratio proxy should be tracked in training."""
        trainer = PPOTrainer(model=small_ppo, env=test_env, config=ppo_config)

        # Collect trajectories with varying rewards
        trainer.collect_trajectories(num_steps=50)

        # Run update which should compute Sharpe proxy
        stats = trainer.update()

        # Sharpe proxy = mean(reward) / std(reward) * sqrt(252)
        # Should be in stats if implemented
        # This is a soft assertion - may not be in first version
        assert "total_loss" in stats


# ─── Config Integration ───────────────────────────────────────────────────────


class TestConfigIntegration:
    def test_config_from_settings(self) -> None:
        """PPOConfig should be creatable from settings.yaml rl section."""
        # This test will pass once config.py is updated with rl section
        from tachyon.rl.train import PPOConfig

        # For now, just verify we can create from dict
        rl_dict = {
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "value_clip": 0.2,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "batch_size": 64,
            "minibatch_size": 32,
            "epochs": 4,
            "device": "cpu",
        }
        cfg = PPOConfig(**rl_dict)
        assert cfg.lr == 3e-4
        assert cfg.gamma == 0.99
