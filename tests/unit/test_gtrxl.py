"""GTrXL Actor-Critic — unit tests for tachyon.rl.models.gtrxl.

Validates the forward contract (shapes/dtypes of logits, value, and the per-layer
memory cache) and the segment-recurrence semantics: memories must change the
output, must be reproducible when replayed, and must never carry gradients.
"""

from __future__ import annotations

import pytest
import torch

from tachyon.rl.models.gtrxl import GTrXLBlock, TachyonActorCriticGTrXL


@pytest.fixture
def model() -> TachyonActorCriticGTrXL:
    """Small deterministic GTrXL for CPU tests."""
    torch.manual_seed(7)
    return TachyonActorCriticGTrXL(
        obs_dim=25,
        d_model=32,
        n_heads=4,
        n_layers=2,
        mem_len=8,
        num_actions=4,
        mlp_ratio=2,
    )


class TestForwardShapes:
    def test_logits_value_and_memory_shapes(self, model: TachyonActorCriticGTrXL) -> None:
        obs = torch.randn(4, 16, 25)
        logits, value, new_mems = model(obs)

        assert logits.shape == (4, 16, 4)
        assert value.shape == (4, 16, 1)
        assert logits.dtype == torch.float32
        assert value.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()

        # One detached memory per layer, each exactly mem_len long.
        assert len(new_mems) == model.n_layers == 2
        for mem in new_mems:
            assert mem.shape == (4, 8, 32)
            assert not mem.requires_grad

    def test_single_timestep_step_like_call(self, model: TachyonActorCriticGTrXL) -> None:
        """Live rollout path: one bar at a time through a warm memory."""
        obs = torch.randn(1, 1, 25)
        _, _, mems = model(torch.randn(1, 12, 25))
        logits, value, new_mems = model(obs, mems=mems)
        assert logits.shape == (1, 1, 4)
        assert value.shape == (1, 1, 1)
        for old, new in zip(mems, new_mems, strict=True):
            # Memory slides forward by one: drops oldest row, appends newest.
            assert torch.equal(new[:, :-1, :], old[:, 1:, :])

    def test_invalid_obs_dim_raises(self, model: TachyonActorCriticGTrXL) -> None:
        with pytest.raises(ValueError, match="expected obs"):
            model(torch.randn(2, 4, 24))

    def test_wrong_memory_count_raises(self, model: TachyonActorCriticGTrXL) -> None:
        obs = torch.randn(1, 4, 25)
        with pytest.raises(ValueError, match="memory tensors"):
            model(obs, mems=[torch.zeros(1, 8, 32)])

    def test_block_rejects_undivisible_d_model(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            GTrXLBlock(d_model=30, n_heads=4)


class TestMemoryRecurrence:
    def test_output_diverges_with_and_without_memory(
        self, model: TachyonActorCriticGTrXL
    ) -> None:
        """Same second segment must differ once memory provides extended context."""
        model.eval()  # disable dropout so divergence is attributable to memory alone
        with torch.no_grad():
            seg_a = torch.randn(2, 10, 25)
            _, _, mems_from_a = model(seg_a)

            seg_b = torch.randn(2, 10, 25)

            logits_warm, value_warm, _ = model(seg_b, mems=mems_from_a)
            logits_cold, value_cold, _ = model(seg_b, mems=None)

        assert not torch.allclose(logits_warm, logits_cold, atol=1e-6), (
            "warm memory had no effect on action logits"
        )
        assert not torch.allclose(value_warm, value_cold, atol=1e-6), (
            "warm memory had no effect on value estimates"
        )

    def test_memory_replay_is_deterministic(self, model: TachyonActorCriticGTrXL) -> None:
        """Identical inputs + identical memories => bit-identical outputs."""
        model.eval()
        with torch.no_grad():
            seg_a = torch.randn(3, 8, 25)
            seg_b = torch.randn(3, 8, 25)
            _, _, mems = model(seg_a)

            logits_1, value_1, _ = model(seg_b, mems=list(mems))
            logits_2, value_2, _ = model(seg_b, mems=list(mems))

        assert torch.equal(logits_1, logits_2)
        assert torch.equal(value_1, value_2)

    def test_longer_memory_context_changes_output(self, model: TachyonActorCriticGTrXL) -> None:
        """A different history (not just any non-zero memory) changes the output."""
        model.eval()
        with torch.no_grad():
            seg_b = torch.randn(2, 6, 25)
            _, _, mems_x = model(torch.randn(2, 10, 25))
            _, _, mems_y = model(torch.randn(2, 10, 25))

            logits_x, _, _ = model(seg_b, mems=mems_x)
            logits_y, _, _ = model(seg_b, mems=mems_y)

        assert not torch.allclose(logits_x, logits_y, atol=1e-6)


class TestGradientIsolation:
    def test_backward_flows_to_heads_not_through_memories(
        self, model: TachyonActorCriticGTrXL
    ) -> None:
        obs = torch.randn(2, 8, 25)
        logits, value, _ = model(obs)
        loss = logits.sum() + value.sum()
        loss.backward()

        assert model.actor_head.weight.grad is not None
        assert model.encoder[0].weight.grad is not None

    def test_provided_memories_stay_grad_free(self, model: TachyonActorCriticGTrXL) -> None:
        model.train()  # dropout on — gradient isolation must hold regardless
        seg_a = torch.randn(2, 6, 25)
        _, _, mems = model(seg_a)

        for mem in mems:
            assert not mem.requires_grad
            assert mem.grad_fn is None

        seg_b = torch.randn(2, 6, 25).requires_grad_(True)
        logits, value, new_mems = model(seg_b, mems=mems)
        (logits.sum() + value.sum()).backward()

        assert seg_b.grad is not None  # current segment trains normally
        for new in new_mems:
            assert not new.requires_grad


class TestActionMasking:
    def test_masked_actions_get_neg_inf_mask_value(self, model: TachyonActorCriticGTrXL) -> None:
        obs = torch.randn(2, 5, 25)
        mask = torch.ones(2, 1, 4, dtype=torch.bool)
        mask[..., 1:] = False  # only HOLD legal
        logits, _, _ = model(obs, action_mask=mask)

        assert (logits[..., 0] > -1e3).all()
        assert (logits[..., 1:] <= -1e4 + 1e-3).all()

    def test_unmasked_logits_untouched_by_mask_path(self, model: TachyonActorCriticGTrXL) -> None:
        obs = torch.randn(1, 3, 25)
        full_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        logits_masked, _, _ = model(obs, action_mask=full_mask)
        logits_plain, _, _ = model(obs)
        assert torch.equal(logits_masked, logits_plain)
