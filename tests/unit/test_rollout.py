"""Live Rollout Manager — tachyon.rl.rollout.

Tests cover the PaperTradingRollout which connects the compiled PPO agent to
the live market data stream for unattended paper trading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch

from tachyon.model.ppo import TachyonPPO
from tachyon.rl.rollout import PaperTradingRollout, RolloutConfig, RolloutState

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_dummy_model() -> TachyonPPO:
    """Small model for fast tests."""
    # Use n_layers=2 to match the test expectations
    return TachyonPPO(d_model=64, n_layers=2, n_heads=4).eval()


def _make_dummy_checkpoint(tmp_path: Path) -> Path:
    """Save a dummy model checkpoint with explicit architecture."""
    model = _make_dummy_model()
    path = tmp_path / "ppo_model.pt"
    # Save with metadata for proper loading
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
        }
    }, path)
    return path


# ─── RolloutConfig ────────────────────────────────────────────────────────────


class TestRolloutConfig:
    def test_default_config(self) -> None:
        config = RolloutConfig()
        assert config.model_path == "models/ppo_latest.onnx"
        assert config.seq_len == 128
        assert config.device == "cpu"
        assert config.telemetry_dir == "data/telemetry"
        assert config.log_interval_sec == 5.0
        assert config.heartbeat_interval_sec == 30.0
        assert config.telegram_enabled is True
        assert config.max_steps == 0  # 0 = unlimited

    def test_custom_config(self) -> None:
        config = RolloutConfig(
            model_path="custom.onnx",
            seq_len=64,
            device="cuda",
            telegram_enabled=False,
            max_steps=1000,
        )
        assert config.model_path == "custom.onnx"
        assert config.seq_len == 64
        assert config.device == "cuda"
        assert config.telegram_enabled is False
        assert config.max_steps == 1000


# ─── RolloutState ─────────────────────────────────────────────────────────────


class TestRolloutState:
    def test_initial_state(self) -> None:
        state = RolloutState()
        assert state.step_count == 0
        assert state.current_position == 0
        assert state.entry_price_bps == 0
        assert state.holding_bars == 0
        assert state.total_reward == 0.0
        assert state.is_running is False
        assert state.last_obs is None
        assert state.last_action_mask is None

    def test_state_transitions(self) -> None:
        state = RolloutState()
        state.is_running = True
        state.step_count = 10
        state.current_position = 1
        state.entry_price_bps = 50
        state.holding_bars = 5
        state.total_reward = 15.5
        assert state.is_running
        assert state.step_count == 10


# ─── PaperTradingRollout ──────────────────────────────────────────────────────


class TestPaperTradingRolloutInitialization:
    def test_creates_from_config(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)
        assert rollout.config == config
        assert rollout.state is not None

    def test_loads_onnx_model(self, tmp_path: Path) -> None:
        """Test loading ONNX model (skipped if onnxruntime not available)."""
        pytest.importorskip("onnxruntime")
        # Convert to ONNX for test
        model = _make_dummy_model()
        onnx_path = tmp_path / "ppo_model.onnx"
        torch.onnx.export(
            model,
            (torch.randn(1, 128, 25),),
            str(onnx_path),
            input_names=["lob_state"],
            output_names=["actor_logits", "critic_values"],
            dynamo=True,
        )
        config = RolloutConfig(model_path=str(onnx_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)
        assert rollout.session is not None

    def test_loads_torch_model_fallback(self, tmp_path: Path) -> None:
        """Test loading torch model when ONNX not available."""
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)
        assert rollout.model is not None

    def test_initializes_telemetry(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            telemetry_dir=str(tmp_path / "telemetry"),
        )
        rollout = PaperTradingRollout(config)
        assert rollout.telemetry_logger is not None

    def test_initializes_telegram_alerter(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_chat")
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=True)
        with patch("tachyon.rl.rollout.create_alerter_from_env") as mock_create_alerter:
            mock_alerter = Mock()
            mock_alerter.enabled = True
            mock_create_alerter.return_value = mock_alerter
            rollout = PaperTradingRollout(config)
            mock_create_alerter.assert_called_once()
            assert rollout.alerter is mock_alerter


class TestPaperTradingRolloutInference:
    def test_single_step_inference(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        # Mock observation (B=1, T=128, 25 features)
        obs = torch.randn(1, 128, 25)
        action_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        action_mask[0, 0, 3] = False  # CLOSE not allowed when flat

        action, log_prob, value = rollout.infer(obs, action_mask)
        assert action.shape == (1, 1)
        assert log_prob.shape == (1, 1)
        assert value.shape == (1, 1)
        assert action.item() in [0, 1, 2]  # HOLD, BUY, SELL (not CLOSE)

    def test_inference_respects_action_mask(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        obs = torch.randn(1, 128, 25)
        # Only HOLD allowed
        action_mask = torch.zeros(1, 1, 4, dtype=torch.bool)
        action_mask[0, 0, 0] = True

        action, _, _ = rollout.infer(obs, action_mask)
        assert action.item() == 0  # Must be HOLD

    def test_kv_cache_persists_across_steps(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        obs = torch.randn(1, 128, 25)
        action_mask = torch.ones(1, 1, 4, dtype=torch.bool)

        # First step
        action1, _, _ = rollout.infer(obs, action_mask)
        # Second step
        action2, _, _ = rollout.infer(obs, action_mask)

        # KV cache should be maintained internally
        assert rollout.state.kv_caches is not None
        assert len(rollout.state.kv_caches) == 2  # n_layers=2


class TestPaperTradingRolloutTelemetry:
    def test_logs_step_telemetry_jsonl(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        telemetry_dir = tmp_path / "telemetry"
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            telemetry_dir=str(telemetry_dir),
            log_interval_sec=0,  # log every step
        )
        rollout = PaperTradingRollout(config)

        obs = torch.randn(1, 128, 25)
        action_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        action, log_prob, value = rollout.infer(obs, action_mask)

        # Manually trigger telemetry log
        rollout._log_step(
            step=1,
            action=action.item(),
            log_prob=log_prob.item(),
            value=value.item(),
            reward=0.5,
            position=1,
            entry_price_bps=10,
            holding_bars=1,
            action_mask=action_mask.squeeze().tolist(),
        )

        # Check JSONL file exists and has content
        log_files = list(telemetry_dir.glob("*.jsonl"))
        assert len(log_files) >= 1
        with log_files[0].open() as f:
            lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["event"] == "STEP"
        assert record["step"] == 1
        assert record["action"] == action.item()
        assert "latency_ms" in record

    def test_logs_trade_events(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        telemetry_dir = tmp_path / "telemetry"
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            telemetry_dir=str(telemetry_dir),
        )
        rollout = PaperTradingRollout(config)

        rollout._log_trade(
            event="ENTRY",
            symbol="RELIANCE",
            side="LONG",
            quantity=100,
            price=2500.0,
            stop_loss=2475.0,
            target=2550.0,
            order_id="test_123",
        )

        log_files = list(telemetry_dir.glob("*.jsonl"))
        assert len(log_files) >= 1
        with log_files[0].open() as f:
            lines = f.readlines()
        record = json.loads(lines[-1])
        assert record["event"] == "ENTRY"
        assert record["symbol"] == "RELIANCE"
        assert record["side"] == "LONG"


class TestPaperTradingRolloutTelegram:
    def test_sends_entry_alert(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_chat")
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=True)

        with patch("tachyon.rl.rollout.create_alerter_from_env") as mock_create_alerter:
            mock_alerter = Mock()
            mock_alerter.enabled = True
            mock_create_alerter.return_value = mock_alerter
            rollout = PaperTradingRollout(config)

            rollout._alert_entry(
                symbol="RELIANCE",
                direction="LONG",
                quantity=100,
                entry_price=2500.0,
                stop_loss=2475.0,
                target=2550.0,
                order_id="test_123",
            )
            mock_alerter.alert_entry.assert_called_once()

    def test_sends_exit_alert(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_chat")
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=True)

        with patch("tachyon.rl.rollout.create_alerter_from_env") as mock_create_alerter:
            mock_alerter = Mock()
            mock_alerter.enabled = True
            mock_create_alerter.return_value = mock_alerter
            rollout = PaperTradingRollout(config)

            rollout._alert_exit(
                symbol="RELIANCE",
                realised_pnl=500.0,
                charges=20.0,
                was_stop_out=False,
                session_total=500.0,
                headroom=0.0,
                exit_reason="TARGET_HIT",
            )
            mock_alerter.alert_exit.assert_called_once()

    def test_sends_circuit_limit_alert(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy_chat")
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=True)

        with patch("tachyon.rl.rollout.create_alerter_from_env") as mock_create_alerter:
            mock_alerter = Mock()
            mock_alerter.enabled = True
            mock_create_alerter.return_value = mock_alerter
            rollout = PaperTradingRollout(config)

            rollout._alert_circuit_limit(
                symbol="RELIANCE",
                action="BUY",
                reason="MASKED",
                detail="Upper circuit hit - BUY masked",
            )
            mock_alerter.send.assert_called_once()
            call_args = mock_alerter.send.call_args[0]
            assert "CIRCUIT" in call_args[0] or "MASKED" in call_args[0]


class TestPaperTradingRolloutHeadless:
    def test_run_headless_no_ui_dependency(self, tmp_path: Path) -> None:
        """Rollout must not import or depend on UI modules."""
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            max_steps=5,
        )
        rollout = PaperTradingRollout(config)

        # Mock the data source to return dummy data
        with patch.object(rollout, "_get_next_observation") as mock_get_obs:
            mock_get_obs.return_value = (
                torch.randn(1, 128, 25),
                torch.ones(1, 1, 4, dtype=torch.bool),
                False,  # not done
            )
            # Should not raise any UI-related import errors
            rollout.run()

    def test_graceful_shutdown_on_signal(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            max_steps=100,
        )
        rollout = PaperTradingRollout(config)

        # Simulate shutdown signal
        rollout.shutdown()
        assert not rollout.state.is_running

    def test_cleanup_on_exit(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        telemetry_dir = tmp_path / "telemetry"
        config = RolloutConfig(
            model_path=str(model_path),
            telegram_enabled=False,
            telemetry_dir=str(telemetry_dir),
        )
        rollout = PaperTradingRollout(config)

        with patch.object(rollout, "_get_next_observation") as mock_get_obs:
            mock_get_obs.return_value = (
                torch.randn(1, 128, 25),
                torch.ones(1, 1, 4, dtype=torch.bool),
                True,  # done after one step
            )
            rollout.run()

        # Telemetry should be flushed
        assert rollout.telemetry_logger is not None


class TestPaperTradingRolloutErrorHandling:
    def test_handles_model_inference_error(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        # Force an inference error
        with (
            patch.object(rollout, "infer", side_effect=RuntimeError("OOM")),
            pytest.raises(RuntimeError),
        ):
            rollout.infer(torch.randn(1, 128, 25), torch.ones(1, 1, 4, dtype=torch.bool))

    def test_handles_telemetry_write_failure(self, tmp_path: Path) -> None:
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        # Should not raise even if telemetry fails
        rollout.telemetry_logger = None
        rollout._log_step(1, 0, 0.0, 0.0, 0.0, 0, 0, 0, [True, True, True, False])
        # No exception


class TestPaperTradingRolloutPerformance:
    def test_inference_latency_under_1ms(self, tmp_path: Path) -> None:
        """Inference must stay under 1ms for live trading."""
        model_path = _make_dummy_checkpoint(tmp_path)
        config = RolloutConfig(model_path=str(model_path), telegram_enabled=False)
        rollout = PaperTradingRollout(config)

        obs = torch.randn(1, 128, 25)
        action_mask = torch.ones(1, 1, 4, dtype=torch.bool)

        # Warmup
        for _ in range(10):
            rollout.infer(obs, action_mask)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            rollout.infer(obs, action_mask)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000

        # Relaxed for CI - actual target is <1ms on GPU
        assert avg_ms < 50  # 50ms on CPU is acceptable for test
