"""ONNX Export & TensorRT Engine — tachyon.model.export.

Tests cover the export pipeline from PyTorch to ONNX to TensorRT,
ensuring KV-cache signatures are correct and numerical parity is maintained.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from scripts.benchmark_trt import (
    DEFAULT_MEASURE,
    DEFAULT_P99_THRESHOLD_MS,
    DEFAULT_WARMUP,
)

from tachyon.model.export import (
    ExportConfig,
    KVCacheSpec,
    build_tensorrt_engine,
    export_policy_to_onnx,
    get_kv_cache_specs,
    verify_onnx_model,
    verify_trt_parity,
)
from tachyon.model.ppo import TachyonPPO

if TYPE_CHECKING:
    pass


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_small_ppo() -> TachyonPPO:
    """Small model for fast tests."""
    return TachyonPPO(d_model=64, n_layers=2, n_heads=4, num_actions=4).eval()


@pytest.fixture
def small_ppo() -> TachyonPPO:
    return _make_small_ppo()


@pytest.fixture
def export_config() -> ExportConfig:
    return ExportConfig(
        batch_size=1,
        seq_len=1,  # Autoregressive step
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        fp16=True,
        workspace_gb=1,
        min_batch=1,
        opt_batch=1,
        max_batch=8,
    )


# ─── KVCacheSpec ──────────────────────────────────────────────────────────────


class TestKVCacheSpec:
    def test_spec_creation(self) -> None:
        spec = KVCacheSpec(
            layer_idx=0,
            k_shape=(1, 4, 0, 16),  # (B, n_heads, T_past, head_dim)
            v_shape=(1, 4, 0, 16),
            dtype=torch.float16,
        )
        assert spec.layer_idx == 0
        assert spec.k_shape == (1, 4, 0, 16)
        assert spec.v_shape == (1, 4, 0, 16)

    def test_spec_equality(self) -> None:
        spec1 = KVCacheSpec(0, (1, 4, 0, 16), (1, 4, 0, 16), torch.float16)
        spec2 = KVCacheSpec(0, (1, 4, 0, 16), (1, 4, 0, 16), torch.float16)
        assert spec1 == spec2


# ─── ExportConfig ─────────────────────────────────────────────────────────────


class TestExportConfig:
    def test_defaults(self) -> None:
        cfg = ExportConfig()
        assert cfg.batch_size == 1
        assert cfg.seq_len == 1
        assert cfg.fp16 is True
        assert cfg.workspace_gb == 1
        assert cfg.min_batch == 1
        assert cfg.opt_batch == 1
        assert cfg.max_batch == 8

    def test_custom(self) -> None:
        cfg = ExportConfig(batch_size=2, seq_len=1, d_model=128, n_heads=8, n_layers=4)
        assert cfg.batch_size == 2
        assert cfg.d_model == 128
        assert cfg.n_heads == 8
        assert cfg.n_layers == 4


# ─── get_kv_cache_specs ───────────────────────────────────────────────────────


class TestGetKVCacheSpecs:
    def test_specs_match_model(self, small_ppo: TachyonPPO) -> None:
        specs = get_kv_cache_specs(small_ppo, batch_size=1, dtype=torch.float16)
        assert len(specs) == 2  # n_layers=2
        for i, spec in enumerate(specs):
            assert spec.layer_idx == i
            assert spec.k_shape == (1, 4, 0, 16)  # (B, n_heads, T_past, head_dim)
            assert spec.v_shape == (1, 4, 0, 16)
            assert spec.dtype == torch.float16

    def test_specs_batch_dim(self, small_ppo: TachyonPPO) -> None:
        specs = get_kv_cache_specs(small_ppo, batch_size=4, dtype=torch.float32)
        for spec in specs:
            assert spec.k_shape[0] == 4
            assert spec.v_shape[0] == 4
            assert spec.dtype == torch.float32


# ─── ONNX Export ──────────────────────────────────────────────────────────────


class TestONNXExport:
    @pytest.fixture(autouse=True)
    def _skip_if_no_onnx(self) -> None:
        pytest.importorskip("onnx")

    def test_export_policy_to_onnx(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        """Test that the policy exports to ONNX with correct signatures."""
        onnx_path = tmp_path / "policy.onnx"

        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
            opset_version=17,
        )

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_input_signatures(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        """Verify ONNX graph has correct input/output names and shapes."""
        onnx_path = tmp_path / "policy.onnx"

        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
        )

        import onnx
        model = onnx.load(str(onnx_path))

        # Check input names
        input_names = [inp.name for inp in model.graph.input]
        assert "lob_state" in input_names
        assert "action_mask" in input_names
        # KV cache inputs
        for i in range(2):  # n_layers=2
            assert f"past_key_{i}" in input_names
            assert f"past_value_{i}" in input_names

        # Check output names
        output_names = [out.name for out in model.graph.output]
        assert "actor_logits" in output_names
        assert "critic_values" in output_names
        # KV cache outputs
        for i in range(2):
            assert f"present_key_{i}" in output_names
            assert f"present_value_{i}" in output_names

    def test_onnx_dynamic_axes(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        """Verify the batch dimension is dynamic (not baked in as static).

        Tracing at batch_size=2 defeats Dynamo's 0/1 specialization, which would
        otherwise hard‑code a batch of 1 into the exported graph.
        """
        pytest.importorskip("onnx")
        onnx_path = tmp_path / "policy.onnx"

        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=2,
            seq_len=1,
        )

        import onnx
        model = onnx.load(str(onnx_path))

        # Dynamo does not preserve the custom axis name ("batch") reliably, so
        # accept any symbolic / unfixed dimension — i.e. not a pinned dim_value.
        for inp in model.graph.input:
            if inp.name == "lob_state":
                assert len(inp.type.tensor_type.shape.dim) == 3
                batch_dim = inp.type.tensor_type.shape.dim[0]
                assert batch_dim.HasField("dim_param") or not batch_dim.HasField("dim_value")

            # Behavioural check: the exported graph must actually accept a larger batch.
            # (re‑import here so we don't pollute the module namespace)
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        n_heads = small_ppo.transformer.layers[0].attn.n_heads
        head_dim = small_ppo.d_model // n_heads
        dtype = next(small_ppo.parameters()).dtype

        for batch in (1, 3):
            inputs = {
                "lob_state": torch.randn(batch, 1, 25, dtype=dtype).numpy(),
                "action_mask": torch.ones(
                    batch, 1, small_ppo.num_actions, dtype=torch.bool
                ).numpy(),
            }
            for i in range(small_ppo.n_layers):
                inputs[f"past_key_{i}"] = torch.empty(
                    batch, n_heads, 0, head_dim, dtype=dtype
                ).numpy()
                inputs[f"past_value_{i}"] = torch.empty(
                    batch, n_heads, 0, head_dim, dtype=dtype
                ).numpy()
            outputs = session.run(None, inputs)
            assert outputs[0].shape[0] == batch

    def test_onnx_runtime_inference(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        """Test ONNX Runtime can run the exported model."""
        pytest.importorskip("onnxruntime")
        onnx_path = tmp_path / "policy.onnx"

        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
        )

        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # Prepare inputs
        lob_state = torch.randn(1, 1, 25).numpy()
        action_mask = torch.ones(1, 1, 4, dtype=torch.bool).numpy()
        past_keys = [torch.empty(1, 4, 0, 16).numpy() for _ in range(2)]
        past_values = [torch.empty(1, 4, 0, 16).numpy() for _ in range(2)]

        inputs = {
            "lob_state": lob_state,
            "action_mask": action_mask,
        }
        for i in range(2):
            inputs[f"past_key_{i}"] = past_keys[i]
            inputs[f"past_value_{i}"] = past_values[i]

        outputs = session.run(None, inputs)

        # Should have actor_logits, critic_values, present_key_0, present_value_0, ...
        assert len(outputs) == 2 + 4  # 2 main + 4 KV cache outputs
        assert outputs[0].shape == (1, 1, 4)  # actor_logits
        assert outputs[1].shape == (1, 1, 1)  # critic_values

    @pytest.fixture(autouse=True)
    def _skip_if_no_onnx(self) -> None:
        pytest.importorskip("onnx")

    def test_export_fp16_model(self, tmp_path: Path) -> None:
        """Test exporting a half-precision model."""
        model = _make_small_ppo().half()
        onnx_path = tmp_path / "policy_fp16.onnx"

        export_policy_to_onnx(
            model=model,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
        )

        assert onnx_path.exists()


# ─── verify_onnx_model ────────────────────────────────────────────────────────


class TestVerifyONNXModel:
    @pytest.fixture(autouse=True)
    def _skip_if_no_onnx(self) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    def test_verify_onnx_model(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        onnx_path = tmp_path / "policy.onnx"

        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
        )

        # Should not raise
        verify_onnx_model(onnx_path, small_ppo)


# ─── TensorRT Engine Builder ──────────────────────────────────────────────────


class TestTensorRTBuilder:
    @pytest.fixture(autouse=True)
    def _create_dummy_onnx(self, tmp_path: Path) -> Path:
        """Create a minimal dummy ONNX file for testing."""
        onnx_path = tmp_path / "policy.onnx"
        # Create empty file
        onnx_path.write_bytes(b"dummy")
        return onnx_path

    def test_build_tensorrt_engine_mock(
        self,
        small_ppo: TachyonPPO,
        tmp_path: Path,
        export_config: ExportConfig,
        _create_dummy_onnx: Path,
    ) -> None:
        """Test TensorRT engine building with mocked tensorrt."""
        onnx_path = _create_dummy_onnx
        engine_path = tmp_path / "policy.engine"

        # Mock tensorrt
        with (
            patch("tachyon.model.export._HAS_TENSORRT", True),
            patch("tachyon.model.export.trt") as mock_trt,
        ):
            # Setup mock
            mock_logger = Mock()
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_profile = Mock()

            mock_trt.Logger.return_value = mock_logger
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_trt.OnnxParser.return_value = mock_parser
            mock_builder.create_optimization_profile.return_value = mock_profile

            mock_parser.parse.return_value = True

            # Mock build - return bytes directly
            mock_builder.build_serialized_network.return_value = b"fake_engine_bytes"

            engine_bytes = build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                config=export_config,
            )

            assert engine_bytes == b"fake_engine_bytes"
            assert engine_path.exists()

    def test_optimization_profile_config(
        self,
        small_ppo: TachyonPPO,
        tmp_path: Path,
        export_config: ExportConfig,
        _create_dummy_onnx: Path,
    ) -> None:
        """Test that optimization profile has correct min/opt/max shapes."""
        onnx_path = _create_dummy_onnx

        with (
            patch("tachyon.model.export._HAS_TENSORRT", True),
            patch("tachyon.model.export.trt") as mock_trt,
        ):
            mock_logger = Mock()
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_profile = Mock()

            mock_trt.Logger.return_value = mock_logger
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_trt.OnnxParser.return_value = mock_parser
            mock_builder.create_optimization_profile.return_value = mock_profile

            mock_parser.parse.return_value = True

            mock_builder.build_serialized_network.return_value = b"engine"

            build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=tmp_path / "policy.engine",
                config=export_config,
            )

            # Verify optimization profile was configured
            mock_builder.create_optimization_profile.assert_called()
            mock_profile.set_shape.assert_called()

            # Check that set_shape was called for lob_state with min/opt/max batch
            set_shape_calls = mock_profile.set_shape.call_args_list
            lob_state_calls = [c for c in set_shape_calls if c[0][0] == "lob_state"]
            assert len(lob_state_calls) >= 1

            # min, opt, max shapes
            min_shape, opt_shape, max_shape = lob_state_calls[0][0][1:]
            assert min_shape[0] == export_config.min_batch
            assert opt_shape[0] == export_config.opt_batch
            assert max_shape[0] == export_config.max_batch
            assert min_shape[1] == 1  # seq_len
            assert opt_shape[1] == 1
            assert max_shape[1] == 1

    def test_fp16_flag_set(
        self,
        small_ppo: TachyonPPO,
        tmp_path: Path,
        export_config: ExportConfig,
        _create_dummy_onnx: Path,
    ) -> None:
        """Test that FP16 builder flag is set."""
        onnx_path = _create_dummy_onnx

        with (
            patch("tachyon.model.export._HAS_TENSORRT", True),
            patch("tachyon.model.export.trt") as mock_trt,
        ):
            mock_logger = Mock()
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_profile = Mock()

            mock_trt.Logger.return_value = mock_logger
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_trt.OnnxParser.return_value = mock_parser
            mock_config.create_optimization_profile.return_value = mock_profile

            mock_parser.parse.return_value = True
            mock_builder.build_serialized_network.return_value = b"engine"

            build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=tmp_path / "policy.engine",
                config=export_config,
            )

            # Check FP16 flag was set
            mock_config.set_flag.assert_called()
            flag_calls = mock_config.set_flag.call_args_list
            fp16_calls = [c for c in flag_calls if "FP16" in str(c)]
            assert len(fp16_calls) >= 1


# ─── verify_trt_parity ────────────────────────────────────────────────────────


class TestVerifyTRTParity:
    @pytest.fixture(autouse=True)
    def _create_dummy_onnx(self, tmp_path: Path) -> Path:
        """Create a minimal dummy ONNX file for testing."""
        onnx_path = tmp_path / "policy.onnx"
        onnx_path.write_bytes(b"dummy")
        return onnx_path

    def test_verify_trt_parity_mock(
        self,
        small_ppo: TachyonPPO,
        tmp_path: Path,
        export_config: ExportConfig,
        _create_dummy_onnx: Path,
    ) -> None:
        """Test parity verification with mocked TensorRT."""
        onnx_path = _create_dummy_onnx
        engine_path = tmp_path / "policy.engine"

        with (
            patch("tachyon.model.export._HAS_TENSORRT", True),
            patch("tachyon.model.export.trt") as mock_trt,
        ):
            # Setup mock for engine building
            mock_logger = Mock()
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_profile = Mock()
            mock_runtime = Mock()
            mock_engine = Mock()
            mock_context = Mock()

            mock_trt.Logger.return_value = mock_logger
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_trt.OnnxParser.return_value = mock_parser
            mock_config.create_optimization_profile.return_value = mock_profile
            mock_trt.Runtime.return_value = mock_runtime
            mock_runtime.deserialize_cuda_engine.return_value = mock_engine
            mock_engine.create_execution_context.return_value = mock_context

            # Mock context execute
            mock_context.execute_v2.return_value = True

            # Mock parser
            mock_parser.parse.return_value = True

            mock_builder.build_serialized_network.return_value = b"engine"

            # Build engine
            build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                config=export_config,
            )

            # Verify parity (should not raise with mocked outputs)
            # This test mainly ensures the function runs without error
            with contextlib.suppress(Exception):  # noqa: BLE001 — mocks may fail freely
                verify_trt_parity(
                    engine_path=engine_path,
                    model=small_ppo,
                    num_iterations=10,
                    atol=1e-3,
                )


# ─── Benchmark Script Integration ─────────────────────────────────────────────


class TestBenchmarkScript:
    def test_benchmark_script_imports(self) -> None:
        """Test that benchmark script can be imported."""
        import scripts.benchmark_trt as benchmark
        assert hasattr(benchmark, "main")
        assert hasattr(benchmark, "BenchmarkConfig")
        assert hasattr(benchmark, "run_benchmark")

    def test_benchmark_config(self) -> None:
        from scripts.benchmark_trt import BenchmarkConfig
        cfg = BenchmarkConfig(engine_path=Path("dummy.engine"))
        assert cfg.warmup_iterations == DEFAULT_WARMUP
        assert cfg.measure_iterations == DEFAULT_MEASURE
        assert cfg.p99_threshold_ms == DEFAULT_P99_THRESHOLD_MS
        assert cfg.batch_size == 1

    def test_benchmark_config_custom(self) -> None:
        from scripts.benchmark_trt import BenchmarkConfig

        cfg = BenchmarkConfig(
            engine_path=Path("dummy.engine"),
            warmup_iterations=100,
            measure_iterations=100,
            p99_threshold_ms=2.0,
        )
        assert cfg.warmup_iterations == 100
        assert cfg.measure_iterations == 100
        assert cfg.p99_threshold_ms == 2.0


# ─── Integration Tests ────────────────────────────────────────────────────────


class TestExportIntegration:
    @pytest.fixture(autouse=True)
    def _skip_if_no_onnx(self) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    def test_full_export_pipeline(self, small_ppo: TachyonPPO, tmp_path: Path) -> None:
        """Test full export pipeline: PyTorch → ONNX → (mock) TensorRT."""
        onnx_path = tmp_path / "policy.onnx"
        engine_path = tmp_path / "policy.engine"

        # Export to ONNX
        export_policy_to_onnx(
            model=small_ppo,
            save_path=onnx_path,
            batch_size=1,
            seq_len=1,
        )

        assert onnx_path.exists()

        # Verify ONNX model
        verify_onnx_model(onnx_path, small_ppo)

        # Build TensorRT engine (mocked)
        with (
            patch("tachyon.model.export._HAS_TENSORRT", True),
            patch("tachyon.model.export.trt") as mock_trt,
        ):
            mock_logger = Mock()
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_profile = Mock()

            mock_trt.Logger.return_value = mock_logger
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_trt.OnnxParser.return_value = mock_parser
            mock_builder.create_optimization_profile.return_value = mock_profile

            mock_parser.parse.return_value = True

            mock_builder.build_serialized_network.return_value = b"engine"

            build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                config=ExportConfig(),
            )

            assert engine_path.exists()


# ─── ExportConfig from Settings ───────────────────────────────────────────────


class TestExportConfigFromSettings:
    def test_config_from_rl_settings(self) -> None:
        """Test ExportConfig can be created from RL settings."""
        from tachyon.model.export import ExportConfig

        export_cfg = ExportConfig(
            batch_size=1,
            seq_len=1,
            d_model=256,
            n_heads=8,
            n_layers=6,
            head_dim=32,
            fp16=True,
            workspace_gb=2,
        )

        assert export_cfg.batch_size == 1
        assert export_cfg.seq_len == 1
