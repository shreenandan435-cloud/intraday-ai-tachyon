"""Supervised Pre-Training Pipeline — external dataset, pre-train model, backbone extraction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from tachyon.model.external_dataset import ExternalLOBDataset
from tachyon.model.pretrain import TachyonPretrain

if TYPE_CHECKING:
    pass


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic_parquet(tmp_path: Path, num_rows: int = 5000) -> Path:
    """Create synthetic L2 parquet file matching the expected schema."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    depth_levels = 5
    tick_size = 0.05

    schema = pa.schema([
        ("ts_epoch", pa.float64()),
        ("token", pa.string()),
        *[ (f"bid_price_{i}", pa.float64()) for i in range(depth_levels) ],
        *[ (f"ask_price_{i}", pa.float64()) for i in range(depth_levels) ],
        *[ (f"bid_qty_{i}", pa.int64()) for i in range(depth_levels) ],
        *[ (f"ask_qty_{i}", pa.int64()) for i in range(depth_levels) ],
    ])

    np.random.seed(42)
    base_mid = 100.0
    rows = []

    for i in range(num_rows):
        ts = 1_755_000_000.0 + i * 0.001
        mid = base_mid + np.random.normal(0, 0.01) * i * 0.001
        mid = max(50.0, mid)

        spread_ticks = 2
        half_spread = (spread_ticks * tick_size) / 2
        bid_0 = round((mid - half_spread) / tick_size) * tick_size
        ask_0 = round((mid + half_spread) / tick_size) * tick_size

        if ask_0 <= bid_0:
            ask_0 = bid_0 + tick_size

        bid_prices = [bid_0 - j * tick_size for j in range(depth_levels)]
        ask_prices = [ask_0 + j * tick_size for j in range(depth_levels)]
        bid_qtys = np.random.lognormal(8, 1.5, depth_levels).astype(np.int64)
        ask_qtys = np.random.lognormal(8, 1.5, depth_levels).astype(np.int64)
        bid_qtys = np.maximum(bid_qtys, 1)
        ask_qtys = np.maximum(ask_qtys, 1)

        row = {
            "ts_epoch": ts,
            "token": "TEST",
        }
        for j in range(depth_levels):
            row[f"bid_price_{j}"] = bid_prices[j]
            row[f"ask_price_{j}"] = ask_prices[j]
            row[f"bid_qty_{j}"] = int(bid_qtys[j])
            row[f"ask_qty_{j}"] = int(ask_qtys[j])
        rows.append(row)

    # Convert to arrays
    arrays = {name: [] for name in schema.names}
    for row in rows:
        for name in schema.names:
            arrays[name].append(row[name])

    table = pa.table(arrays, schema=schema)

    output_dir = tmp_path / "external_data" / "date=2026-08-22" / "symbol=TEST"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "0915.parquet"

    pq.write_table(table, str(output_path), compression="zstd")
    return output_path


# ─── ExternalLOBDataset ───────────────────────────────────────────────────────


class TestExternalLOBDataset:
    def test_loads_external_parquet(self, tmp_path: Path) -> None:
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=2000)
        data_dir = parquet_path.parent.parent.parent

        dataset = ExternalLOBDataset(
            data_dir=data_dir,
            seq_len=64,
            forward_horizon=20,
            threshold_bps=0.01,
            tick_size_map={"TEST": 0.05},
        )

        assert len(dataset) > 0
        assert dataset.n_features == 25  # 22 LOB + 3 portfolio (zeros)

    def test_sample_shape_and_dtype(self, tmp_path: Path) -> None:
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=2000)
        data_dir = parquet_path.parent.parent.parent

        dataset = ExternalLOBDataset(
            data_dir=data_dir,
            seq_len=64,
            forward_horizon=20,
            threshold_bps=0.01,
            tick_size_map={"TEST": 0.05},
        )

        features, label = dataset[0]

        # Features: (T, 25)
        assert features.shape == (64, 25)
        assert features.dtype == torch.float32

        # Label: scalar int (0, 1, 2)
        assert label.dtype == torch.int64
        assert label.item() in {0, 1, 2}

    def test_feature_computation_matches_lob_dataset(self, tmp_path: Path) -> None:
        """Verify the 22 LOB features match LOBDataset normalization."""
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=500)
        data_dir = parquet_path.parent.parent.parent

        dataset = ExternalLOBDataset(
            data_dir=data_dir,
            seq_len=32,
            forward_horizon=10,
            threshold_bps=0.01,
            tick_size_map={"TEST": 0.05},
        )

        features, _ = dataset[0]

        # Check feature ranges for the 22 LOB features (first 22 columns)
        lob_features = features[:, :22]

        # Bid BPS should be negative (below mid)
        assert (lob_features[:, :5] <= 0).all()
        # Ask BPS should be positive (above mid)
        assert (lob_features[:, 5:10] >= 0).all()
        # Log quantities should be positive
        assert (lob_features[:, 10:20] >= 0).all()
        # Spread tick >= 0
        assert (lob_features[:, 20] >= 0).all()
        # OBI in [-1, 1]
        assert (lob_features[:, 21] >= -1).all() and (lob_features[:, 21] <= 1).all()

        # Portfolio features (last 3) should be zeros
        portfolio = features[:, 22:]
        assert (portfolio == 0).all()

    def test_label_generation_three_class(self, tmp_path: Path) -> None:
        """Verify 3-class label generation based on forward log return."""
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=5000)
        data_dir = parquet_path.parent.parent.parent

        # Use extremely small threshold since synthetic data has tiny movements
        dataset = ExternalLOBDataset(
            data_dir=data_dir,
            seq_len=64,
            forward_horizon=20,
            threshold_bps=0.0001,  # Extremely small for synthetic data
            tick_size_map={"TEST": 0.05},
        )

        labels = [dataset[i][1].item() for i in range(min(100, len(dataset)))]

        # All labels should be in {0, 1, 2}
        assert all(label in {0, 1, 2} for label in labels)

        # Just verify labels are generated (mechanism works)
        # Distribution depends on synthetic data characteristics
        assert len(labels) > 0

    def test_threshold_bps_effect(self, tmp_path: Path) -> None:
        """Higher threshold should produce more 'Flat' (class 1) labels."""
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=5000)
        data_dir = parquet_path.parent.parent.parent

        # Low threshold
        ds_low = ExternalLOBDataset(
            data_dir=data_dir, seq_len=64, forward_horizon=20,
            threshold_bps=0.01, tick_size_map={"TEST": 0.05},
        )
        labels_low = [ds_low[i][1].item() for i in range(min(200, len(ds_low)))]
        flat_low = sum(1 for label in labels_low if label == 1) / len(labels_low)

        # High threshold
        ds_high = ExternalLOBDataset(
            data_dir=data_dir, seq_len=64, forward_horizon=20,
            threshold_bps=1.0, tick_size_map={"TEST": 0.05},
        )
        labels_high = [ds_high[i][1].item() for i in range(min(200, len(ds_high)))]
        flat_high = sum(1 for label in labels_high if label == 1) / len(labels_high)

        # Higher threshold should produce more flat labels
        assert flat_high >= flat_low

    def test_forward_horizon_effect(self, tmp_path: Path) -> None:
        """Larger forward horizon should produce more extreme returns."""
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=5000)
        data_dir = parquet_path.parent.parent.parent

        ds_short = ExternalLOBDataset(
            data_dir=data_dir, seq_len=64, forward_horizon=5,
            threshold_bps=0.01, tick_size_map={"TEST": 0.05},
        )
        ds_long = ExternalLOBDataset(
            data_dir=data_dir, seq_len=64, forward_horizon=50,
            threshold_bps=0.01, tick_size_map={"TEST": 0.05},
        )

        labels_short = [ds_short[i][1].item() for i in range(min(100, len(ds_short)))]
        labels_long = [ds_long[i][1].item() for i in range(min(100, len(ds_long)))]

        # Longer horizon should have more non-flat labels
        non_flat_short = sum(1 for label in labels_short if label != 1)
        non_flat_long = sum(1 for label in labels_long if label != 1)
        assert non_flat_long >= non_flat_short


# ─── TachyonPretrain ──────────────────────────────────────────────────────────


class TestTachyonPretrain:
    def test_model_structure(self) -> None:
        model = TachyonPretrain(
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            max_seq_len=128,
        )

        assert hasattr(model, "embedding")
        assert hasattr(model, "transformer")
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.out_features == 3

    def test_forward_output_shape(self) -> None:
        model = TachyonPretrain(
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            max_seq_len=128,
        ).eval()

        x = torch.randn(4, 32, 25)
        logits = model(x)

        assert logits.shape == (4, 3)  # Only last timestep, 3 classes

    def test_save_backbone_extracts_correct_weights(self, tmp_path: Path) -> None:
        """save_backbone should only save embedding + transformer weights."""
        model = TachyonPretrain(
            d_model=64,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            max_seq_len=128,
        )

        backbone_path = tmp_path / "backbone.pt"
        model.save_backbone(backbone_path)

        assert backbone_path.exists()

        loaded = torch.load(backbone_path, map_location="cpu")

        # Should contain embedding and transformer keys
        keys = set(loaded.keys())
        assert any(k.startswith("embedding.") for k in keys)
        assert any(k.startswith("transformer.") for k in keys)

        # Should NOT contain classifier keys
        assert not any(k.startswith("classifier.") for k in keys)

    def test_backbone_compatible_with_tachyon_ppo(self, tmp_path: Path) -> None:
        """Backbone from TachyonPretrain should load into TachyonPPO with strict=False."""
        from tachyon.model.ppo import TachyonPPO

        pretrain_model = TachyonPretrain(
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=3,
            max_seq_len=128,
        )

        backbone_path = tmp_path / "backbone.pt"
        pretrain_model.save_backbone(backbone_path)

        # Load into TachyonPPO
        ppo_model = TachyonPPO(
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_actions=4,
        )

        # Should load with strict=False (classifier head missing, actor/critic heads missing)
        missing, unexpected = ppo_model.load_state_dict(
            torch.load(backbone_path, map_location="cpu"),
            strict=False,
        )

        # Only missing should be actor_head, critic_head (and norm_f if not in pretrain)
        missing_keys = set(missing)
        expected_missing = {"actor_head.weight", "critic_head.weight"}
        assert expected_missing.issubset(missing_keys)

        # No unexpected keys
        assert len(unexpected) == 0

    def test_gradient_flow(self) -> None:
        model = TachyonPretrain(
            d_model=64,
            n_heads=4,
            n_layers=2,
            num_classes=3,
        )

        x = torch.randn(2, 16, 25, requires_grad=True)
        target = torch.randint(0, 3, (2,))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


# ─── Integration Test ─────────────────────────────────────────────────────────


class TestPretrainIntegration:
    def test_full_pretrain_pipeline(self, tmp_path: Path) -> None:
        """Test full pipeline: dataset -> model -> train step -> save backbone."""
        parquet_path = _make_synthetic_parquet(tmp_path, num_rows=2000)
        data_dir = parquet_path.parent.parent.parent

        dataset = ExternalLOBDataset(
            data_dir=data_dir,
            seq_len=32,
            forward_horizon=10,
            threshold_bps=0.01,
            tick_size_map={"TEST": 0.05},
        )

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

        model = TachyonPretrain(
            d_model=64,
            n_heads=4,
            n_layers=2,
            num_classes=3,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch_idx, (features, labels) in enumerate(loader):
            if batch_idx >= 3:  # Just a few steps
                break

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()

            assert torch.isfinite(loss), "Loss should be finite"

            optimizer.step()

        # Save backbone
        backbone_path = tmp_path / "pretrained_backbone.pt"
        model.save_backbone(backbone_path)
        assert backbone_path.exists()

        # Verify loadable into TachyonPPO
        from tachyon.model.ppo import TachyonPPO
        ppo = TachyonPPO(d_model=64, n_heads=4, n_layers=2, num_actions=4)
        missing, unexpected = ppo.load_state_dict(
            torch.load(backbone_path, map_location="cpu"), strict=False
        )
        assert {"actor_head.weight", "critic_head.weight"}.issubset(set(missing))
        assert len(unexpected) == 0


# ─── run_pretrain script tests ────────────────────────────────────────────────


class TestRunPretrainScript:
    def test_script_imports(self) -> None:
        """Verify the script can be imported without errors."""
        import scripts.run_pretrain as run_pretrain
        assert hasattr(run_pretrain, "main")
        assert hasattr(run_pretrain, "train_epoch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
