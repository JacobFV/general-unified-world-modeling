"""Tests for the DAG-based curriculum trainer."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from general_unified_world_model.projection.subset import project
from general_unified_world_model.training.backbone import build_world_model
from general_unified_world_model.training.heterogeneous import (
    DatasetSpec, DataSource, InputSpec, OutputSpec, FieldEncoder, FieldDecoder,
)
from general_unified_world_model.training.dag_curriculum import (
    TrainingNode, DAGCurriculumTrainer, DAGCheckpoint,
    TIER_0_FOUNDATION, TIER_1_CROSS_DOMAIN, TIER_2_COMPLEX,
    TIER_3_INTEGRATION, STANDARD_DAG,
)


# ── TrainingNode ──────────────────────────────────────────────────────────

class TestTrainingNode:
    def test_defaults(self):
        node = TrainingNode(
            name="test", description="A test node",
            include=["financial"],
        )
        assert node.parents == []
        assert node.H is None  # auto-sized by project()
        assert node.d_model == 64
        assert node.n_steps == 5000
        assert node.entities == {}
        assert node.data_sources == []

    def test_with_parents(self):
        node = TrainingNode(
            name="child", description="Child node",
            include=["regime"],
            parents=["parent_a", "parent_b"],
        )
        assert node.parents == ["parent_a", "parent_b"]

    def test_with_entities(self):
        node = TrainingNode(
            name="corp", description="Corporate",
            include=["financial.equities", "regime"],
            entities={"firm_AAPL": "Business", "country_jp": "Country"},
        )
        assert node.entities == {"firm_AAPL": "Business", "country_jp": "Country"}


# ── Standard DAG definitions ──────────────────────────────────────────────

class TestStandardDAG:
    def test_tier_0_count(self):
        assert len(TIER_0_FOUNDATION) == 6

    def test_tier_0_no_parents(self):
        for node in TIER_0_FOUNDATION:
            assert node.parents == [], f"{node.name} should have no parents"

    def test_tier_1_has_parents(self):
        for node in TIER_1_CROSS_DOMAIN:
            assert len(node.parents) >= 2, f"{node.name} should have >= 2 parents"

    def test_tier_2_has_parents(self):
        for node in TIER_2_COMPLEX:
            assert len(node.parents) >= 1

    def test_tier_2_has_entities(self):
        corp = next(n for n in TIER_2_COMPLEX if n.name == "corporate_strategy")
        assert "firm_AAPL" in corp.entities
        assert corp.entities["firm_AAPL"] == "Business"

    def test_tier_3_full_integration(self):
        assert len(TIER_3_INTEGRATION) == 1
        assert TIER_3_INTEGRATION[0].name == "full_integration"
        assert len(TIER_3_INTEGRATION[0].parents) >= 2

    def test_standard_dag_total(self):
        assert len(STANDARD_DAG) == 12

    def test_all_parents_reference_existing_nodes(self):
        names = {n.name for n in STANDARD_DAG}
        for node in STANDARD_DAG:
            for parent in node.parents:
                assert parent in names, f"{node.name} references unknown parent '{parent}'"


# ── DAGCheckpoint ─────────────────────────────────────────────────────────

class TestDAGCheckpoint:
    def test_to_dict(self):
        ckpt = DAGCheckpoint(
            node_name="test", step=100, timestamp=1.0,
            loss=0.5, n_fields=10, n_params=1000, parents=["a"],
        )
        d = ckpt.to_dict()
        assert d["node_name"] == "test"
        assert d["loss"] == 0.5
        assert d["parents"] == ["a"]


# ── Topological sort ──────────────────────────────────────────────────────

class TestTopoSort:
    def test_linear_chain(self):
        nodes = [
            TrainingNode(name="a", description="", include=["financial"]),
            TrainingNode(name="b", description="", include=["regime"], parents=["a"]),
            TrainingNode(name="c", description="", include=["resources"], parents=["b"]),
        ]
        trainer = DAGCurriculumTrainer(nodes=nodes, data_sources={}, checkpoint_dir=tempfile.mkdtemp())
        order = trainer._topo_sort()
        assert order == ["a", "b", "c"]

    def test_diamond(self):
        nodes = [
            TrainingNode(name="root", description="", include=["financial"]),
            TrainingNode(name="left", description="", include=["regime"], parents=["root"]),
            TrainingNode(name="right", description="", include=["resources"], parents=["root"]),
            TrainingNode(name="merge", description="", include=["financial", "regime"], parents=["left", "right"]),
        ]
        trainer = DAGCurriculumTrainer(nodes=nodes, data_sources={}, checkpoint_dir=tempfile.mkdtemp())
        order = trainer._topo_sort()
        assert order[0] == "root"
        assert order[-1] == "merge"
        assert set(order[1:3]) == {"left", "right"}

    def test_no_parents(self):
        nodes = [
            TrainingNode(name="a", description="", include=["financial"]),
            TrainingNode(name="b", description="", include=["regime"]),
        ]
        trainer = DAGCurriculumTrainer(nodes=nodes, data_sources={}, checkpoint_dir=tempfile.mkdtemp())
        order = trainer._topo_sort()
        assert set(order) == {"a", "b"}


# ── Weight merging ────────────────────────────────────────────────────────

class TestWeightMerging:
    def test_merge_identical_backbones(self):
        """Merging two identical backbones should produce the same weights."""
        bound = project(include=["financial.yield_curves"], T=1, H=24, W=24, d_model=32)

        bb1 = build_world_model(bound, n_layers=2, n_loops=1)
        bb2 = build_world_model(bound, n_layers=2, n_loops=1)

        # Copy bb1 weights to bb2
        bb2.load_state_dict(bb1.state_dict())

        nodes = [
            TrainingNode(name="a", description="", include=["financial.yield_curves"]),
            TrainingNode(name="b", description="", include=["financial.yield_curves"]),
        ]
        trainer = DAGCurriculumTrainer(nodes=nodes, data_sources={}, checkpoint_dir=tempfile.mkdtemp())
        trainer.trained["a"] = {"backbone": bb1}
        trainer.trained["b"] = {"backbone": bb2}

        target = build_world_model(bound, n_layers=2, n_loops=1)
        trainer._merge_backbones(["a", "b"], target)

        # Target should have same weights as bb1 (since bb1 == bb2)
        for key in bb1.state_dict():
            if key in target.state_dict() and bb1.state_dict()[key].shape == target.state_dict()[key].shape:
                assert torch.allclose(
                    target.state_dict()[key], bb1.state_dict()[key], atol=1e-6
                ), f"Mismatch in {key}"

    def test_merge_averages_weights(self):
        """Merging two backbones should average their parameters."""
        bound = project(include=["financial.yield_curves"], T=1, H=24, W=24, d_model=32)

        bb1 = build_world_model(bound, n_layers=2, n_loops=1)
        bb2 = build_world_model(bound, n_layers=2, n_loops=1)

        # Set known weights on actual parameters (not buffers like pos_enc)
        param_names = {n for n, _ in bb1.named_parameters()}
        with torch.no_grad():
            for p in bb1.parameters():
                p.fill_(2.0)
            for p in bb2.parameters():
                p.fill_(4.0)

        nodes = [
            TrainingNode(name="a", description="", include=["financial.yield_curves"]),
            TrainingNode(name="b", description="", include=["financial.yield_curves"]),
        ]
        trainer = DAGCurriculumTrainer(nodes=nodes, data_sources={}, checkpoint_dir=tempfile.mkdtemp())
        trainer.trained["a"] = {"backbone": bb1}
        trainer.trained["b"] = {"backbone": bb2}

        target = build_world_model(bound, n_layers=2, n_loops=1)
        trainer._merge_backbones(["a", "b"], target)

        # Trainable params should be averaged (3.0); buffers may not be
        for key, param in target.state_dict().items():
            if key not in param_names:
                continue  # skip buffers like positional encoding
            if key in bb1.state_dict() and bb1.state_dict()[key].shape == param.shape:
                assert torch.allclose(param, torch.full_like(param, 3.0), atol=1e-5), \
                    f"{key}: expected 3.0, got {param.flatten()[:3]}"


# ── End-to-end training ──────────────────────────────────────────────────

class TestEndToEndTraining:
    def test_single_node_no_data(self):
        """Training a node with no data sources should still checkpoint."""
        checkpoint_dir = tempfile.mkdtemp()
        nodes = [
            TrainingNode(
                name="empty", description="No data",
                include=["regime"],
                H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                n_steps=1, data_sources=[],
            ),
        ]
        trainer = DAGCurriculumTrainer(
            nodes=nodes, data_sources={}, checkpoint_dir=checkpoint_dir,
        )
        trainer.run()

        ckpt_path = Path(checkpoint_dir) / "empty.pt"
        assert ckpt_path.exists()

        meta_path = Path(checkpoint_dir) / "dag_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert len(meta["nodes"]) == 1
        assert meta["order"] == ["empty"]

    def test_fork_join_with_data(self):
        """Fork-join with synthetic data should train and merge."""
        checkpoint_dir = tempfile.mkdtemp()

        nodes = [
            TrainingNode(
                name="fork_a", description="Fork A",
                include=["financial.yield_curves"],
                H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                n_steps=2, data_sources=["synth_a"],
            ),
            TrainingNode(
                name="fork_b", description="Fork B",
                include=["regime"],
                H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                n_steps=2, data_sources=["synth_b"],
            ),
            TrainingNode(
                name="join", description="Join",
                include=["financial.yield_curves", "regime"],
                parents=["fork_a", "fork_b"],
                H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                n_steps=2, data_sources=["synth_a", "synth_b"],
            ),
        ]

        spec_a = DatasetSpec(
            name="synth_a",
            input_specs=[InputSpec(key="ten_year", semantic_type="10-year yield", field_path="financial.yield_curves.ten_year")],
            output_specs=[OutputSpec(key="ten_year", semantic_type="10-year yield", field_path="financial.yield_curves.ten_year")],
        )
        spec_b = DatasetSpec(
            name="synth_b",
            input_specs=[InputSpec(key="regime_val", semantic_type="growth regime", field_path="regime.growth_regime")],
            output_specs=[OutputSpec(key="regime_val", semantic_type="growth regime", field_path="regime.growth_regime")],
        )

        data_sources = {
            "synth_a": DataSource(spec=spec_a, data={"ten_year": torch.randn(20)}),
            "synth_b": DataSource(spec=spec_b, data={"regime_val": torch.randn(20)}),
        }

        trainer = DAGCurriculumTrainer(
            nodes=nodes, data_sources=data_sources,
            checkpoint_dir=checkpoint_dir,
        )
        trainer.run()

        # All 3 checkpoints should exist
        ckpts = list(Path(checkpoint_dir).glob("*.pt"))
        assert len(ckpts) >= 3

        # Verify metadata
        with open(Path(checkpoint_dir) / "dag_metadata.json") as f:
            meta = json.load(f)
        assert len(meta["nodes"]) == 3
        assert meta["order"][-1] == "join"

    def test_run_specific_nodes(self):
        """Running specific nodes should only train those + ancestors."""
        checkpoint_dir = tempfile.mkdtemp()

        nodes = [
            TrainingNode(name="root", description="", include=["regime"],
                         H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                         n_steps=1, data_sources=[]),
            TrainingNode(name="child", description="", include=["regime"],
                         parents=["root"],
                         H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                         n_steps=1, data_sources=[]),
            TrainingNode(name="other", description="", include=["financial"],
                         H=24, W=24, d_model=32, n_layers=2, n_loops=1,
                         n_steps=1, data_sources=[]),
        ]

        trainer = DAGCurriculumTrainer(
            nodes=nodes, data_sources={}, checkpoint_dir=checkpoint_dir,
        )
        trainer.run(nodes=["child"])

        # Should train root + child, not "other"
        assert (Path(checkpoint_dir) / "root.pt").exists()
        assert (Path(checkpoint_dir) / "child.pt").exists()
        assert not (Path(checkpoint_dir) / "other.pt").exists()


# ── run_tier ──────────────────────────────────────────────────────────────

class TestRunTier:
    def test_invalid_tier(self):
        trainer = DAGCurriculumTrainer(
            nodes=STANDARD_DAG, data_sources={},
            checkpoint_dir=tempfile.mkdtemp(),
        )
        with pytest.raises(ValueError, match="Unknown tier"):
            trainer.run_tier(5)
