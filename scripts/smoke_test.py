"""Smoke test: verify the full training pipeline runs on CPU.

Tests:
1. Schema compilation with semantic types
2. SemanticConditioner creation and conditioning
3. Backbone forward/backward with semantic conditioning
4. Heterogeneous dataset creation with synthetic data
5. Masked canvas training step
6. Diffusion training step
7. DAG curriculum with 2 tiny nodes (fork-join)
8. Weight merging between nodes
9. Inference (observe + predict)

No real data needed. No GPU needed. Should complete in < 60 seconds.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from general_unified_world_model import WorldProjection, project, World
from general_unified_world_model.training.backbone import build_world_model
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, FieldMapping, build_mixed_dataloader,
)
from general_unified_world_model.training.diffusion import (
    DiffusionWorldModelTrainer, CosineNoiseSchedule,
)
from general_unified_world_model.training.dag_curriculum import (
    DAGCurriculumTrainer, TrainingNode,
)
from canvas_engineering import SemanticConditioner, compile_schema, ConnectivityPolicy


def timer(name):
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"  [{name}] {elapsed:.2f}s")
    return _Timer()


def test_schema_and_semantic():
    """Test 1: Schema compilation auto-populates semantic types."""
    print("\n1. Schema + Semantic Types")
    with timer("compile full world"):
        world = World()
        bound = compile_schema(
            world, T=1, H=160, W=160, d_model=32,
            connectivity=ConnectivityPolicy(intra="dense", parent_child="hub_spoke"),
        )
    print(f"  {len(bound.field_names)} fields, {bound.layout.num_positions} positions")

    # Check semantic types auto-generated
    for name in list(bound.field_names)[:5]:
        spec = bound[name].spec
        assert spec.semantic_type is not None, f"Missing semantic_type for {name}"
    print(f"  Semantic types: OK (sample: '{bound[bound.field_names[0]].spec.semantic_type}')")


def test_semantic_conditioner():
    """Test 2: SemanticConditioner with random embeddings."""
    print("\n2. SemanticConditioner")
    proj = WorldProjection(include=["financial", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)

    embed_dim = 16

    def random_embed(texts):
        return [torch.randn(embed_dim).tolist() for _ in texts]

    with timer("build conditioner"):
        conditioner = bound.build_semantic_conditioner(
            random_embed, embed_dim=embed_dim
        )
    print(f"  {conditioner.n_regions} regions, {sum(p.numel() for p in conditioner.parameters())} params")

    # Test conditioning
    canvas = torch.zeros(2, bound.layout.num_positions, 32)
    conditioned = conditioner.condition_canvas(canvas, bound.layout)
    assert conditioned.shape == canvas.shape
    assert conditioned.abs().sum() > 0
    print("  Conditioning: OK")


def test_backbone_with_conditioning():
    """Test 3: Backbone forward/backward with semantic conditioning."""
    print("\n3. Backbone + Semantic Conditioning")
    proj = WorldProjection(include=["financial", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)

    def random_embed(texts):
        return [torch.randn(16).tolist() for _ in texts]

    conditioner = bound.build_semantic_conditioner(random_embed, embed_dim=16)

    with timer("build model"):
        backbone = build_world_model(bound, n_layers=2, n_loops=2)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"  Backbone: {n_params:,} params")

    # Forward pass
    canvas = torch.randn(2, bound.layout.num_positions, 32)
    canvas = conditioner.condition_canvas(canvas, bound.layout)

    attn_mask = None
    if bound.topology:
        attn_mask = bound.topology.to_additive_mask(bound.layout)

    with timer("forward + backward"):
        out = backbone(canvas, mask=attn_mask)
        loss = out.sum()
        loss.backward()

    assert out.shape == canvas.shape
    print("  Forward/backward: OK")


def test_heterogeneous_training():
    """Test 4: Masked canvas training with synthetic data."""
    print("\n4. Heterogeneous Training (synthetic)")
    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)

    # Create synthetic data
    spec = DatasetSpec(
        name="synthetic",
        mappings=[
            FieldMapping("ten_year", "financial.yield_curves.ten_year"),
            FieldMapping("two_year", "financial.yield_curves.two_year"),
        ],
    )
    data = {
        "ten_year": torch.randn(50),
        "two_year": torch.randn(50),
    }

    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    encoder = FieldEncoder(bound)
    decoder = FieldDecoder(bound)

    all_params = (
        list(backbone.parameters())
        + list(encoder.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    trainer = MaskedCanvasTrainer(
        bound, backbone, encoder, decoder, optimizer, device="cpu"
    )

    dataloader = build_mixed_dataloader(bound, [(spec, data)], batch_size=4)

    with timer("3 training steps"):
        for i, batch in enumerate(dataloader):
            metrics = trainer.train_step(batch)
            if i >= 2:
                break

    print(f"  Loss: {metrics['loss']:.4f}, Coverage: {metrics['coverage']:.1%}")


def test_diffusion_training():
    """Test 5: Diffusion training step."""
    print("\n5. Diffusion Training")
    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)

    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    schedule = CosineNoiseSchedule(n_steps=100)
    trainer = DiffusionWorldModelTrainer(
        bound, backbone, schedule, device="cpu"
    )

    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    x_clean = torch.randn(2, bound.layout.num_positions, 32)
    presence = torch.ones(2, bound.layout.num_positions)

    with timer("diffusion step"):
        metrics = trainer.train_step(x_clean, presence, optimizer)

    print(f"  Loss: {metrics['loss']:.4f}")


def test_dag_curriculum():
    """Test 6: DAG curriculum with tiny fork-join."""
    print("\n6. DAG Curriculum (fork-join smoke test)")

    import tempfile
    checkpoint_dir = tempfile.mkdtemp()

    # Two foundation nodes → one merge node
    nodes = [
        TrainingNode(
            name="tiny_finance",
            description="Minimal financial model",
            include=["financial.yield_curves"],
            H=24, W=24, d_model=32, n_layers=2, n_loops=1,
            n_steps=3,
            data_sources=["synthetic_finance"],
        ),
        TrainingNode(
            name="tiny_macro",
            description="Minimal macro model",
            include=["regime"],
            H=24, W=24, d_model=32, n_layers=2, n_loops=1,
            n_steps=3,
            data_sources=["synthetic_macro"],
        ),
        TrainingNode(
            name="tiny_merged",
            description="Merge finance + macro",
            include=["financial.yield_curves", "regime"],
            parents=["tiny_finance", "tiny_macro"],
            H=24, W=24, d_model=32, n_layers=2, n_loops=1,
            n_steps=3,
            data_sources=["synthetic_finance", "synthetic_macro"],
        ),
    ]

    # Synthetic data sources
    finance_spec = DatasetSpec(
        name="synthetic_finance",
        mappings=[
            FieldMapping("ten_year", "financial.yield_curves.ten_year"),
        ],
    )
    macro_spec = DatasetSpec(
        name="synthetic_macro",
        mappings=[
            FieldMapping("regime_val", "regime.growth_regime"),
        ],
    )

    data_sources = {
        "synthetic_finance": (finance_spec, {"ten_year": torch.randn(20)}),
        "synthetic_macro": (macro_spec, {"regime_val": torch.randn(20)}),
    }

    with timer("DAG training"):
        trainer = DAGCurriculumTrainer(
            nodes=nodes,
            data_sources=data_sources,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
            embed_dim=16,
        )
        trainer.run()

    # Verify checkpoints
    ckpt_files = list(Path(checkpoint_dir).glob("*.pt"))
    print(f"  Checkpoints: {len(ckpt_files)} files")
    assert len(ckpt_files) >= 3, f"Expected 3 checkpoints, got {len(ckpt_files)}"

    # Verify metadata
    import json
    with open(Path(checkpoint_dir) / "dag_metadata.json") as f:
        meta = json.load(f)
    assert len(meta["nodes"]) == 3
    print(f"  DAG metadata: {len(meta['nodes'])} nodes, order: {meta['order']}")


def test_inference():
    """Test 7: WorldModel observe + predict."""
    print("\n7. Inference")
    from general_unified_world_model.inference import WorldModel

    proj = WorldProjection(include=["financial.yield_curves", "regime"])
    bound = project(proj, T=1, H=24, W=24, d_model=32)

    # Create a mock model (untrained)
    backbone = build_world_model(bound, n_layers=2, n_loops=1)
    encoder = FieldEncoder(bound)
    decoder = FieldDecoder(bound)

    with timer("inference pipeline"):
        model = WorldModel(bound, backbone, encoder, decoder, device="cpu")
        model.observe("financial.yield_curves.ten_year", 4.25)
        predictions = model.predict(n_steps=5)

    assert len(predictions) > 0
    print(f"  Predicted {len(predictions)} fields")


def test_fog_regions():
    """Test 8: Fog regions for partial projections."""
    print("\n8. Fog Regions")
    with timer("fog projection"):
        proj = WorldProjection(include=["financial.yield_curves", "regime"], fog=True)
        bound = project(proj, T=1, H=24, W=24, d_model=32)

    fog_fields = [n for n in bound.field_names if "_fog_" in n]
    regular_fields = [n for n in bound.field_names if "_fog_" not in n]
    print(f"  {len(regular_fields)} modeled fields, {len(fog_fields)} fog regions")
    assert len(fog_fields) > 0, "Expected fog fields"

    # Each fog field should have connectivity
    fog_conns = [c for c in bound.topology.connections
                 if "_fog_" in c.src or "_fog_" in c.dst]
    print(f"  {len(fog_conns)} fog connections")
    assert len(fog_conns) > 0

    # Fog semantic types should be descriptive
    for name in fog_fields[:2]:
        print(f"  {name}: {bound[name].spec.semantic_type}")


def test_nl_curriculum():
    """Test 9: Natural language curriculum specification."""
    print("\n9. NL Curriculum")
    from general_unified_world_model.training.dag_curriculum import (
        resolve_subject, CurriculumSpec, STANDARD_CURRICULUM,
    )

    with timer("resolve subjects"):
        tests = {
            "Financial markets": "financial",
            "GDP and inflation": "macro",
            "Oil and commodities": "resource",
        }
        for desc, expected in tests.items():
            paths = resolve_subject(desc)
            matched = any(expected in p for p in paths)
            assert matched, f"Expected '{expected}' in resolve('{desc}'), got {paths}"

    with timer("build standard curriculum"):
        nodes = STANDARD_CURRICULUM.to_training_nodes()

    print(f"  Standard curriculum: {len(nodes)} nodes, 4 stages")
    assert len(nodes) == 12

    # Check DAG structure
    foundations = [n for n in nodes if n.parents == []]
    print(f"  Foundations: {len(foundations)} parallel nodes")
    assert len(foundations) == 6


def main():
    print("=" * 60)
    print("SMOKE TEST: General Unified World Model")
    print("=" * 60)

    start = time.time()

    test_schema_and_semantic()
    test_semantic_conditioner()
    test_backbone_with_conditioning()
    test_heterogeneous_training()
    test_diffusion_training()
    test_dag_curriculum()
    test_inference()
    test_fog_regions()
    test_nl_curriculum()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"ALL SMOKE TESTS PASSED in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
