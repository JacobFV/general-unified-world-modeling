"""DAG-based training curriculum for the General Unified World Model.

The training curriculum is a directed acyclic graph (DAG) where:
- **Fork nodes** copy model weights and train each copy on different
  domain-specific projections in parallel
- **Join nodes** merge the trained copies by weight averaging (model souping)

This combines the speed of parallel training with the breadth of cross-domain
integration. The semantic embedding conditioner makes this work: because
dynamics are conditioned on field identity (not hardcoded heads), merged
weights can handle any field type.

Each training node is described linguistically. The description generates
the WorldProjection via llm_project() or explicit field paths.

Example DAG:
                        ┌─ basic_finance ──────┐
                        │                       │
    init ───────────────├─ basic_economics ─────├── merge ──┬─ econ_drives_finance ──┐
                        │                       │           │                        │
                        ├─ basic_politics ──────┤           └─ geopolitics_commodities┤
                        │                       │                                     │
                        └─ basic_resources ─────┘                                     │
                                                            ┌─────────────────────────┘
                                                            │
                                                        merge ── full_integration
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from general_unified_world_model.projection.subset import WorldProjection, project
from general_unified_world_model.training.backbone import build_world_model, WorldModelBackbone
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, build_mixed_dataloader,
)


# ── DAG Node Definition ───────────────────────────────────────────────────

@dataclass
class TrainingNode:
    """A single node in the training DAG.

    Args:
        name: Human-readable name (e.g. "basic_finance").
        description: Linguistic description of what this node trains.
            Used for logging and optionally for llm_project().
        include: Explicit world model field paths to include.
        parents: Names of parent nodes (whose merged weights initialize this node).
        H, W: Canvas dimensions.
        d_model: Latent dimension.
        n_layers: Transformer depth.
        n_loops: Looped attention iterations.
        n_steps: Number of training steps.
        lr: Learning rate.
        batch_size: Batch size.
        data_sources: Names of data source keys to use.
        firms: Dynamic firm entities.
        individuals: Dynamic individual entities.
        countries: Additional country codes.
    """
    name: str
    description: str
    include: list[str]
    parents: list[str] = dc_field(default_factory=list)
    H: int = 32
    W: int = 32
    d_model: int = 64
    n_layers: int = 4
    n_loops: int = 3
    n_steps: int = 5000
    lr: float = 1e-4
    batch_size: int = 32
    data_sources: list[str] = dc_field(default_factory=list)
    firms: list[str] = dc_field(default_factory=list)
    individuals: list[str] = dc_field(default_factory=list)
    countries: list[str] = dc_field(default_factory=list)


# ── Standard DAG Definitions ──────────────────────────────────────────────

TIER_0_FOUNDATION = [
    TrainingNode(
        name="basic_finance",
        description="Core financial markets: yield curves, credit spreads, equity indices, FX, crypto, central bank policy rates",
        include=["financial"],
        H=48, W=48, n_layers=6,
        data_sources=["yahoo_finance", "fred_rates"],
    ),
    TrainingNode(
        name="basic_economics",
        description="Macroeconomic fundamentals: GDP, inflation, unemployment, industrial production, trade, housing",
        include=["country_us.macro", "country_cn.macro", "country_eu.macro"],
        H=48, W=48, n_layers=4,
        data_sources=["fred_macro"],
    ),
    TrainingNode(
        name="basic_politics",
        description="Political systems and governance: elections, policy regimes, institutional quality, geopolitical tensions",
        include=["country_us.politics", "country_cn.politics", "country_eu.politics"],
        H=32, W=32, n_layers=4,
        data_sources=[],
    ),
    TrainingNode(
        name="basic_resources",
        description="Energy, metals, agriculture, water, compute resources: production, consumption, prices, reserves",
        include=["resources"],
        H=32, W=32, n_layers=4,
        data_sources=["yahoo_commodities"],
    ),
    TrainingNode(
        name="basic_technology",
        description="Technology frontier: AI capabilities, semiconductor production, data center capacity, R&D spending",
        include=["technology"],
        H=32, W=32, n_layers=4,
        data_sources=[],
    ),
    TrainingNode(
        name="basic_narratives",
        description="Narratives and beliefs: media sentiment, elite consensus, public opinion, investor positioning",
        include=["narratives", "events"],
        H=32, W=32, n_layers=4,
        data_sources=["news_embeddings"],
    ),
]

TIER_1_CROSS_DOMAIN = [
    TrainingNode(
        name="econ_drives_finance",
        description="How macroeconomic indicators drive financial markets: GDP growth → equities, inflation → rates, employment → credit",
        include=["financial", "country_us.macro", "regime", "forecasts.macro", "forecasts.financial"],
        parents=["basic_finance", "basic_economics"],
        H=64, W=64, n_layers=8,
        data_sources=["yahoo_finance", "fred_macro", "fred_rates"],
    ),
    TrainingNode(
        name="geopolitics_commodities",
        description="Geopolitical events driving commodity markets: sanctions → oil, trade wars → metals, conflict → food prices",
        include=["resources", "country_us.politics", "country_cn.politics", "events", "regime"],
        parents=["basic_politics", "basic_resources"],
        H=48, W=48, n_layers=6,
        data_sources=["yahoo_commodities"],
    ),
    TrainingNode(
        name="narratives_drive_markets",
        description="How narratives and sentiment drive market dynamics: media tone → equity flows, positioning → volatility",
        include=["narratives", "financial.equities", "financial.credit", "events", "regime"],
        parents=["basic_narratives", "basic_finance"],
        H=48, W=48, n_layers=6,
        data_sources=["yahoo_finance", "news_embeddings"],
    ),
]

TIER_2_COMPLEX = [
    TrainingNode(
        name="corporate_strategy",
        description="Corporate decision-making in macro context: firm financials, competitive positioning, sector dynamics, executive incentives",
        include=["financial.equities", "country_us.macro", "regime", "forecasts.business"],
        parents=["econ_drives_finance"],
        firms=["AAPL", "NVDA", "MSFT"],
        H=64, W=64, n_layers=8,
        data_sources=["yahoo_finance", "fred_macro"],
    ),
    TrainingNode(
        name="policy_impact",
        description="Policy analysis: monetary policy transmission, fiscal stimulus multipliers, regulatory impact on markets",
        include=["country_us", "country_cn.macro", "country_eu.macro", "financial",
                 "interventions", "regime", "forecasts"],
        parents=["econ_drives_finance", "geopolitics_commodities"],
        countries=["jp", "uk"],
        H=64, W=64, n_layers=8,
        data_sources=["fred_macro", "fred_rates", "yahoo_finance"],
    ),
]

TIER_3_INTEGRATION = [
    TrainingNode(
        name="full_integration",
        description="Full world model integration: all domains on one canvas, regime state receives gradient from everything",
        include=["*"],
        parents=["corporate_strategy", "policy_impact", "narratives_drive_markets"],
        H=128, W=128, n_layers=12,
        n_steps=10000,
        lr=3e-5,
        batch_size=16,
        data_sources=["yahoo_finance", "fred_macro", "fred_rates", "yahoo_commodities"],
    ),
]

STANDARD_DAG = TIER_0_FOUNDATION + TIER_1_CROSS_DOMAIN + TIER_2_COMPLEX + TIER_3_INTEGRATION


# ── DAG Curriculum Trainer ─────────────────────────────────────────────────

@dataclass
class DAGCheckpoint:
    """Metadata for a training checkpoint."""
    node_name: str
    step: int
    timestamp: float
    loss: float
    n_fields: int
    n_params: int
    parents: list[str]

    def to_dict(self):
        return {
            "node_name": self.node_name,
            "step": self.step,
            "timestamp": self.timestamp,
            "loss": self.loss,
            "n_fields": self.n_fields,
            "n_params": self.n_params,
            "parents": self.parents,
        }


class DAGCurriculumTrainer:
    """Orchestrates DAG-based curriculum training.

    Topologically sorts the DAG, trains each node, merges parent weights
    at join points, and checkpoints every node.
    """

    def __init__(
        self,
        nodes: list[TrainingNode],
        data_sources: dict[str, tuple[DatasetSpec, dict]],
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        embed_fn=None,
        embed_dim: int = 16,
    ):
        """
        Args:
            nodes: List of TrainingNode definitions forming the DAG.
            data_sources: Dict mapping source names to (DatasetSpec, data) tuples.
            checkpoint_dir: Where to save checkpoints.
            device: Training device ("cpu", "cuda", etc.).
            embed_fn: Optional embedding function for semantic conditioning.
                Signature: (texts: list[str]) -> list[list[float]].
                If None, uses random embeddings (for smoke testing).
            embed_dim: Dimension of embeddings returned by embed_fn.
        """
        self.nodes = {n.name: n for n in nodes}
        self.data_sources = data_sources
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.embed_fn = embed_fn
        self.embed_dim = embed_dim

        # Trained models: name → {backbone, encoder, decoder, bound, conditioner}
        self.trained: dict[str, dict] = {}
        self.checkpoints: list[DAGCheckpoint] = []

    def _topo_sort(self) -> list[str]:
        """Topologically sort nodes by parent dependencies."""
        visited = set()
        order = []

        def _visit(name):
            if name in visited:
                return
            visited.add(name)
            node = self.nodes[name]
            for parent in node.parents:
                if parent in self.nodes:
                    _visit(parent)
            order.append(name)

        for name in self.nodes:
            _visit(name)

        return order

    def _build_projection(self, node: TrainingNode):
        """Build a WorldProjection and BoundSchema from a node."""
        proj = WorldProjection(
            include=node.include,
            firms=node.firms,
            individuals=node.individuals,
            countries=node.countries,
        )
        bound = project(proj, T=1, H=node.H, W=node.W, d_model=node.d_model)
        return proj, bound

    def _build_conditioner(self, bound):
        """Build a SemanticConditioner for a BoundSchema."""
        if self.embed_fn is not None:
            return bound.build_semantic_conditioner(
                self.embed_fn, embed_dim=self.embed_dim
            )
        else:
            # Random embeddings for smoke testing
            def random_embed(texts):
                return [torch.randn(self.embed_dim).tolist() for _ in texts]
            return bound.build_semantic_conditioner(
                random_embed, embed_dim=self.embed_dim
            )

    def _merge_backbones(self, parent_names: list[str], target_backbone: nn.Module):
        """Merge parent backbone weights by averaging.

        Only merges parameters that have matching shapes. This handles the
        case where parents have different canvas sizes but compatible backbones.
        """
        parent_backbones = []
        for name in parent_names:
            if name in self.trained and "backbone" in self.trained[name]:
                parent_backbones.append(self.trained[name]["backbone"])

        if not parent_backbones:
            return

        target_sd = target_backbone.state_dict()
        merged_sd = {}

        for key in target_sd:
            matching = []
            for pb in parent_backbones:
                pb_sd = pb.state_dict()
                if key in pb_sd and pb_sd[key].shape == target_sd[key].shape:
                    matching.append(pb_sd[key])

            if matching:
                merged_sd[key] = torch.stack(matching).mean(dim=0)
            else:
                merged_sd[key] = target_sd[key]

        target_backbone.load_state_dict(merged_sd)

    def _transfer_encoders(self, parent_names: list[str], encoder: FieldEncoder, decoder: FieldDecoder):
        """Transfer encoder/decoder weights from parents where field names match."""
        for name in parent_names:
            if name not in self.trained:
                continue
            parent = self.trained[name]
            if "encoder" not in parent:
                continue

            src_enc = parent["encoder"]
            src_dec = parent["decoder"]

            for param_name, param in src_enc.encoders.named_parameters():
                if param_name in dict(encoder.encoders.named_parameters()):
                    dst = dict(encoder.encoders.named_parameters())[param_name]
                    if dst.shape == param.shape:
                        dst.data.copy_(param.data)

            for param_name, param in src_dec.decoders.named_parameters():
                if param_name in dict(decoder.decoders.named_parameters()):
                    dst = dict(decoder.decoders.named_parameters())[param_name]
                    if dst.shape == param.shape:
                        dst.data.copy_(param.data)

    def train_node(self, name: str) -> dict:
        """Train a single node in the DAG.

        Returns:
            Dict with training metrics.
        """
        node = self.nodes[name]
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"Description: {node.description}")
        print(f"Parents: {node.parents or ['(none — foundation)']}")
        print(f"{'='*60}")

        # Build projection
        proj, bound = self._build_projection(node)
        n_fields = len(bound.field_names)
        print(f"  Fields: {n_fields}, Canvas: {node.H}x{node.W}, "
              f"Positions: {bound.layout.num_positions}")

        # Build conditioner
        conditioner = self._build_conditioner(bound)

        # Build backbone
        backbone = build_world_model(
            bound, n_layers=node.n_layers, n_loops=node.n_loops
        )
        n_params = sum(p.numel() for p in backbone.parameters())
        print(f"  Backbone: {n_params:,} params, Conditioner: "
              f"{sum(p.numel() for p in conditioner.parameters()):,} params")

        # Merge parent weights
        if node.parents:
            self._merge_backbones(node.parents, backbone)
            print(f"  Merged weights from: {node.parents}")

        # Build encoder/decoder
        encoder = FieldEncoder(bound)
        decoder = FieldDecoder(bound)

        if node.parents:
            self._transfer_encoders(node.parents, encoder, decoder)

        # Collect data sources
        sources = []
        for source_name in node.data_sources:
            if source_name in self.data_sources:
                sources.append(self.data_sources[source_name])

        # Move to device
        backbone = backbone.to(self.device)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        conditioner = conditioner.to(self.device)

        last_loss = 0.0

        if sources:
            dataloader = build_mixed_dataloader(
                bound, sources, batch_size=node.batch_size
            )

            all_params = (
                list(backbone.parameters())
                + list(encoder.parameters())
                + list(decoder.parameters())
                + list(conditioner.parameters())
            )
            optimizer = torch.optim.AdamW(all_params, lr=node.lr, weight_decay=0.01)

            trainer = MaskedCanvasTrainer(
                bound, backbone, encoder, decoder, optimizer,
                device=self.device,
            )

            step = 0
            for epoch in range(10000):
                for batch in dataloader:
                    metrics = trainer.train_step(batch)
                    step += 1
                    last_loss = metrics["loss"]

                    if step % 100 == 0:
                        print(f"  Step {step}/{node.n_steps}: "
                              f"loss={last_loss:.4f}, "
                              f"coverage={metrics['coverage']:.1%}")

                    if step >= node.n_steps:
                        break
                if step >= node.n_steps:
                    break
        else:
            print(f"  No data sources — skipping training (untrained init)")

        # Save checkpoint
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "backbone": backbone.state_dict(),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "conditioner": conditioner.state_dict(),
            "node": {
                "name": name,
                "description": node.description,
                "include": node.include,
                "parents": node.parents,
                "n_fields": n_fields,
            },
        }, ckpt_path)

        ckpt = DAGCheckpoint(
            node_name=name,
            step=node.n_steps,
            timestamp=time.time(),
            loss=last_loss,
            n_fields=n_fields,
            n_params=n_params,
            parents=node.parents,
        )
        self.checkpoints.append(ckpt)

        # Store for child nodes
        self.trained[name] = {
            "backbone": backbone.cpu(),
            "encoder": encoder.cpu(),
            "decoder": decoder.cpu(),
            "conditioner": conditioner.cpu(),
            "bound": bound,
        }

        print(f"  Saved: {ckpt_path}")
        return {"loss": last_loss, "n_fields": n_fields, "n_params": n_params}

    def run(self, nodes: list[str] | None = None):
        """Run the full DAG curriculum (or a subset of nodes).

        Topologically sorts nodes and trains them in order.
        Fork nodes (no parents or same parents) could run in parallel
        on separate GPUs — this sequential implementation is for simplicity.

        Args:
            nodes: Optional list of specific node names to train.
                If None, trains all nodes in topological order.
        """
        order = self._topo_sort()

        if nodes:
            # Filter to requested nodes + their ancestors
            needed = set()
            def _add_ancestors(name):
                needed.add(name)
                for p in self.nodes[name].parents:
                    if p in self.nodes:
                        _add_ancestors(p)
            for n in nodes:
                _add_ancestors(n)
            order = [n for n in order if n in needed]

        print(f"\nDAG Curriculum: {len(order)} nodes")
        print(f"Order: {' → '.join(order)}\n")

        for name in order:
            self.train_node(name)

        # Save DAG metadata
        meta_path = self.checkpoint_dir / "dag_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "nodes": [c.to_dict() for c in self.checkpoints],
                "order": order,
            }, f, indent=2)

        print(f"\nDone! Trained {len(order)} nodes.")
        print(f"Checkpoints in: {self.checkpoint_dir}")

    def run_tier(self, tier: int):
        """Run a specific tier of the standard DAG.

        Tier 0: Foundation (basic_finance, basic_economics, etc.)
        Tier 1: Cross-domain (econ_drives_finance, etc.)
        Tier 2: Complex (corporate_strategy, policy_impact)
        Tier 3: Integration (full_integration)
        """
        tiers = {
            0: [n.name for n in TIER_0_FOUNDATION],
            1: [n.name for n in TIER_1_CROSS_DOMAIN],
            2: [n.name for n in TIER_2_COMPLEX],
            3: [n.name for n in TIER_3_INTEGRATION],
        }

        if tier not in tiers:
            raise ValueError(f"Unknown tier {tier}. Valid: 0-3")

        self.run(tiers[tier])
