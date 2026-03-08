"""DAG-based training curriculum for the General Unified World Model.

The training curriculum is a directed acyclic graph (DAG) where:
- **Fork nodes** copy model weights and train each copy on different
  domain-specific projections in parallel
- **Join nodes** merge the trained copies by weight averaging (model souping)

This combines the speed of parallel training with the breadth of cross-domain
integration. The semantic embedding conditioner makes this work: because
dynamics are conditioned on field identity (not hardcoded heads), merged
weights can handle any field type.

Each training node specifies which world model fields to include and which
entities to instantiate. Curriculum stages can be described in YAML or
built programmatically by an LLM that examines available datasets.

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
from general_unified_world_model.training.backbone import (
    build_world_model, WorldModelBackbone,
    build_cogvideox_world_model, CogVideoXBackbone,
)
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, InputSpec, OutputSpec, build_mixed_dataloader,
)


# ── Entity type registry ─────────────────────────────────────────────────

def _resolve_entities(entities: dict[str, str]) -> dict[str, Any]:
    """Resolve entity type names to instances.

    Args:
        entities: Dict mapping names to type names.
            E.g. {"firm_AAPL": "Business", "country_jp": "Country"}.

    Returns:
        Dict mapping names to dataclass instances.
    """
    from general_unified_world_model.schema.business import Business
    from general_unified_world_model.schema.individual import Individual
    from general_unified_world_model.schema.sector import Sector
    from general_unified_world_model.schema.supply_chain import SupplyChainNode
    from general_unified_world_model.schema.country import Country

    ENTITY_TYPES = {
        "Business": Business,
        "Individual": Individual,
        "Sector": Sector,
        "SupplyChainNode": SupplyChainNode,
        "Country": Country,
    }

    result = {}
    for name, type_name in entities.items():
        if type_name not in ENTITY_TYPES:
            raise ValueError(
                f"Unknown entity type '{type_name}' for '{name}'. "
                f"Valid types: {list(ENTITY_TYPES.keys())}"
            )
        result[name] = ENTITY_TYPES[type_name]()
    return result


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
        entities: Dict mapping entity names to type names for YAML serialization.
            E.g. {"firm_AAPL": "Business", "country_jp": "Country"}.
        H, W: Canvas dimensions.
        d_model: Latent dimension.
        n_layers: Transformer depth.
        n_loops: Looped attention iterations.
        n_steps: Number of training steps.
        lr: Learning rate.
        batch_size: Batch size.
        data_sources: Names of data source keys to use.
    """
    name: str
    description: str
    include: list[str]
    parents: list[str] = dc_field(default_factory=list)
    entities: dict[str, str] = dc_field(default_factory=dict)
    H: int | None = None
    W: int | None = None
    d_model: int = 64
    n_layers: int = 4
    n_loops: int = 3
    n_steps: int = 5000
    lr: float = 1e-4
    batch_size: int = 32
    data_sources: list[str] = dc_field(default_factory=list)


# ── Standard DAG Definitions ──────────────────────────────────────────────

TIER_0_FOUNDATION = [
    TrainingNode(
        name="basic_finance",
        description="Core financial markets: yield curves, credit spreads, equity indices, FX, crypto, central bank policy rates",
        include=["financial"],
        n_layers=6,
        data_sources=["yahoo_finance", "fred_rates"],
    ),
    TrainingNode(
        name="basic_economics",
        description="Macroeconomic fundamentals: GDP, inflation, unemployment, industrial production, trade, housing",
        include=["country_us.macro", "country_cn.macro", "country_eu.macro"],
        n_layers=4,
        data_sources=["fred_macro"],
    ),
    TrainingNode(
        name="basic_politics",
        description="Political systems and governance: elections, policy regimes, institutional quality, geopolitical tensions",
        include=["country_us.politics", "country_cn.politics", "country_eu.politics"],
        n_layers=4,
        data_sources=[],
    ),
    TrainingNode(
        name="basic_resources",
        description="Energy, metals, agriculture, water, compute resources: production, consumption, prices, reserves",
        include=["resources"],
        n_layers=4,
        data_sources=["yahoo_commodities"],
    ),
    TrainingNode(
        name="basic_technology",
        description="Technology frontier: AI capabilities, semiconductor production, data center capacity, R&D spending",
        include=["technology"],
        n_layers=4,
        data_sources=[],
    ),
    TrainingNode(
        name="basic_narratives",
        description="Narratives and beliefs: media sentiment, elite consensus, public opinion, investor positioning",
        include=["narratives", "events"],
        n_layers=4,
        data_sources=["news_embeddings"],
    ),
]

TIER_1_CROSS_DOMAIN = [
    TrainingNode(
        name="econ_drives_finance",
        description="How macroeconomic indicators drive financial markets: GDP growth → equities, inflation → rates, employment → credit",
        include=["financial", "country_us.macro", "regime", "forecasts.macro", "forecasts.financial"],
        parents=["basic_finance", "basic_economics"],
        n_layers=8,
        data_sources=["yahoo_finance", "fred_macro", "fred_rates"],
    ),
    TrainingNode(
        name="geopolitics_commodities",
        description="Geopolitical events driving commodity markets: sanctions → oil, trade wars → metals, conflict → food prices",
        include=["resources", "country_us.politics", "country_cn.politics", "events", "regime"],
        parents=["basic_politics", "basic_resources"],
        n_layers=6,
        data_sources=["yahoo_commodities"],
    ),
    TrainingNode(
        name="narratives_drive_markets",
        description="How narratives and sentiment drive market dynamics: media tone → equity flows, positioning → volatility",
        include=["narratives", "financial.equities", "financial.credit", "events", "regime"],
        parents=["basic_narratives", "basic_finance"],
        n_layers=6,
        data_sources=["yahoo_finance", "news_embeddings"],
    ),
]

TIER_2_COMPLEX = [
    TrainingNode(
        name="corporate_strategy",
        description="Corporate decision-making in macro context: firm financials, competitive positioning, sector dynamics, executive incentives",
        include=["financial.equities", "country_us.macro", "regime", "forecasts.business"],
        parents=["econ_drives_finance"],
        entities={"firm_AAPL": "Business", "firm_NVDA": "Business", "firm_MSFT": "Business"},
        n_layers=8,
        data_sources=["yahoo_finance", "fred_macro"],
    ),
    TrainingNode(
        name="policy_impact",
        description="Policy analysis: monetary policy transmission, fiscal stimulus multipliers, regulatory impact on markets",
        include=["country_us", "country_cn.macro", "country_eu.macro", "financial",
                 "interventions", "regime", "forecasts"],
        parents=["econ_drives_finance", "geopolitics_commodities"],
        entities={"country_jp": "Country", "country_uk": "Country"},
        n_layers=8,
        data_sources=["fred_macro", "fred_rates", "yahoo_finance"],
    ),
]

TIER_3_INTEGRATION = [
    TrainingNode(
        name="full_integration",
        description="Full world model integration: all domains on one canvas, regime state receives gradient from everything",
        include=["*"],
        parents=["corporate_strategy", "policy_impact", "narratives_drive_markets"],
        n_layers=12,
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
        backbone: str = "cogvideox",
        pretrained_model_id: str = "THUDM/CogVideoX-2b",
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
            backbone: "cogvideox" (default) to graft onto pretrained CogVideoX,
                or "scratch" to train a fresh WorldModelBackbone.
            pretrained_model_id: HuggingFace model ID for CogVideoX.
        """
        self.nodes = {n.name: n for n in nodes}
        self.data_sources = data_sources
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.embed_fn = embed_fn
        self.embed_dim = embed_dim
        self.backbone_type = backbone
        self.pretrained_model_id = pretrained_model_id

        # Lazy-loaded CogVideoX transformer (shared across all nodes)
        self._cogvideox_transformer = None

        # Trained models: name → {backbone_state|backbone, encoder, decoder, bound, conditioner}
        self.trained: dict[str, dict] = {}
        self.checkpoints: list[DAGCheckpoint] = []

    def _ensure_cogvideox_loaded(self):
        """Lazy-load the CogVideoX transformer (once, shared across nodes)."""
        if self._cogvideox_transformer is not None:
            return

        try:
            from diffusers import CogVideoXTransformer3DModel
        except ImportError:
            print("WARNING: diffusers not installed — falling back to scratch backbone")
            print("  Install with: pip install 'general-unified-world-model[cogvideox]'")
            self.backbone_type = "scratch"
            return

        print(f"Loading CogVideoX from {self.pretrained_model_id}...")
        self._cogvideox_transformer = CogVideoXTransformer3DModel.from_pretrained(
            self.pretrained_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self._cogvideox_transformer = self._cogvideox_transformer.to(self.device)

        n_blocks = len(self._cogvideox_transformer.transformer_blocks)
        n_params = sum(p.numel() for p in self._cogvideox_transformer.parameters())
        print(f"  Loaded: {n_blocks} blocks, {n_params / 1e9:.1f}B params, "
              f"device={self.device}")

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
        entities = _resolve_entities(node.entities) if node.entities else {}
        proj = WorldProjection(
            include=node.include,
            entities=entities,
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

        Only merges parameters that have matching shapes. For CogVideoX backbones,
        frozen block params are skipped (they're shared and identical).
        """
        parent_state_dicts = []
        for name in parent_names:
            if name not in self.trained:
                continue
            if "backbone_state" in self.trained[name]:
                parent_state_dicts.append(self.trained[name]["backbone_state"])
            elif "backbone" in self.trained[name]:
                parent_state_dicts.append(self.trained[name]["backbone"].state_dict())

        if not parent_state_dicts:
            return

        target_sd = target_backbone.state_dict()
        merged_sd = {}

        for key in target_sd:
            matching = []
            for psd in parent_state_dicts:
                if key in psd and psd[key].shape == target_sd[key].shape:
                    matching.append(psd[key])

            if matching:
                merged_sd[key] = torch.stack(matching).mean(dim=0)
            else:
                merged_sd[key] = target_sd[key]

        target_backbone.load_state_dict(merged_sd, strict=False)

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
        if self.backbone_type == "cogvideox":
            self._ensure_cogvideox_loaded()

        if self.backbone_type == "cogvideox" and self._cogvideox_transformer is not None:
            backbone = build_cogvideox_world_model(
                self._cogvideox_transformer, bound, n_loops=node.n_loops,
            )
            n_trainable = backbone.trainable_param_count()
            n_frozen = backbone.frozen_param_count()
            print(f"  CogVideoX backbone: {n_trainable:,} trainable, "
                  f"{n_frozen / 1e9:.1f}B frozen")
        else:
            backbone = build_world_model(
                bound, n_layers=node.n_layers, n_loops=node.n_loops
            )
            n_trainable = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone: {n_trainable:,} params")
        print(f"  Conditioner: "
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

            all_params = [
                p for p in backbone.parameters() if p.requires_grad
            ] + [
                p for p in encoder.parameters() if p.requires_grad
            ] + [
                p for p in decoder.parameters() if p.requires_grad
            ] + [
                p for p in conditioner.parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(all_params, lr=node.lr, weight_decay=0.01)

            trainer = MaskedCanvasTrainer(
                bound, backbone, encoder, decoder, optimizer,
                device=self.device,
                conditioner=conditioner,
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

        # Save checkpoint (trainable params only for CogVideoX)
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "backbone": backbone.state_dict(),
            "backbone_type": self.backbone_type,
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
            n_params=n_trainable,
            parents=node.parents,
        )
        self.checkpoints.append(ckpt)

        # Store for child nodes.
        # For CogVideoX: store only trainable state dict (frozen blocks are shared).
        # For scratch: store the full backbone on CPU.
        if isinstance(backbone, CogVideoXBackbone):
            self.trained[name] = {
                "backbone_state": backbone.state_dict(),
                "encoder": encoder.cpu(),
                "decoder": decoder.cpu(),
                "conditioner": conditioner.cpu(),
                "bound": bound,
            }
        else:
            self.trained[name] = {
                "backbone": backbone.cpu(),
                "encoder": encoder.cpu(),
                "decoder": decoder.cpu(),
                "conditioner": conditioner.cpu(),
                "bound": bound,
            }

        print(f"  Saved: {ckpt_path}")
        return {"loss": last_loss, "n_fields": n_fields, "n_params": n_trainable}

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


# ── Curriculum Specification ──────────────────────────────────────────────

@dataclass
class Stage:
    """A single training stage (unit of work) within a curriculum group.

    Args:
        description: Natural language description of what to learn.
        datasets: Data sources to use for training.
        include: Explicit include paths (required — no NL resolution).
        H, W, d_model: Canvas dimensions (override defaults).
        n_layers, n_loops: Model architecture (override defaults).
        n_steps: Training steps (override defaults).
    """
    description: str
    datasets: list[str] = dc_field(default_factory=list)
    include: list[str] | None = None
    H: int | None = None
    W: int | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_loops: int | None = None
    n_steps: int | None = None


@dataclass
class StagesInParallel:
    """A group of stages that train in parallel within a curriculum.

    A group contains parallel stages (fork) that all train simultaneously
    from the same parent weights. After all stages in a group complete,
    their weights are merged (join) before proceeding to the next group.

    Args:
        name: Group name (e.g. "foundations", "cross_domain").
        stages: List of training stages to run in parallel.
        builds_on: Name of the previous group (or None for the first).
    """
    name: str
    stages: list[Stage] = dc_field(default_factory=list)
    builds_on: str | None = None


@dataclass
class CurriculumSpec:
    """Full curriculum specification — a sequence of stage groups forming a DAG.

    Each group forks into parallel stages, trains them, then joins
    (merges weights) before the next group.

    Args:
        name: Curriculum name.
        plan: Ordered list of stage groups (StagesInParallel).
        defaults: Default hyperparameters for all nodes.
    """
    name: str = "curriculum"
    plan: list[StagesInParallel] = dc_field(default_factory=list)
    defaults: dict = dc_field(default_factory=lambda: {
        "d_model": 64,
        "n_layers": 4, "n_loops": 3,
        "n_steps": 5000, "lr": 1e-4, "batch_size": 32,
    })

    def to_training_nodes(self) -> list[TrainingNode]:
        """Convert this curriculum spec to a list of TrainingNodes.

        Each Stage becomes a TrainingNode. Groups (StagesInParallel) define
        the DAG structure: all stages within a group share the same
        parents (the stages of the previous group).

        Returns:
            List of TrainingNode instances ready for DAGCurriculumTrainer.
        """
        nodes = []
        prev_stage_names = []

        for group in self.plan:
            group_node_names = []

            for i, stg in enumerate(group.stages):
                # include must be specified explicitly
                include = stg.include or ["*"]

                # Generate a clean node name
                name = f"{group.name}_{i}" if len(group.stages) > 1 else group.name
                if len(group.stages) > 1:
                    # Use first two keywords from description
                    words = stg.description.lower().split()[:3]
                    clean = "_".join(w for w in words if w.isalnum())[:20]
                    name = f"{group.name}_{clean}"

                # Determine parents
                if group.builds_on and prev_stage_names:
                    parents = list(prev_stage_names)
                else:
                    parents = []

                # Merge defaults with overrides
                d = dict(self.defaults)
                for key in ["H", "W", "d_model", "n_layers", "n_loops", "n_steps"]:
                    val = getattr(stg, key, None)
                    if val is not None:
                        d[key] = val

                node = TrainingNode(
                    name=name,
                    description=stg.description,
                    include=include,
                    parents=parents,
                    H=d.get("H"), W=d.get("W"), d_model=d["d_model"],
                    n_layers=d["n_layers"], n_loops=d["n_loops"],
                    n_steps=d["n_steps"], lr=d.get("lr", 1e-4),
                    batch_size=d.get("batch_size", 32),
                    data_sources=stg.datasets,
                )
                nodes.append(node)
                group_node_names.append(name)

            prev_stage_names = group_node_names

        return nodes

    def to_dict(self) -> dict:
        """Serialize to a plain dict (suitable for YAML/JSON output)."""
        groups = []
        for group in self.plan:
            stage_dicts = []
            for stg in group.stages:
                s = {"description": stg.description}
                if stg.datasets:
                    s["datasets"] = stg.datasets
                if stg.include is not None:
                    s["include"] = stg.include
                for key in ["H", "W", "d_model", "n_layers", "n_loops", "n_steps"]:
                    val = getattr(stg, key, None)
                    if val is not None:
                        s[key] = val
                stage_dicts.append(s)
            st = {"name": group.name, "stages": stage_dicts}
            if group.builds_on:
                st["builds_on"] = group.builds_on
            groups.append(st)

        return {"name": self.name, "defaults": dict(self.defaults), "plan": groups}

    def to_yaml(self, path: str | Path) -> None:
        """Write this curriculum to a YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, width=120)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CurriculumSpec:
        """Load a curriculum specification from a YAML file."""
        import yaml

        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> CurriculumSpec:
        """Load a curriculum specification from a Python dict.

        Same structure as the YAML format.
        """
        defaults = d.get("defaults", {})
        groups = []

        raw_groups = d.get("plan", [])

        for group_raw in raw_groups:
            stage_list = []
            raw_stages = group_raw.get("stages", [])
            for s in raw_stages:
                description = s.get("description", "")
                stage_list.append(Stage(
                    description=description,
                    datasets=s.get("datasets", []),
                    include=s.get("include"),
                    H=s.get("H"), W=s.get("W"), d_model=s.get("d_model"),
                    n_layers=s.get("n_layers"), n_loops=s.get("n_loops"),
                    n_steps=s.get("n_steps"),
                ))
            groups.append(StagesInParallel(
                name=group_raw["name"],
                stages=stage_list,
                builds_on=group_raw.get("builds_on"),
            ))

        return cls(name=d.get("name", "curriculum"), plan=groups, defaults=defaults)


# ── Dataset Profile ──────────────────────────────────────────────────────

@dataclass
class DatasetProfile:
    """Metadata about a dataset for LLM-driven curriculum planning.

    The LLM examines these profiles to decide how to structure the
    training curriculum — which datasets to train in parallel, which
    to pair for cross-domain transfer, and in what order.

    Args:
        name: Human-readable name (e.g. "FRED Macro", "Hospital EHR").
        description: What this dataset contains and its domain.
        input_specs: How columns map to world model field paths.
        n_samples: Number of rows/samples in the dataset.
        columns: Column names in the raw dataset.
        temporal_range: Date range or tick range covered.
        update_frequency: How often the data updates (e.g. "daily", "quarterly").
        source: Where the data comes from ("local", "huggingface", "api").
        metadata: Any additional metadata (README excerpts, tags, etc.).
    """
    name: str
    description: str
    input_specs: list[InputSpec] = dc_field(default_factory=list)
    n_samples: int = 0
    columns: list[str] = dc_field(default_factory=list)
    temporal_range: str = ""
    update_frequency: str = ""
    source: str = "local"
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def summary(self) -> str:
        """Generate a concise summary for LLM consumption."""
        mapped = [f"  {m.key} → {m.field_path}" for m in self.input_specs[:20]]
        mapped_str = "\n".join(mapped)
        extra = f"\n  ... and {len(self.input_specs) - 20} more" if len(self.input_specs) > 20 else ""

        parts = [
            f"Dataset: {self.name}",
            f"Source: {self.source}",
            f"Description: {self.description}",
            f"Samples: {self.n_samples}",
        ]
        if self.temporal_range:
            parts.append(f"Temporal range: {self.temporal_range}")
        if self.update_frequency:
            parts.append(f"Update frequency: {self.update_frequency}")
        if self.columns:
            parts.append(f"Columns ({len(self.columns)}): {', '.join(self.columns[:15])}")
            if len(self.columns) > 15:
                parts[-1] += f", ... ({len(self.columns) - 15} more)"
        if mapped:
            parts.append(f"Field mappings:\n{mapped_str}{extra}")

        return "\n".join(parts)


def build_curriculum(
    goal: str,
    datasets: list[DatasetProfile],
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
) -> CurriculumSpec:
    """LLM-driven curriculum builder.

    Examines available datasets and builds a multi-stage curriculum:
    - Stage 1: Parallel domain training (each dataset within its modality)
    - Stage 2: Paired cross-domain transfer (semantically close pairs)
    - Stage 3: Full integration (all datasets on one canvas)
    - Stage 4 (optional): 2nd order relationship associations

    The LLM examines dataset metadata, field mappings, and semantic
    distances between domains to design an optimal training schedule.

    Args:
        goal: Natural language description of what the user wants.
            E.g. "fine-tune to learn cardiovascular patient health
                  for our hospital using our private EHR dataset"
        datasets: Available datasets with their metadata.
        provider: "anthropic" or "openai".
        api_key: API key (or set env var).
        model: Model to use.

    Returns:
        CurriculumSpec ready for DAGCurriculumTrainer.

    Note:
        Stages 2+ require datasets to share a common world reference
        frame (temporal alignment, geographic context). The LLM considers
        this when deciding which datasets can be co-trained.
    """
    import dataclasses as _dc
    import os
    import urllib.request
    import urllib.error

    from general_unified_world_model.schema.world import World
    from canvas_engineering import Field

    # Build schema description
    all_paths = []
    def _walk(obj, prefix):
        if not _dc.is_dataclass(obj):
            return
        for f in _dc.fields(obj):
            val = getattr(obj, f.name)
            fp = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(val, Field):
                all_paths.append(fp)
            elif _dc.is_dataclass(val):
                _walk(val, fp)
    _walk(World(), "")

    top_level = [f.name for f in _dc.fields(World)]

    # Build dataset summaries
    dataset_block = "\n\n".join(ds.summary() for ds in datasets)

    system_prompt = f"""You are a training curriculum designer for the General Unified World Model.

The world model has {len(all_paths)} fields across these top-level domains:
{', '.join(top_level)}

You are given:
1. A user's goal description
2. Available datasets with their field mappings

Your job: design a multi-stage training curriculum as a JSON CurriculumSpec.

## Curriculum stages

Stage 1 — PARALLEL DOMAIN TRAINING:
Train each dataset (or closely related group) independently within its modality structure.
Multiple datasets can train in parallel if they cover different domains.

Stage 2 — PAIRED CROSS-DOMAIN TRANSFER:
Pair datasets that are semantically close (e.g. macro + financial, health + demographics).
Co-train paired datasets on a shared canvas. This only works when datasets share a
common world reference frame (same time period, compatible geography).

Stage 3 — FULL INTEGRATION:
Train all datasets together on one canvas. Again, requires shared reference frame.

Stage 4 (optional) — 2ND ORDER ASSOCIATIONS:
Associate the primary dataset's conditions to indirect/second-order relationships
discovered in stages 1-3.

## Output format
Respond with ONLY a JSON object:
{{
  "name": "curriculum_name",
  "defaults": {{"d_model": 64, "n_layers": 6, "n_loops": 3, "n_steps": 5000}},
  "plan": [
    {{
      "name": "group_name",
      "stages": [
        {{
          "description": "Description of what this stage trains",
          "include": ["financial.yield_curves", "country_us.macro", "regime"],
          "datasets": ["dataset_name"],
          "n_layers": 6,
          "n_steps": 5000
        }}
      ],
      "builds_on": "previous_group_name_or_null"
    }}
  ],
  "reasoning": "Why this curriculum structure was chosen"
}}

Rules:
1. Each stage MUST have an explicit "include" list of dotted field paths.
2. Use exact field paths from the schema. You can include whole domains (e.g. "financial")
   or specific sub-paths (e.g. "financial.yield_curves").
3. Always include "regime" in cross-domain and integration stages.
4. Match dataset names exactly as provided.
5. Only pair datasets in stage 2 if they plausibly share a temporal/geographic reference frame.
6. Keep stage 1 lean — just the fields each dataset covers.
"""

    user_msg = f"""Goal: {goal}

Available datasets:
{dataset_block}"""

    # Resolve API key
    if api_key is None:
        env_key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"No API key. Set {env_key} or pass api_key=.")

    # Call the LLM
    if provider == "anthropic":
        body = json.dumps({
            "model": model or "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        raw_text = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
    elif provider == "openai":
        body = json.dumps({
            "model": model or "gpt-4o-mini",
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        raw_text = data["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Parse response
    import re
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    parsed = json.loads(text)

    return CurriculumSpec.from_dict(parsed)
