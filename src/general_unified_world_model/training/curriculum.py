"""Curriculum training for the General Unified World Model.

The training proceeds in phases:

Phase 1: Independent Modality Training
    Train each domain (financial, macro, events, etc.) independently.
    Each domain gets its own small canvas and backbone. This is fast
    because the canvases are small and the data sources are separate.

Phase 2: Domain Coupling
    Merge domains that are causally adjacent (financial + macro,
    events + narratives, etc.) onto shared canvases. The pretrained
    domain weights initialize the corresponding blocks. The shared
    regime latent begins to learn cross-domain structure.

Phase 3: Full Integration
    All domains on one canvas. The regime state, forecast heads, and
    intervention space now receive gradient from everything. This is
    the most expensive phase but benefits from all the pretrained
    domain structure.

Phase 4: Fine-tuning on Specific Tasks
    Task-specific projections (e.g. "predict recession probability
    given financial + macro + sentiment") with frozen backbone and
    trainable heads.

The key insight: because canvas positions have semantic meaning via
the type system, we can approximate transfer distance between any
two field embeddings using their semantic similarity. Fields that
are semantically close (GDP and industrial production) will have
correlated latent dynamics. Fields that are semantically far
(GDP and seismic risk) will be nearly independent. This lets us
prioritize which domain couplings to train first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from canvas_engineering import ConnectivityPolicy, compile_schema

from general_unified_world_model.projection.subset import project
from general_unified_world_model.training.backbone import build_world_model, WorldModelBackbone
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, MaskedCanvasTrainer,
    DatasetSpec, DataSource, HeterogeneousDataset, build_mixed_dataloader,
)


# ── Phase Definitions ───────────────────────────────────────────────────

@dataclass
class DomainSpec:
    """Specification for an independent domain training run.

    Args:
        name: Domain name (e.g. "financial", "macro_us").
        include: World model paths to include.
        H, W: Canvas spatial dimensions for this domain.
        d_model: Latent dimension.
        n_layers: Transformer depth.
        n_loops: Looped attention iterations.
        sources: List of DatasetSpec names that map to this domain.
    """
    name: str
    include: list[str]
    H: int = 32
    W: int = 32
    d_model: int = 64
    n_layers: int = 4
    n_loops: int = 3
    sources: list[str] = dc_field(default_factory=list)


# Standard domain decomposition
STANDARD_DOMAINS = [
    DomainSpec(
        name="financial_core",
        include=["financial", "events"],
        H=48, W=48, d_model=64, n_layers=6,
        sources=["yahoo_finance", "fred_rates"],
    ),
    DomainSpec(
        name="macro_us",
        include=["country_us.macro", "country_us.domestic_sentiment"],
        H=32, W=32, d_model=64, n_layers=4,
        sources=["fred_macro"],
    ),
    DomainSpec(
        name="macro_cn",
        include=["country_cn.macro"],
        H=32, W=32, d_model=64, n_layers=4,
        sources=["cn_macro"],
    ),
    DomainSpec(
        name="macro_eu",
        include=["country_eu.macro"],
        H=32, W=32, d_model=64, n_layers=4,
        sources=["eu_macro"],
    ),
    DomainSpec(
        name="narratives",
        include=["narratives", "events"],
        H=32, W=32, d_model=64, n_layers=4,
        sources=["news_embeddings", "social_sentiment"],
    ),
    DomainSpec(
        name="resources",
        include=["resources"],
        H=32, W=32, d_model=64, n_layers=4,
        sources=["yahoo_commodities", "eia_data"],
    ),
    DomainSpec(
        name="regime",
        include=["regime", "forecasts"],
        H=24, W=24, d_model=64, n_layers=6,
        sources=[],  # regime is latent-only, trained via coupling
    ),
]


@dataclass
class CouplingSpec:
    """Specification for coupling two or more domains.

    Args:
        name: Coupling name.
        domains: List of domain names to merge.
        include: Additional paths to include (e.g. regime latent).
        H, W: Canvas dimensions for the coupled model.
    """
    name: str
    domains: list[str]
    include: list[str] = dc_field(default_factory=list)
    H: int = 64
    W: int = 64


STANDARD_COUPLINGS = [
    CouplingSpec(
        name="financial_macro",
        domains=["financial_core", "macro_us"],
        include=["regime"],
        H=64, W=64,
    ),
    CouplingSpec(
        name="global_macro",
        domains=["macro_us", "macro_cn", "macro_eu"],
        include=["regime", "financial.fx"],
        H=64, W=64,
    ),
    CouplingSpec(
        name="narratives_financial",
        domains=["narratives", "financial_core"],
        include=["regime"],
        H=64, W=64,
    ),
]


# ── Curriculum Scheduler ────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    """Configuration for one training phase."""
    name: str
    n_steps: int
    lr: float
    batch_size: int = 32
    warmup_steps: int = 100
    save_every: int = 1000


@dataclass
class CurriculumConfig:
    """Full curriculum configuration.

    Args:
        phases: Ordered list of training phases.
        checkpoint_dir: Where to save checkpoints.
        device: Training device.
    """
    phases: list[PhaseConfig] = dc_field(default_factory=list)
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"

    @classmethod
    def default(cls) -> CurriculumConfig:
        """Default 4-phase curriculum."""
        return cls(phases=[
            PhaseConfig("phase1_independent", n_steps=10000, lr=1e-4, batch_size=64),
            PhaseConfig("phase2_coupling", n_steps=5000, lr=5e-5, batch_size=32),
            PhaseConfig("phase3_integration", n_steps=10000, lr=3e-5, batch_size=16),
            PhaseConfig("phase4_finetune", n_steps=5000, lr=1e-5, batch_size=32),
        ])


class CurriculumTrainer:
    """Orchestrates multi-phase curriculum training.

    Phase 1: Train each domain independently (parallelizable)
    Phase 2: Merge causally adjacent domains
    Phase 3: Full integration on single canvas
    Phase 4: Task-specific fine-tuning
    """

    def __init__(
        self,
        config: CurriculumConfig,
        data_sources: dict[str, DataSource],
        domains: list[DomainSpec] | None = None,
        couplings: list[CouplingSpec] | None = None,
    ):
        self.config = config
        self.data_sources = data_sources
        self.domains = domains or STANDARD_DOMAINS
        self.couplings = couplings or STANDARD_COUPLINGS
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated during training
        self.domain_models: dict[str, dict] = {}
        self.coupled_models: dict[str, dict] = {}
        self.integrated_model: dict | None = None

    def run_phase1(self, phase_config: PhaseConfig):
        """Phase 1: Independent domain training.

        Each domain gets its own projection, backbone, encoder, decoder.
        These can run in parallel on different GPUs.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: Independent Domain Training")
        print(f"{'='*60}")

        for domain in self.domains:
            print(f"\n--- Training domain: {domain.name} ---")

            # Build projection for this domain
            bound = project(
                include=domain.include,
                T=1, H=domain.H, W=domain.W, d_model=domain.d_model,
            )

            print(f"  Fields: {len(bound.field_names)}, "
                  f"Positions: {bound.layout.num_positions}, "
                  f"Connections: {len(bound.topology.connections)}")

            # Build backbone
            backbone = build_world_model(
                bound, n_layers=domain.n_layers, n_loops=domain.n_loops
            )
            n_params = sum(p.numel() for p in backbone.parameters())
            print(f"  Backbone: {n_params:,} parameters")

            # Build encoder/decoder
            encoder = FieldEncoder(bound)
            decoder = FieldDecoder(bound)

            # Collect data sources for this domain
            sources = []
            for source_name in domain.sources:
                if source_name in self.data_sources:
                    sources.append(self.data_sources[source_name])

            if not sources:
                print(f"  No data sources found, skipping training")
                self.domain_models[domain.name] = {
                    "bound": bound,
                    "backbone": backbone,
                    "encoder": encoder,
                    "decoder": decoder,
                }
                continue

            # Build dataloader
            dataloader = build_mixed_dataloader(bound, sources, batch_size=phase_config.batch_size)

            # Build optimizer
            all_params = (
                list(backbone.parameters())
                + list(encoder.parameters())
                + list(decoder.parameters())
            )
            optimizer = torch.optim.AdamW(all_params, lr=phase_config.lr, weight_decay=0.01)

            # Build trainer
            trainer = MaskedCanvasTrainer(
                bound, backbone, encoder, decoder, optimizer,
                device=self.config.device,
            )

            # Training loop
            step = 0
            for epoch in range(1000):  # max epochs
                for batch in dataloader:
                    metrics = trainer.train_step(batch)
                    step += 1

                    if step % 100 == 0:
                        print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                              f"coverage={metrics['coverage']:.1%}")

                    if step % phase_config.save_every == 0:
                        self._save_checkpoint(
                            f"phase1_{domain.name}_step{step}",
                            backbone, encoder, decoder,
                        )

                    if step >= phase_config.n_steps:
                        break
                if step >= phase_config.n_steps:
                    break

            self.domain_models[domain.name] = {
                "bound": bound,
                "backbone": backbone,
                "encoder": encoder,
                "decoder": decoder,
            }

    def run_phase2(self, phase_config: PhaseConfig):
        """Phase 2: Domain coupling.

        Merge causally adjacent domains onto shared canvases.
        Initialize from Phase 1 weights where field names match.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: Domain Coupling")
        print(f"{'='*60}")

        for coupling in self.couplings:
            print(f"\n--- Coupling: {coupling.name} ---")

            # Merge include paths from all constituent domains
            all_includes = list(coupling.include)
            for domain_name in coupling.domains:
                domain = next((d for d in self.domains if d.name == domain_name), None)
                if domain:
                    all_includes.extend(domain.include)

            # Build coupled projection
            bound = project(
                include=list(set(all_includes)),
                T=1, H=coupling.H, W=coupling.W, d_model=64,
            )

            print(f"  Fields: {len(bound.field_names)}, "
                  f"Positions: {bound.layout.num_positions}")

            # Build backbone (slightly deeper for coupling)
            backbone = build_world_model(bound, n_layers=8, n_loops=3)
            encoder = FieldEncoder(bound)
            decoder = FieldDecoder(bound)

            # Initialize from domain models where possible
            for domain_name in coupling.domains:
                if domain_name in self.domain_models:
                    domain_data = self.domain_models[domain_name]
                    self._transfer_weights(
                        domain_data["encoder"], encoder,
                        domain_data["decoder"], decoder,
                    )

            # Collect all data sources
            sources = []
            for domain_name in coupling.domains:
                domain = next((d for d in self.domains if d.name == domain_name), None)
                if domain:
                    for source_name in domain.sources:
                        if source_name in self.data_sources:
                            sources.append(self.data_sources[source_name])

            if not sources:
                print(f"  No data sources, skipping")
                continue

            # Train coupled model
            dataloader = build_mixed_dataloader(bound, sources, batch_size=phase_config.batch_size)
            all_params = (
                list(backbone.parameters())
                + list(encoder.parameters())
                + list(decoder.parameters())
            )
            optimizer = torch.optim.AdamW(all_params, lr=phase_config.lr)
            trainer = MaskedCanvasTrainer(
                bound, backbone, encoder, decoder, optimizer,
                device=self.config.device,
            )

            step = 0
            for epoch in range(1000):
                for batch in dataloader:
                    metrics = trainer.train_step(batch)
                    step += 1
                    if step % 100 == 0:
                        print(f"  Step {step}: loss={metrics['loss']:.4f}")
                    if step >= phase_config.n_steps:
                        break
                if step >= phase_config.n_steps:
                    break

            self.coupled_models[coupling.name] = {
                "bound": bound,
                "backbone": backbone,
                "encoder": encoder,
                "decoder": decoder,
            }

    def run_phase3(self, phase_config: PhaseConfig):
        """Phase 3: Full integration.

        All domains on one canvas. The regime state, forecast heads, and
        intervention space now receive gradient from everything.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 3: Full Integration")
        print(f"{'='*60}")

        # Full world projection
        bound = project(include=["*"], T=1, H=128, W=128, d_model=64)

        print(f"  Fields: {len(bound.field_names)}, "
              f"Positions: {bound.layout.num_positions}, "
              f"Connections: {len(bound.topology.connections)}")

        backbone = build_world_model(bound, n_layers=12, n_loops=3)
        encoder = FieldEncoder(bound)
        decoder = FieldDecoder(bound)

        # Initialize from coupled models
        for coupling_name, model_data in self.coupled_models.items():
            self._transfer_weights(
                model_data["encoder"], encoder,
                model_data["decoder"], decoder,
            )

        # Collect all data sources
        sources = list(self.data_sources.values())

        if not sources:
            print("  No data sources available")
            return

        dataloader = build_mixed_dataloader(bound, sources, batch_size=phase_config.batch_size)
        all_params = (
            list(backbone.parameters())
            + list(encoder.parameters())
            + list(decoder.parameters())
        )
        optimizer = torch.optim.AdamW(all_params, lr=phase_config.lr)
        trainer = MaskedCanvasTrainer(
            bound, backbone, encoder, decoder, optimizer,
            device=self.config.device,
        )

        step = 0
        for epoch in range(1000):
            for batch in dataloader:
                metrics = trainer.train_step(batch)
                step += 1
                if step % 100 == 0:
                    print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                          f"coverage={metrics['coverage']:.1%}")
                if step % phase_config.save_every == 0:
                    self._save_checkpoint("phase3_integrated", backbone, encoder, decoder)
                if step >= phase_config.n_steps:
                    break
            if step >= phase_config.n_steps:
                break

        self.integrated_model = {
            "bound": bound,
            "backbone": backbone,
            "encoder": encoder,
            "decoder": decoder,
        }

    def run(self):
        """Run the full curriculum."""
        phases = self.config.phases
        if len(phases) >= 1:
            self.run_phase1(phases[0])
        if len(phases) >= 2:
            self.run_phase2(phases[1])
        if len(phases) >= 3:
            self.run_phase3(phases[2])
        # Phase 4 is task-specific, handled by user code

    def _transfer_weights(
        self,
        src_encoder: FieldEncoder,
        dst_encoder: FieldEncoder,
        src_decoder: FieldDecoder,
        dst_decoder: FieldDecoder,
    ):
        """Transfer encoder/decoder weights where field names match."""
        for name, param in src_encoder.encoders.named_parameters():
            if name in dict(dst_encoder.encoders.named_parameters()):
                dst_param = dict(dst_encoder.encoders.named_parameters())[name]
                if dst_param.shape == param.shape:
                    dst_param.data.copy_(param.data)

        for name, param in src_decoder.decoders.named_parameters():
            if name in dict(dst_decoder.decoders.named_parameters()):
                dst_param = dict(dst_decoder.decoders.named_parameters())[name]
                if dst_param.shape == param.shape:
                    dst_param.data.copy_(param.data)

    def _save_checkpoint(
        self,
        name: str,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """Save a training checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "backbone": backbone.state_dict(),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        }, path)
