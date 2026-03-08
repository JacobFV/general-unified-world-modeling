"""General Unified World Model — structured latent space for civilization-scale modeling.

Built on canvas-engineering: prompt engineering for latent space.

Quick start:
    from general_unified_world_model import World, WorldProjection, project

    # Full world model (857 fields, 19 layers)
    world = World()

    # Or project to a subset
    proj = WorldProjection(include=["financial", "regime", "forecasts"])
    bound = project(proj, T=1, H=64, W=64, d_model=64)
"""

__version__ = "0.0.1"

# Schema
from general_unified_world_model.schema.world import World

# Projection
from general_unified_world_model.projection.subset import WorldProjection, project
from general_unified_world_model.projection.temporal import TemporalTopology, TemporalEntity
from general_unified_world_model.projection.transfer import TransferDistanceEstimator

# Training
from general_unified_world_model.training.heterogeneous import (
    HeterogeneousDataset,
    MaskedCanvasTrainer,
    DatasetSpec,
    FieldMapping,
    FieldEncoder,
    FieldDecoder,
    build_mixed_dataloader,
)
from general_unified_world_model.training.backbone import WorldModelBackbone, build_world_model
from general_unified_world_model.training.curriculum import CurriculumTrainer, CurriculumConfig
from general_unified_world_model.training.diffusion import DiffusionWorldModelTrainer, CosineNoiseSchedule

# Inference
from general_unified_world_model.inference import WorldModel
