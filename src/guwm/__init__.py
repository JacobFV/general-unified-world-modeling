"""Grand Unified World Model — structured latent space for civilization-scale modeling.

Built on canvas-engineering: prompt engineering for latent space.

Quick start:
    from guwm import World, WorldProjection, project

    # Full world model
    world = World()

    # Or project to a subset
    proj = WorldProjection(include=["financial", "regime", "forecasts"])
    bound = project(proj, T=1, H=64, W=64, d_model=64)
"""

__version__ = "0.1.0"

# Schema
from guwm.schema.world import World

# Projection
from guwm.projection.subset import WorldProjection, project
from guwm.projection.temporal import TemporalTopology, TemporalEntity
from guwm.projection.transfer import TransferDistanceEstimator

# Training
from guwm.training.heterogeneous import (
    HeterogeneousDataset,
    MaskedCanvasTrainer,
    DatasetSpec,
    FieldMapping,
    FieldEncoder,
    FieldDecoder,
    build_mixed_dataloader,
)
from guwm.training.backbone import WorldModelBackbone, build_world_model
from guwm.training.curriculum import CurriculumTrainer, CurriculumConfig
from guwm.training.diffusion import DiffusionWorldModelTrainer, CosineNoiseSchedule

# Inference
from guwm.inference import WorldModel
