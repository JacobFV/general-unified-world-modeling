"""General Unified World Model — structured latent space for civilization-scale modeling.

Built on canvas-engineering: prompt engineering for latent space.

Quick start:
    from general_unified_world_model import World, project, GeneralUnifiedWorldModel

    # Clean API: schema root + include
    bound = project(World(), include=["financial", "regime", "forecasts"])

    # Or use the convenience class directly
    model = GeneralUnifiedWorldModel(include=["financial", "regime"])
    model.observe("financial.yield_curves.ten_year", 4.25)
    predictions = model.predict()
"""

__version__ = "0.0.3"

# Schema
from general_unified_world_model.schema.world import World

# Projection
from general_unified_world_model.projection.subset import project
from general_unified_world_model.projection.temporal import TemporalTopology, TemporalEntity
from general_unified_world_model.projection.transfer import TransferDistanceEstimator

# Training
from general_unified_world_model.training.heterogeneous import (
    HeterogeneousDataset,
    MaskedCanvasTrainer,
    DatasetSpec,
    DataSource,
    CoverageReport,
    check_coverage,
    FieldEncoder,
    FieldDecoder,
    InputSpec,
    OutputSpec,
    build_mixed_dataloader,
)
from general_unified_world_model.training.backbone import WorldModelBackbone, build_world_model
from general_unified_world_model.training.curriculum import CurriculumTrainer, CurriculumConfig
from general_unified_world_model.training.diffusion import DiffusionWorldModelTrainer, CosineNoiseSchedule
from general_unified_world_model.training.dag_curriculum import (
    DAGCurriculumTrainer,
    TrainingNode,
    CurriculumSpec,
    Stage,
    StagesInParallel,
    DatasetProfile,
    build_curriculum,
)

# Inference
from general_unified_world_model.inference import WorldModel, GeneralUnifiedWorldModel

# Rendering (lazy — only loaded when accessed, needs matplotlib)
def render(bound_schema, renderer, **kwargs):
    from general_unified_world_model.rendering.base import render as _render
    return _render(bound_schema, renderer, **kwargs)

# LLM projection builder (lazy — only loaded when accessed, needs API key)
def llm_project(description, **kwargs):
    from general_unified_world_model.llm.projection_builder import llm_project as _llm_project
    return _llm_project(description, **kwargs)
