"""Training infrastructure for heterogeneous, multi-frequency world model training."""

from general_unified_world_model.training.backbone import WorldModelBackbone, build_world_model
from general_unified_world_model.training.heterogeneous import (
    HeterogeneousDataset,
    MaskedCanvasTrainer,
    DatasetSpec,
    FieldMapping,
    FieldEncoder,
    FieldDecoder,
    build_mixed_dataloader,
)
from general_unified_world_model.training.diffusion import DiffusionWorldModelTrainer, CosineNoiseSchedule
from general_unified_world_model.training.dag_curriculum import DAGCurriculumTrainer, TrainingNode
