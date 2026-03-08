"""Grand Unified World Model — structured latent space for civilization-scale modeling.

Built on canvas-engineering: prompt engineering for latent space.
"""

__version__ = "0.1.0"

from guwm.schema.world import World
from guwm.projection.subset import WorldProjection, project
from guwm.training.heterogeneous import HeterogeneousDataset, MaskedCanvasTrainer
