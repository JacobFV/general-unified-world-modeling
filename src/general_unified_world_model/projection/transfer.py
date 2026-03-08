"""Transfer distance estimation between world model fields.

The key insight: if canvas schemas produce stable latent representations,
then the semantic distance between two field descriptions approximates
the real cost of bridging them — how many adapter layers, how much data.

Two fields that describe similar things (GDP and industrial production)
will have correlated latent dynamics. Two fields that describe very
different things (GDP and seismic risk) will be nearly independent.

This lets us:
1. Prioritize which domain couplings to train first (start with close pairs)
2. Estimate how much data we need for each coupling
3. Design adapter architectures (close = linear, far = deep MLP)
4. Validate learned representations against semantic priors
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ── Default semantic descriptions ────────────────────────────────────────

# These are auto-generated from field paths. Users can override with
# explicit semantic_type on their RegionSpec/Field declarations.

def _path_to_description(field_path: str) -> str:
    """Convert a dotted field path to a human-readable semantic description.

    'country_us.macro.inflation.headline_cpi' →
    'US macroeconomic headline CPI inflation rate'
    """
    parts = field_path.replace("_", " ").split(".")

    # Handle country prefixes
    country_map = {"us": "US", "cn": "Chinese", "eu": "European", "jp": "Japanese",
                   "uk": "UK", "in": "Indian", "kr": "Korean"}
    result_parts = []
    for part in parts:
        # Check for country prefix
        if part.startswith("country "):
            code = part.split(" ")[1]
            result_parts.append(country_map.get(code, code.upper()))
        elif part.startswith("firm "):
            result_parts.append(f"firm {part.split(' ')[1]}")
        elif part.startswith("person "):
            result_parts.append(f"individual {part.split(' ')[1]}")
        elif part.startswith("sector "):
            result_parts.append(f"{part.split(' ')[1]} sector")
        elif part.startswith("sc "):
            result_parts.append(f"supply chain {part.split(' ')[1]}")
        else:
            result_parts.append(part)

    return " ".join(result_parts)


# ── Embedding-based distance ────────────────────────────────────────────

class TransferDistanceEstimator:
    """Estimates transfer distance between world model fields.

    Uses semantic embeddings to compute pairwise distances between fields.
    The embeddings can come from:
    1. A fixed text embedding model (deterministic, reproducible)
    2. Cached embeddings loaded from file
    3. Manually specified embeddings

    The distance metric is cosine distance in the embedding space.
    """

    def __init__(
        self,
        bound_schema=None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: str | None = None,
    ):
        self.bound = bound_schema
        self.embedding_model_name = embedding_model
        self.cache_path = cache_path
        self._embeddings: dict[str, torch.Tensor] = {}
        self._model = None

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers for automatic embeddings: "
                    "pip install sentence-transformers"
                )
        return self._model

    def embed(self, text: str) -> torch.Tensor:
        """Get embedding for a text description."""
        if text in self._embeddings:
            return self._embeddings[text]

        model = self._get_model()
        emb = model.encode(text, convert_to_tensor=True)
        self._embeddings[text] = emb
        return emb

    def embed_field(self, field_path: str) -> torch.Tensor:
        """Get embedding for a field path."""
        desc = _path_to_description(field_path)
        return self.embed(desc)

    def distance(self, field_a: str, field_b: str) -> float:
        """Compute transfer distance between two fields.

        Returns cosine distance in [0, 2]. Lower = more similar.
        """
        emb_a = self.embed_field(field_a)
        emb_b = self.embed_field(field_b)
        cos_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0))
        return (1 - cos_sim.item())

    def distance_matrix(self, field_paths: list[str] | None = None) -> tuple[list[str], torch.Tensor]:
        """Compute pairwise distance matrix for all or selected fields.

        Args:
            field_paths: Specific fields to compare. None = all fields.

        Returns:
            (field_names, distance_matrix) where matrix is (N, N).
        """
        if field_paths is None and self.bound is not None:
            field_paths = list(self.bound.field_names)

        embeddings = torch.stack([self.embed_field(f) for f in field_paths])
        # Cosine distance matrix
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1
        )
        dist_matrix = 1 - sim_matrix

        return field_paths, dist_matrix

    def nearest_neighbors(self, field_path: str, k: int = 10) -> list[tuple[str, float]]:
        """Find k nearest fields to a given field.

        Args:
            field_path: Source field.
            k: Number of neighbors.

        Returns:
            List of (field_path, distance) sorted by distance.
        """
        if self.bound is None:
            raise ValueError("Need bound_schema for nearest neighbor search")

        source_emb = self.embed_field(field_path)
        all_fields = list(self.bound.field_names)

        distances = []
        for f in all_fields:
            if f == field_path:
                continue
            emb = self.embed_field(f)
            cos_sim = F.cosine_similarity(source_emb.unsqueeze(0), emb.unsqueeze(0))
            distances.append((f, 1 - cos_sim.item()))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def domain_coupling_priority(self) -> list[tuple[str, str, float]]:
        """Suggest domain coupling order based on inter-domain distances.

        For each pair of top-level domains, computes the mean transfer distance
        between their fields. Returns pairs sorted by distance (closest first).

        This informs the curriculum: couple close domains first, distant later.

        Returns:
            List of (domain_a, domain_b, mean_distance) sorted by distance.
        """
        if self.bound is None:
            raise ValueError("Need bound_schema for coupling priority")

        # Group fields by top-level domain
        domains: dict[str, list[str]] = {}
        for name in self.bound.field_names:
            top = name.split(".")[0]
            domains.setdefault(top, []).append(name)

        # Compute inter-domain distances
        pairs = []
        domain_names = sorted(domains.keys())
        for i, da in enumerate(domain_names):
            for db in domain_names[i + 1:]:
                # Sample fields for efficiency (cap at 20 per domain)
                fa = domains[da][:20]
                fb = domains[db][:20]

                total_dist = 0
                count = 0
                for a in fa:
                    for b in fb:
                        total_dist += self.distance(a, b)
                        count += 1

                mean_dist = total_dist / max(count, 1)
                pairs.append((da, db, mean_dist))

        pairs.sort(key=lambda x: x[2])
        return pairs


# ── Convenience function ─────────────────────────────────────────────────

def estimate_adapter_depth(distance: float) -> int:
    """Estimate the adapter depth needed to bridge a transfer distance.

    Based on empirical heuristics:
    - distance < 0.2: linear projection (1 layer)
    - distance < 0.4: shallow MLP (2 layers)
    - distance < 0.6: moderate MLP (3 layers)
    - distance >= 0.6: deep MLP (4+ layers)

    Args:
        distance: Cosine distance between field embeddings.

    Returns:
        Suggested number of adapter layers.
    """
    if distance < 0.2:
        return 1
    elif distance < 0.4:
        return 2
    elif distance < 0.6:
        return 3
    else:
        return 4
