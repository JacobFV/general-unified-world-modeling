"""Temporal topology: handle entities that appear/disappear over time.

The real world doesn't have a fixed set of entities. Companies are founded,
people enter and leave positions, countries form and dissolve. The canvas
schema needs to handle this gracefully.

Solution: temporal_extent on fields/entities. When compiling, fields with
temporal constraints are placed on the canvas only for their active period.
The attention mask prevents future information from leaking into past predictions.

Example:
    Apple Inc was founded in 1976. If our training data starts in 1960,
    Apple's fields should only appear after 1976 (tick offset 16 * 365 * 16
    in sub-minute ticks, but practically we'd use a coarser tick).

    More practically: in a daily-tick model, Apple appears at tick
    (1976 - 1960) * 252 ≈ tick 4032.

This is handled by:
1. Setting temporal_extent on the Field declarations
2. The compiler places these fields only for the active timesteps
3. The attention mask blocks attention from inactive periods
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Optional

from canvas_engineering import Field


@dataclass
class TemporalEntity:
    """An entity with a temporal existence range.

    Args:
        name: Entity name (e.g. "firm_AAPL", "person_cook").
        start_tick: First tick this entity exists (inclusive).
        end_tick: Last tick this entity exists (exclusive). None = forever.
        obj: The schema object (Business, Individual, etc.).
    """
    name: str
    start_tick: int = 0
    end_tick: Optional[int] = None
    obj: object = None


@dataclass
class TemporalTopology:
    """Manages temporal constraints for world model entities.

    Tracks which entities are active at which timesteps and generates
    appropriate attention masks and loss masks.
    """
    entities: list[TemporalEntity] = dc_field(default_factory=list)

    def add(self, name: str, obj: object, start_tick: int = 0, end_tick: int | None = None):
        """Register a temporally-bounded entity."""
        self.entities.append(TemporalEntity(
            name=name, start_tick=start_tick, end_tick=end_tick, obj=obj,
        ))

    def active_at(self, tick: int) -> list[TemporalEntity]:
        """Return entities active at a given tick."""
        return [
            e for e in self.entities
            if e.start_tick <= tick and (e.end_tick is None or tick < e.end_tick)
        ]

    def generate_presence_mask(self, tick: int, bound_schema, device: str = "cpu"):
        """Generate a presence mask for a given tick.

        Positions belonging to inactive entities get masked to 0.

        Args:
            tick: Current tick.
            bound_schema: Compiled BoundSchema.
            device: Torch device.

        Returns:
            (N,) tensor with 1.0 for active positions, 0.0 for inactive.
        """
        import torch

        n_positions = bound_schema.layout.num_positions
        mask = torch.ones(n_positions, device=device)

        # Find which entity prefixes are inactive at this tick
        active_names = {e.name for e in self.active_at(tick)}
        all_names = {e.name for e in self.entities}
        inactive_names = all_names - active_names

        # Zero out positions belonging to inactive entities
        for name in inactive_names:
            for field_name in bound_schema.field_names:
                if field_name.startswith(name + ".") or field_name.startswith(name + "["):
                    bf = bound_schema[field_name]
                    for idx in bf.indices():
                        if idx < n_positions:
                            mask[idx] = 0.0

        return mask

    def generate_temporal_attention_mask(
        self,
        tick_range: tuple[int, int],
        bound_schema,
        device: str = "cpu",
    ):
        """Generate a temporal attention mask over a tick range.

        This creates a block-diagonal-like mask where positions belonging
        to entities that don't exist yet at a given tick cannot attend
        to or be attended by other positions.

        For training on historical data with entities that appear/disappear,
        this prevents the model from leaking information across temporal boundaries.

        Args:
            tick_range: (start_tick, end_tick) range to generate mask for.
            bound_schema: Compiled BoundSchema.
            device: Torch device.

        Returns:
            (N, N) additive attention mask (-inf for blocked, 0 for allowed).
        """
        import torch

        n_positions = bound_schema.layout.num_positions
        # Start with the topology mask if available
        if bound_schema.topology is not None:
            mask = bound_schema.topology.to_additive_mask(bound_schema.layout, device=device)
        else:
            mask = torch.zeros(n_positions, n_positions, device=device)

        # For each tick in range, determine which entities are active
        # and block attention to/from inactive entity positions
        all_entity_names = {e.name for e in self.entities}

        for tick in range(tick_range[0], tick_range[1]):
            active = {e.name for e in self.active_at(tick)}
            inactive = all_entity_names - active

            for name in inactive:
                inactive_indices = set()
                for field_name in bound_schema.field_names:
                    if field_name.startswith(name + ".") or field_name.startswith(name + "["):
                        bf = bound_schema[field_name]
                        inactive_indices.update(bf.indices())

                # Block attention to/from inactive positions
                for idx in inactive_indices:
                    if idx < n_positions:
                        mask[idx, :] = float("-inf")
                        mask[:, idx] = float("-inf")
                        mask[idx, idx] = 0.0  # self-attention always allowed

        return mask
