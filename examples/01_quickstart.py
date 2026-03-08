"""Quickstart: compile the full World schema and inspect it.

This is the simplest possible usage — instantiate World(), compile it,
and see what you get. No training, no data, just the type system at work.

Run: python examples/01_quickstart.py
"""

from canvas_engineering import compile_schema, ConnectivityPolicy
from general_unified_world_model.schema.world import World

# Instantiate the full world model
world = World()

# Compile to canvas — this walks the entire type hierarchy,
# packs all fields onto a (T, H, W) grid, and generates
# the connectivity topology from the type structure.
bound = compile_schema(
    world,
    T=1,           # single tick (expand for temporal modeling)
    H=128,         # spatial height
    W=128,         # spatial width
    d_model=64,    # latent dimension per position
    connectivity=ConnectivityPolicy(
        intra="dense",           # all fields within a type attend to each other
        parent_child="hub_spoke", # parent ↔ all children
        array_element="ring",     # array elements form a ring
        temporal="dense",         # all timesteps attend to all timesteps
    ),
)

print("=" * 70)
print("GRAND UNIFIED WORLD MODEL — Canvas Schema")
print("=" * 70)
print(f"Total fields:      {len(bound.field_names)}")
print(f"Canvas positions:  {bound.layout.num_positions}")
print(f"Connections:       {len(bound.topology.connections)}")
print(f"d_model:           {bound.layout.d_model}")
print(f"Grid:              T={bound.layout.T}, H={bound.layout.H}, W={bound.layout.W}")
print()

# Show field hierarchy
print("Field paths (first 50):")
for i, name in enumerate(sorted(bound.field_names)):
    if i >= 50:
        print(f"  ... and {len(bound.field_names) - 50} more")
        break
    bf = bound[name]
    print(f"  {name:60s} → {bf.num_positions:>4d} positions")

print()
print("Layer breakdown:")
layers = {}
for name in bound.field_names:
    top = name.split(".")[0]
    bf = bound[name]
    layers[top] = layers.get(top, 0) + bf.num_positions
for layer, n_pos in sorted(layers.items(), key=lambda x: -x[1]):
    pct = n_pos / bound.layout.num_positions * 100
    print(f"  {layer:30s} {n_pos:>6d} positions ({pct:5.1f}%)")
