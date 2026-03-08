"""Computer use agent: model the user's psychology and world context.

A professional's computer use agent needs to understand not just what
the user wants to do right now, but WHY — their incentives, cognitive
state, career pressures, the organizational dynamics they're embedded in.

The world model provides the structured context for intelligent agency.

Run: python examples/04_computer_use_agent.py
"""

from canvas_engineering import ConnectivityPolicy
from general_unified_world_model.projection.subset import project
from general_unified_world_model.schema.individual import Individual
from general_unified_world_model.schema.business import Business

# Computer use agent's world model
bound = project(
    include=[
        # The user's psychological decomposition
        # (projected onto the Individual schema)

        # What's happening in the world right now (context for the user's work)
        "events",                          # breaking news, policy announcements
        "financial.equities",              # if user works in finance
        "narratives.media.crisis_framing", # is there a crisis the user cares about?

        # Regime state (compressed world context)
        "regime.compressed_world_state",

        # Forecasts the user might want to know about
        "forecasts.macro.recession_prob_3m",
        "forecasts.financial.credit_stress_3m",
    ],

    # The user and their organization
    entities={
        "person_user": Individual(),
        "firm_user_org": Business(),
    },

    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        temporal="causal",     # causal temporal: don't leak future context
    ),

    T=1, H=32, W=32, d_model=64,
)

print("=" * 70)
print("Computer Use Agent — World Model Context")
print("=" * 70)
print(f"Total fields:      {len(bound.field_names)}")
print(f"Canvas positions:  {bound.layout.num_positions}")
print(f"Connections:       {len(bound.topology.connections)}")
print()

# Show the user model
user_fields = [f for f in bound.field_names if "user" in f.lower() or "person" in f.lower()]
print(f"User Model ({len(user_fields)} fields):")
for f in sorted(user_fields):
    bf = bound[f]
    print(f"  {f} → {bf.num_positions} positions")

print()
print("Key capabilities:")
print("  - Track user cognitive load, stress, confidence")
print("  - Model user's incentive structure (career, reputation, legal)")
print("  - Understand organizational context via firm model")
print("  - Contextualize user actions with world events")
print("  - Predict what the user will need next (projected_actions)")
print("  - Estimate surprise risk (unexpected user behavior)")
