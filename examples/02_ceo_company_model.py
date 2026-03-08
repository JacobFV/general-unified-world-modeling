"""CEO use case: "I want to model my company, our employees, and our products."

A company CEO tells their engineer: model the firm's financials, operations,
key people, relevant sector dynamics, and the macro environment they operate in.

This example shows how WorldProjection selects the relevant subset of the
full world model and compiles it to a focused canvas.

Run: python examples/02_ceo_company_model.py
"""

from canvas_engineering import ConnectivityPolicy
from general_unified_world_model.projection.subset import WorldProjection, project

# CEO's specification: "Model our company in context"
proj = WorldProjection(
    include=[
        # The macro environment we operate in
        "country_us.macro.output",       # GDP, PMI, retail sales
        "country_us.macro.inflation",    # CPI, PPI, wage growth
        "country_us.macro.labor",        # unemployment, claims, LFPR

        # Our sector
        "sector_tech",                    # tech sector dynamics

        # Financial markets we're exposed to
        "financial.yield_curves",         # interest rates matter for our debt
        "financial.equities",             # our stock + comps
        "financial.credit",               # credit conditions for our bonds

        # Narratives that affect us
        "narratives.elites.techno_optimism",
        "narratives.elites.ceo_confidence",
        "narratives.positioning",         # investor positioning

        # Regime state (always include — it's the compressed summary)
        "regime",

        # Events that matter
        "events",

        # Forecasts we care about
        "forecasts.macro",
        "forecasts.financial",
        "forecasts.business",
    ],

    # Our company and key competitor
    firms=["ACME", "RIVAL"],

    # Key decision makers
    individuals=["ceo", "cfo", "cto", "board_chair"],

    # Custom connectivity
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="dense",     # all individuals see each other
        temporal="dense",
    ),
)

# Compile to canvas
bound = project(proj, T=1, H=64, W=64, d_model=64)

print("=" * 70)
print("CEO Company Model — Canvas Schema")
print("=" * 70)
print(f"Total fields:      {len(bound.field_names)}")
print(f"Canvas positions:  {bound.layout.num_positions}")
print(f"Connections:       {len(bound.topology.connections)}")
print()

# Show what we got
print("Fields by category:")
categories = {}
for name in sorted(bound.field_names):
    cat = name.split(".")[0]
    categories.setdefault(cat, []).append(name)

for cat, fields in sorted(categories.items()):
    print(f"\n  {cat} ({len(fields)} fields):")
    for f in fields[:5]:
        bf = bound[f]
        print(f"    {f} → {bf.num_positions} positions")
    if len(fields) > 5:
        print(f"    ... and {len(fields) - 5} more")

print()
print("Key model capabilities:")
print("  - Firm financials: revenue, margins, FCF, debt metrics")
print("  - Firm operations: capacity, utilization, pricing power")
print("  - Key person psychology: decision style, incentives, network")
print("  - Macro context: GDP, inflation, labor market")
print("  - Market exposure: yields, credit spreads, equity sentiment")
print("  - Latent outputs: firm health, momentum, tail risk, fair value")
