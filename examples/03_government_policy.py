"""Government agency: "Model the macroeconomic impact of this policy decision."

A government analyst needs to understand how a monetary policy change
propagates through the economy. They declare the relevant subgraph:
macro economy, financial system, trade, fiscal, and the intervention space
for counterfactual analysis.

Run: python examples/03_government_policy.py
"""

from canvas_engineering import ConnectivityPolicy
from general_unified_world_model.projection.subset import project
from general_unified_world_model.schema.country import Country

# Government analyst's specification
bound = project(
    include=[
        # Full US macroeconomy
        "country_us",

        # Other major economies (for trade/contagion effects)
        "country_cn.macro",
        "country_eu.macro",

        # Financial system (transmission mechanism)
        "financial",

        # Resources (commodity channel)
        "resources.energy",
        "resources.food",

        # Regime state
        "regime",

        # Intervention space — this is the key
        "interventions",

        # Full forecast bundle
        "forecasts",

        # Epistemic calibration
        "trust",

        # Narratives (policy credibility matters)
        "narratives.elites.cb_hawkishness",
        "narratives.public.economic_anxiety",
        "narratives.public.consumer_confidence",
        "narratives.public.institutional_trust",
    ],

    # Add extra countries for trade analysis
    entities={
        "country_jp": Country(),
        "country_uk": Country(),
        "country_in": Country(),
    },

    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        temporal="dense",
    ),

    T=1, H=96, W=96, d_model=64,
)

print("=" * 70)
print("Government Policy Analysis — Canvas Schema")
print("=" * 70)
print(f"Total fields:      {len(bound.field_names)}")
print(f"Canvas positions:  {bound.layout.num_positions}")
print(f"Connections:       {len(bound.topology.connections)}")
print()

# Show the intervention → forecast causal chain
print("Causal chain: Intervention → Transmission → Outcome")
print()

intervention_fields = [f for f in bound.field_names if f.startswith("interventions")]
print(f"  Interventions ({len(intervention_fields)} fields):")
for f in sorted(intervention_fields):
    print(f"    {f}")

forecast_fields = [f for f in bound.field_names if f.startswith("forecasts")]
print(f"\n  Forecasts ({len(forecast_fields)} fields):")
for f in sorted(forecast_fields):
    print(f"    {f}")

print()
print("Key analysis capabilities:")
print("  - Set monetary_policy_change → observe effect on yield curve")
print("  - Set fiscal_policy_change → observe effect on GDP, inflation")
print("  - Cross-country contagion via trade and financial channels")
print("  - Decomposed uncertainty: aleatoric vs epistemic")
print("  - Counterfactual heads: 3-month and 12-month effect estimates")
