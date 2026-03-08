"""Layer 19: The World (top-level composition).

Grand Unified World Model. Composes all layers into a single typed
causal ontology that can be compiled to a canvas schema.

Dense always-on:  physical, resources, financial, narratives,
                  technology, regime, events, trust
Sparse on-demand: countries, sectors, supply chains, firms, individuals

Usage:
    from guwm import World
    from canvas_engineering import compile_schema, ConnectivityPolicy

    world = World()
    bound = compile_schema(
        world,
        T=1, H=128, W=128, d_model=64,
        connectivity=ConnectivityPolicy(
            intra="dense",
            parent_child="hub_spoke",
        ),
    )
"""

from dataclasses import dataclass, field as dc_field

from guwm.schema.physical import PlanetaryPhysicalLayer
from guwm.schema.resources import ResourceLayer
from guwm.schema.financial import GlobalFinancialLayer
from guwm.schema.narrative import NarrativeBeliefLayer
from guwm.schema.technology import TechnologyLayer
from guwm.schema.country import Country
from guwm.schema.sector import Sector
from guwm.schema.supply_chain import SupplyChainNode
from guwm.schema.business import Business
from guwm.schema.individual import Individual
from guwm.schema.events import EventTape
from guwm.schema.trust import DataChannelTrust
from guwm.schema.regime import RegimeState
from guwm.schema.intervention import InterventionSpace
from guwm.schema.forecast import ForecastBundle


@dataclass
class World:
    """Grand Unified World Model.

    Dense always-on:  physical, resources, financial, narratives,
                      technology, regime, events, trust
    Sparse on-demand: countries, sectors, supply chains, firms, individuals
    """
    # Slow structural substrate
    physical:       PlanetaryPhysicalLayer = dc_field(default_factory=PlanetaryPhysicalLayer)
    resources:      ResourceLayer          = dc_field(default_factory=ResourceLayer)
    technology:     TechnologyLayer        = dc_field(default_factory=TechnologyLayer)

    # High-bandwidth financial core
    financial:      GlobalFinancialLayer   = dc_field(default_factory=GlobalFinancialLayer)

    # Collective cognition
    narratives:     NarrativeBeliefLayer   = dc_field(default_factory=NarrativeBeliefLayer)

    # Country blocks (representative subset; scale to ~20 in production)
    country_us:     Country = dc_field(default_factory=Country)
    country_cn:     Country = dc_field(default_factory=Country)
    country_eu:     Country = dc_field(default_factory=Country)

    # Sector blocks (representative subset; scale to 11 GICS in production)
    sector_tech:        Sector = dc_field(default_factory=Sector)
    sector_energy:      Sector = dc_field(default_factory=Sector)
    sector_financials:  Sector = dc_field(default_factory=Sector)

    # Supply chain critical nodes
    sc_semiconductors:  SupplyChainNode = dc_field(default_factory=SupplyChainNode)
    sc_energy:          SupplyChainNode = dc_field(default_factory=SupplyChainNode)
    sc_food:            SupplyChainNode = dc_field(default_factory=SupplyChainNode)

    # Strategic businesses (sparse)
    firm_alpha:     Business   = dc_field(default_factory=Business)
    firm_beta:      Business   = dc_field(default_factory=Business)

    # Strategic individuals (very sparse)
    person_alpha:   Individual = dc_field(default_factory=Individual)
    person_beta:    Individual = dc_field(default_factory=Individual)

    # Real-time event stream
    events:         EventTape         = dc_field(default_factory=EventTape)

    # Epistemic calibration
    trust:          DataChannelTrust  = dc_field(default_factory=DataChannelTrust)

    # Privileged regime latent
    regime:         RegimeState       = dc_field(default_factory=RegimeState)

    # Intervention & counterfactual
    interventions:  InterventionSpace = dc_field(default_factory=InterventionSpace)

    # Forecast outputs
    forecasts:      ForecastBundle    = dc_field(default_factory=ForecastBundle)
