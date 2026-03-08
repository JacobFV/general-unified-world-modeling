"""Layer 19: The World (top-level composition).

General Unified World Model. Composes all layers into a single typed
causal ontology that can be compiled to a canvas schema.

Dense always-on:  physical, resources, financial, narratives,
                  technology, regime, events, trust
Sparse on-demand: countries, sectors, supply chains, firms, individuals

Usage:
    from general_unified_world_model import World
    from canvas_engineering import compile_schema, ConnectivityPolicy

    world = World()
    bound = compile_schema(
        world,
        T=1, H=128, W=128, d_model=64,
        connectivity=ConnectivityPolicy(intra="dense"),
    )
"""

from dataclasses import dataclass, field as dc_field

from general_unified_world_model.schema.physical import PlanetaryPhysicalLayer
from general_unified_world_model.schema.resources import ResourceLayer
from general_unified_world_model.schema.financial import GlobalFinancialLayer
from general_unified_world_model.schema.narrative import NarrativeBeliefLayer
from general_unified_world_model.schema.technology import TechnologyLayer
from general_unified_world_model.schema.biology import BiologicalLayer
from general_unified_world_model.schema.infrastructure import InfrastructureLayer
from general_unified_world_model.schema.cyber import CyberLayer
from general_unified_world_model.schema.space import SpaceLayer
from general_unified_world_model.schema.health import HealthLayer
from general_unified_world_model.schema.education import EducationLayer
from general_unified_world_model.schema.legal import LegalLayer
from general_unified_world_model.schema.country import Country
from general_unified_world_model.schema.sector import Sector
from general_unified_world_model.schema.supply_chain import SupplyChainNode
from general_unified_world_model.schema.business import Business
from general_unified_world_model.schema.individual import Individual
from general_unified_world_model.schema.events import EventTape
from general_unified_world_model.schema.trust import DataChannelTrust
from general_unified_world_model.schema.regime import RegimeState
from general_unified_world_model.schema.intervention import InterventionSpace
from general_unified_world_model.schema.forecast import ForecastBundle


@dataclass
class World:
    """General Unified World Model.

    Dense always-on:  physical, resources, financial, narratives,
                      technology, regime, events, trust
    Sparse on-demand: countries, sectors, supply chains, firms, individuals
    """
    # Slow structural substrate
    physical:       PlanetaryPhysicalLayer = dc_field(default_factory=PlanetaryPhysicalLayer)
    resources:      ResourceLayer          = dc_field(default_factory=ResourceLayer)
    technology:     TechnologyLayer        = dc_field(default_factory=TechnologyLayer)

    # Biological & ecological systems
    biology:        BiologicalLayer        = dc_field(default_factory=BiologicalLayer)

    # Physical & digital infrastructure
    infrastructure: InfrastructureLayer    = dc_field(default_factory=InfrastructureLayer)

    # Cybersecurity & digital threats
    cyber:          CyberLayer             = dc_field(default_factory=CyberLayer)

    # Space & orbital systems
    space:          SpaceLayer             = dc_field(default_factory=SpaceLayer)

    # Healthcare systems
    health:         HealthLayer            = dc_field(default_factory=HealthLayer)

    # Education & human capital
    education:      EducationLayer         = dc_field(default_factory=EducationLayer)

    # Legal & regulatory framework
    legal:          LegalLayer             = dc_field(default_factory=LegalLayer)

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
