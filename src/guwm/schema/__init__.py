"""Schema declarations for the General Unified World Model.

The full ontology lives in submodules organized by domain layer.
Import `World` for the top-level composition.
"""

from guwm.schema.world import World
from guwm.schema.observability import ObservedFast, ObservedDaily, ObservedSlow
from guwm.schema.physical import PlanetaryPhysicalLayer
from guwm.schema.resources import ResourceLayer
from guwm.schema.financial import GlobalFinancialLayer
from guwm.schema.macro import MacroEconomy
from guwm.schema.political import PoliticalLayer
from guwm.schema.narrative import NarrativeBeliefLayer
from guwm.schema.technology import TechnologyLayer
from guwm.schema.demographics import DemographicLayer
from guwm.schema.sector import Sector
from guwm.schema.supply_chain import SupplyChainNode
from guwm.schema.business import Business
from guwm.schema.individual import Individual
from guwm.schema.events import EventTape
from guwm.schema.trust import DataChannelTrust
from guwm.schema.regime import RegimeState
from guwm.schema.intervention import InterventionSpace
from guwm.schema.forecast import ForecastBundle
from guwm.schema.country import Country
