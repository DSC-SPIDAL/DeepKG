# =============================================================================
# HDK V1.5: Core (BaseHarvester Interface)
# Matches the definition in Jan24_2026HDKupdatedV2.ipynb
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any

# Import State for type hinting if available
try:
    from deepcollector.core.state import CatalogState
except ImportError:
    CatalogState = Any

class BaseHarvester(ABC):
    """
    Abstract Base Class defining the interface required for all harvesters.
    """
    def __init__(self, config: Any, tools: Any):
        self.config = config
        self.tools = tools
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)

    @abstractmethod
    def execute_harvest(self, state: CatalogState) -> bool:
        """
        Executes the main harvesting workflow.
        Must update the provided CatalogState object.
        Returns True on success, False on critical failure.
        """
        pass

print("✅ deepcollector/harvesting/base_harvester.py written (HDK Compatible).")