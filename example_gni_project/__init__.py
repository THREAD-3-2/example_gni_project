"""
Support for wildcard import.
"""

from .containers import *
from .time_stepping import *
from .integrators import *
from .plotting import *
from .convergence_study import *


from .containers import __all__ as all_containers
from .time_stepping import __all__ as all_time_stepping
from .integrators import __all__ as all_integrators
from .plotting import __all__ as all_plotting
from .convergence_study import __all__ as all_convergence_study


__all__ = all_containers + all_time_stepping + all_integrators + all_plotting + all_convergence_study
