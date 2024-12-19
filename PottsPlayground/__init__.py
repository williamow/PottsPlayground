__version__ = "0.1.1"

#automatically imports all of the key submodule classes/functions into the PottsPlayground namespace
from PottsPlayground.Annealing import Anneal

import PottsPlayground.Schedules as Schedules
import PottsPlayground.Kernels as Kernels

import PottsPlayground.Tasks as Tasks #tasks relies on Kernels, so it needs to come after