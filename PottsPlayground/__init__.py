#automatically imports all of the key submodule classes/functions into the PottsPlayground namespace
from PottsPlayground.Annealing import Anneal

import PottsPlayground.Schedules as Schedules
import PottsPlayground.Kernels as Kernels

import PottsPlayground.Tasks as Tasks #tasks relies on Kernels, so it needs to come after

# from PottsPlayground.Tasks.BaseTask import Base
# from PottsPlayground.Tasks.Binarized import Binarize
# from PottsPlayground.Tasks.MinorEmbedded import MinorEmbedded
# from PottsPlayground.Tasks.TravelingSalesman import TravelingSalesman
# from PottsPlayground.Tasks.GraphColoring import GraphColoring
# from PottsPlayground.Tasks.Ice40Placer import Ice40Placer
# from PottsPlayground.Tasks.InvertibleLogic import InvertibleLogic