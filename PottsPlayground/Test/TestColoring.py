import PottsPlayground
import random
import numpy

print("Testing: PottsPlayground.Tasks.GraphColoring")

#test that unconnected graphs work fine
PottsTask = PottsPlayground.Tasks.GraphColoring(ncolors=3, p=0, nnodes=10)

