import PottsPlayground
import random
import numpy

print("Testing: PottsPlayground.Tasks.GraphColoring")

#test that unconnected graphs work fine
PottsTask = PottsPlayground.Tasks.GraphColoring(ncolors=3, p=0, nnodes=4)
valid_energies, invalid_energies = PottsTask.EnergyBands()
assert numpy.sum(valid_energies) == 0

PottsTask = PottsPlayground.Tasks.GraphColoring(ncolors=1, p=1.0, nnodes=4)
valid_energies, invalid_energies = PottsTask.EnergyBands()
assert numpy.sum(valid_energies) == 6

PottsTask = PottsPlayground.Tasks.GraphColoring(ncolors=4, p=1.0, nnodes=4)
valid_energies, invalid_energies = PottsTask.EnergyBands()
assert len(valid_energies) == 4**4
assert numpy.sum(numpy.array(valid_energies) == 0) == 4*3*2*1 