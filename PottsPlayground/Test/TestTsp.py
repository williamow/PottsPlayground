import PottsPlayground
import random
import numpy

print("Testing: PottsPlayground.Tasks.TravelingSalesman")

PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=5, ConstraintFactor=10)
valid_energies, invalid_energies = PottsTask.EnergyBands()

exp_val = 5*4*3*2*1
if len(valid_energies) != exp_val:
	print("FAILURE: expected %i valid states, but there are %i"%(exp_val, len(valid_energies)))
	exit()

if numpy.max(valid_energies) > numpy.min(invalid_energies):
	print("FAILURE: invalid energies should all be greater than valid energies")
	exit()

dists = numpy.array(
	[[0,1,10,1],
	 [1,0,1,10],
	 [10,1,0,1],
	 [1,10,1,0]]
	)
PottsTask = PottsPlayground.Tasks.TravelingSalesman(costs=dists, ConstraintFactor=10)
valid_energies, invalid_energies = PottsTask.EnergyBands()
if numpy.min(valid_energies) != 4:
	print("FAILURE: incorrect min energy")
	exit()
if numpy.max(valid_energies) != 22:
	print("FAILURE: incorrect max energy")
	exit()

