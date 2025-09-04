import PottsPlayground
import random
import numpy

print("Testing: PottsPlayground.Tasks.Binarized")

#test that unconnected graphs work fine
PottsTask = PottsPlayground.Tasks.BaseTask()
PottsTask.AddSpins([2,2,3,3])

def k(n, i, j):
	if n:
		return "rndk-%i-%i"%(i,j)
	else:
		return numpy.random.rand(i,j)

PottsTask.AddWeight(lambda n: k(n, 2, 2), 0, 1, 1.)
PottsTask.AddWeight(lambda n: k(n, 2, 2), 0, 1, 1.) #add it a second time, just to check multiedge functionality
PottsTask.AddWeight(lambda n: k(n, 2, 3), 1, 2, 2.)
PottsTask.AddWeight(lambda n: k(n, 3, 3), 2, 3, 3.)

PottsTask.CompileKernels()

IsingTask = PottsPlayground.Tasks.Binarized(PottsTask)


assert len(IsingTask.ListSpins()) == 8
p_valid, p_invalid = PottsTask.EnergyBands()
i_valid, i_invalid = IsingTask.EnergyBands()
assert len(p_valid) == len(i_valid)

#check energy of a Potts state and equivalent ising state:
pottsstate = numpy.array([1, 0, 2, 1], dtype='int32')
isingstate = IsingTask.PottsToIsingSemantics(pottsstate)

#check that min valid energies are the same
assert (numpy.min(p_valid) + 1e-2 > numpy.min(i_valid))
assert (numpy.min(p_valid) - 1e-2 < numpy.min(i_valid))

#check that invalid energies are all higher than the lowest valid energy:
# assert (numpy.min(i_valid) + 1 - 1e-2 < numpy.min(i_invalid))