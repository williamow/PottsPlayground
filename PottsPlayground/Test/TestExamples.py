print("Testing: Examples")


#manually create an Ising model of a logical AND gate, C = A&B

import numpy
import PottsPlayground
from PottsPlayground.Kernels import BinaryQuadratic as BQK

andgate = PottsPlayground.Tasks.BaseTask()

#three variables, each which take on values of 0 or 1
andgate.AddSpins([2,2,2], ['A', 'B', 'C'])

#BQK is the binary quadratic kernel,
#which allows the simplification of the Potts Hamiltonian to the Ising Hamiltonian:
#Potts: E = sum Wij(mi,mj) -> 
#Binary Quadratic Ising: E = sum Wij*mi*mj with m valued as 0 or 1
andgate.AddWeight(BQK, 'A', 'B', weight=1)
andgate.AddWeight(BQK, 'A', 'C', weight=-2)
andgate.AddWeight(BQK, 'B', 'C', weight=-2)

andgate.AddBias('C', [0,3]) #other two Spins need no biases

andgate.CompileKernels()

#run the model at a constant temperature to capture statistics:
temp = PottsPlayground.Schedules.constTemp(niters=1e6, temp=1)
results = PottsPlayground.Anneal(andgate, temp, nReports=int(1e4))

samples = results["AllStates"]
sample_indices = (samples[:,0, andgate.IndexOf('A')]
				+2*samples[:,0,andgate.IndexOf('B')]
				+4*samples[:,0,andgate.IndexOf('C')])
unique, counts = numpy.unique(sample_indices, return_counts=True)

for i, c in zip(unique, counts):
	pr = c/sample_indices.shape[0]
	print("Pr(A=%i, B=%i, C=%i) = %.3f"%(i%2, (i/2)%2, (i/4)%2, pr))
