import numpy
import PottsPlayground
from PottsPlayground.Kernels import BinaryQuadratic as BQK

print("Testing: PottsPlayground.Tasks.BaseTask")

#manually create an Ising model of an logical AND gate, C = A&B
andgate = PottsPlayground.Tasks.BaseTask()

#three variables, each which take on values of 0 or 1
andgate.SetPartitions([2,2,2])

andgate.InitKernelManager()

#connect A to B
andgate.AddKernel(BQK, i=0, j=1, weight=1)
andgate.AddKernel(BQK, i=1, j=0, weight=1)

#connect A to C
andgate.AddKernel(BQK, i=0, j=2, weight=-2)
andgate.AddKernel(BQK, i=2, j=0, weight=-2)

#connect B to C
andgate.AddKernel(BQK, i=2, j=1, weight=-2)
andgate.AddKernel(BQK, i=1, j=2, weight=-2)

andgate.biases[0,:] = [0,0]
andgate.biases[1,:] = [0,0]
andgate.biases[2,:] = [0,3]

andgate.CompileKernels()

#check results:
for A in [0,1]:
	for B in [0,1]:
		for C in [0,1]:
			state = [A, B, C]
			E = andgate.EvalCost(state)
			if (C == (A*B) and E != 0):
				print("FAILED - Energy of valid AND gate is not zero")
			if (A+B == 1 and C == 1 and E != 1):
				print("FAILED - Energy of invalid AND gate should be 1")
			if (A+B == 2 and C == 0 and E != 1):
				print("FAILED - Energy of invalid AND gate should be 1")
			if (A+B == 0 and C == 1 and E != 3):
				print("FAILED - Energy of invalid AND gate should be 3")



#run the model at a constant temperature to capture statistics:
# temp = PottsPlayground.Schedules.constTemp(niters=1e6, temp=1)
# results = PottsPlayground.Anneal(andgate, temp, nReports=int(1e4))

# #plot the distribution of states:
# samples = results["AllStates"]
# sample_indices = samples[:,0,0]+2*samples[:,0,1]+4*samples[:,0,2]
# unique, counts = numpy.unique(sample_indices, return_counts=True)

# for i, c in zip(unique, counts):
# 	pr = c/sample_indices.shape[0]
# 	print("A=%i, B=%i, C=%i -> Pr=%.3f"%(i%2, (i/2)%2, (i/4)%2, pr))
