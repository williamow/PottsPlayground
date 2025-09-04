import numpy
import PottsPlayground
from PottsPlayground.Kernels import BinaryQuadratic as BQK

print("Testing: PottsPlayground.Tasks.BaseTask")

#============================basic spin-index model assembly
#manually create an Ising model of a logical AND gate, C = A&B
andgate = PottsPlayground.Tasks.BaseTask()

#three variables, each which take on values of 0 or 1
andgate.SetPartitions([2,2,2])

#connect A to B
andgate.AddKernel(BQK, i=0, j=1, weight=1)
andgate.AddKernel(BQK, i=1, j=0, weight=1)

#connect A to C
andgate.AddKernel(BQK, i=0, j=2, weight=-2)
andgate.AddKernel(BQK, i=2, j=0, weight=-2)

#connect B to C
andgate.AddKernel(BQK, i=2, j=1, weight=-2)
andgate.AddKernel(BQK, i=1, j=2, weight=-2)

andgate.SetBias(0, [0,0])
andgate.SetBias(1, [0,0])
andgate.SetBias(2, [0,3])

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







#============================compact text-ID based model assembly
#manually create an Ising model of a logical AND gate, C = A&B
andgate = PottsPlayground.Tasks.BaseTask()

#three variables, each which take on values of 0 or 1
andgate.AddSpins([2,2], ['A', 'B'])

andgate.AddWeight(BQK, 'A', 'B', weight=1)
andgate.AddSpins([2], ['C']) #try adding a node after already adding a weight
andgate.AddWeight(BQK, 'A', 'C', weight=-2)
andgate.AddWeight(BQK, 'B', 'C', weight=-2)

andgate.AddBias('C', [0,3]) #other two Spins need no biases

andgate.CompileKernels()

#check results:
for A in [0,1]:
	for B in [0,1]:
		for C in [0,1]:
			state = [A, B, C]
			state = {'A': A, 'B': B, 'C': C}
			E = andgate.EvalCost(state)
			if (C == (A*B) and E != 0):
				print("FAILED - Energy of valid AND gate is not zero")
			if (A+B == 1 and C == 1 and E != 1):
				print("FAILED - Energy of invalid AND gate should be 1")
			if (A+B == 2 and C == 0 and E != 1):
				print("FAILED - Energy of invalid AND gate should be 1")
			if (A+B == 0 and C == 1 and E != 3):
				print("FAILED - Energy of invalid AND gate should be 3")