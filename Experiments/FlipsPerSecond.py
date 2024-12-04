import numpy
import time

import AllSolvers #my C++ and CUDA accelerated algorithms
import Annealing
from Tasks import GraphColoringTask as GC


#for comparing the FPS of different algorithms, and measuring how the FPS changes w.r.t problem size and other parameters


iters = int(1e5)

ntrials = 100

nReplicates = 1024

T = 0.25

nColors = 5
nNodes = 100
p = 20./nNodes #high enough that no solutions will be found
task = GC.GraphColoring(nnodes=nNodes, p=p, ncolors=nColors)

#run this once, first, to skip over GPU init latency
# AllSolvers.PottsGpuKwSolve(tasks[0].kernels, tasks[0].kmap, tasks[0].qSizes, 1, 1, 0.7, 0.1, int(1e4), 0.5)

# for i, task in enumerate(tasks):
	# print("Solving task %i\r"%i, end="")



# tstart = time.perf_counter()
# soln = AllSolvers.ColoringStreamlinedPotts(nColors, iters, T, task.ConnectivityList())
# CpuStreamlinedFps = iters/(time.perf_counter() - tstart)
# print("Cpu streamlined FPS: %.2fM"%(CpuStreamlinedFps*1e-6))
PwlTemp = numpy.zeros([2, 2], dtype="float32")
PwlTemp[0,:] = 0.3
PwlTemp[1,0] = 0
PwlTemp[1,1] = iters

task.e_th = -1

# tstart = time.perf_counter()
res_dict = Annealing.Anneal(task, PwlTemp, method='Cpu', nReplicates=1, nReports=2)
# res_dict = res_dict["MinEnergies"]
# print("here?")
print(res_dict["MinEnergies"])
# CpuFps = iters/(time.perf_counter() - tstart)
# print("Cpu FPS: %.2fM"%(CpuFps*1e-6))


tstart = time.perf_counter()
Annealing.Anneal(task, PwlTemp, method='Cpu', nReplicates=1, nReports=1)
CpuFps = iters/(time.perf_counter() - tstart)
print("Cpu FPS: %.2fM"%(CpuFps*1e-6))

# tstart = time.perf_counter()
# Annealing.Anneal(task, PwlTemp, method='Gpu', nReplicates=nReplicates, nReports=1)
# GpuKernelsFps = iters/(time.perf_counter() - tstart)
# # print("Gpu Kernels per thread FPS: %.2fM"%(GpuKernelsFps*1e-6))
# # print("Gpu Kernels overall FPS: %.2fM"%(nReplicates*GpuKernelsFps*1e-6))

tstart = time.perf_counter()
Annealing.Anneal(task, PwlTemp, method='Gpu', nReplicates=nReplicates, nReports=1)
GpuKernelsFps = iters/(time.perf_counter() - tstart)
print("Gpu Kernels per thread FPS: %.2fM"%(GpuKernelsFps*1e-6))
print("Gpu Kernels overall FPS: %.2fM"%(nReplicates*GpuKernelsFps*1e-6))

# tstart = time.perf_counter()
# soln = AllSolvers.PottsGpuFwSolve(task.weights, task.Partitions, nReplicates, iters, T, 0.5, 32)
# GpuWeightsFps = iters/(time.perf_counter() - tstart)
# print("Gpu Weights per thread FPS: %.2fM"%(GpuWeightsFps*1e-6))
# print("Gpu Weights overall FPS: %.2fM"%(nReplicates*GpuWeightsFps*1e-6))



