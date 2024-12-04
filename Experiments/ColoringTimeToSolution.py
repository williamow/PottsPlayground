import numpy
import time
from matplotlib import pyplot as plt

import AllSolvers #my C++ and CUDA accelerated algorithms
from Tasks import GraphColoringTask as GC


cpu_iters = int(5e6)
gpu_iters = int(1e5)
ntrials = 100
nColors = 5
nNodes = 200
nReplicates = 2048
p = 11.5/nNodes
T = 0.25

CpuTimes = []
CpuStreamlinedTimes = []
GpuFwTimes1 = []
GpuKwTimes1 = []
GpuFwTimes32 = []
GpuKwTimes32 = []
GpuFwTimes1024 = []

GpuKwTimes1 = []
GpuKwTimes2 = []
GpuKwTimes4 = []
GpuKwTimes8 = []
GpuKwTimes16 = []
GpuKwTimes32 = []
GpuKwTimes64 = []
GpuKwTimes128 = []
GpuKwTimes256 = []
GpuKwTimes512 = []
GpuKwTimes1024 = []
GpuKwTimesAnnealing = []

tasks = [GC.GraphColoring(nnodes=nNodes, p=p, ncolors=nColors) for i in range(ntrials)]

#run this once, first, to skip over GPU init latency
# AllSolvers.PottsGpuKwSolve(tasks[0].kernels, tasks[0].kmap, tasks[0].qSizes, 1, 1, 0.7, 0.1, int(1e4), 0.5)

for i, task in enumerate(tasks):
	print("Solving task %i\r"%i, end="")

	tstart = time.perf_counter()
	soln = AllSolvers.ColoringStreamlinedPotts(nColors, cpu_iters, T, task.ConnectivityList())
	CpuStreamlinedTimes.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsCpuStSolve(task.weights, task.Partitions, 1, cpu_iters, T, 0.5)
	# CpuTimes.append(time.perf_counter() - tstart)


	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuFwSolve(task.weights, task.Partitions, 1, gpu_iters, T, 0.5, 32)
	# GpuFwTimes1.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuFwSolve(task.weights, task.Partitions, 32, gpu_iters, T, 0.5, 32)
	# GpuFwTimes32.append(time.perf_counter() - tstart)

	
	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuFwSolve(task.weights, task.Partitions, 1024, gpu_iters, T, 0.5, 32)
	# GpuFwTimes1024.append(time.perf_counter() - tstart)




	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 1, gpu_iters, T, 0.5)
	# GpuKwTimes1.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 2, gpu_iters, T, 0.5)
	# GpuKwTimes2.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 4, gpu_iters, T, 0.5)
	# GpuKwTimes4.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 8, gpu_iters, T, 0.5)
	# GpuKwTimes8.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 16, gpu_iters, T, 0.5)
	# GpuKwTimes16.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 32, gpu_iters, T, 0.5)
	# GpuKwTimes32.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 64, gpu_iters, T, 0.5)
	# GpuKwTimes64.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 128, gpu_iters, T, 0.5)
	# GpuKwTimes128.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 256, gpu_iters, T, 0.5)
	# GpuKwTimes256.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 512, gpu_iters, T, 0.5)
	# GpuKwTimes512.append(time.perf_counter() - tstart)

	# tstart = time.perf_counter()
	# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 1024, gpu_iters, T, T, 1, 0.5)
	# GpuKwTimes1024.append(time.perf_counter() - tstart)

	tstart = time.perf_counter()
	soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, 1024, gpu_iters, 5, T+0.1, T-0.1, gpu_iters, 0.5)
	GpuKwTimesAnnealing.append(time.perf_counter() - tstart)



CpuTimes.sort()
CpuStreamlinedTimes.sort()
GpuFwTimes1.sort()
GpuKwTimes1.sort()
GpuFwTimes32.sort()
GpuKwTimes32.sort()
GpuFwTimes1024.sort()

GpuKwTimes1.sort()
GpuKwTimes2.sort()
GpuKwTimes4.sort()
GpuKwTimes8.sort()
GpuKwTimes16.sort()
GpuKwTimes32.sort()
GpuKwTimes64.sort()
GpuKwTimes128.sort()
GpuKwTimes256.sort()
GpuKwTimes512.sort()
GpuKwTimes1024.sort()

GpuKwTimesAnnealing.sort()

cum_prob = numpy.linspace(0,1,ntrials)

plt.figure(figsize=(4., 2.5))

# plt.plot(CpuTimes, cum_prob, label='CPU')
plt.plot(CpuStreamlinedTimes, cum_prob, label='CPU Streamlined')
# plt.plot(GpuFwTimes1, cum_prob, label='GPU Weights 1')
# plt.plot(GpuFwTimes32, cum_prob, label='GPU Weights 32')
# plt.plot(GpuFwTimes1024, cum_prob, label='GPU Weights 1024')

# plt.plot(GpuKwTimes1, cum_prob, label='GPU Kernels 1')
# plt.plot(GpuKwTimes2, cum_prob, label='GPU Kernels 2')
# plt.plot(GpuKwTimes4, cum_prob, label='GPU Kernels 4')
# plt.plot(GpuKwTimes8, cum_prob, label='GPU Kernels 8')
# plt.plot(GpuKwTimes16, cum_prob, label='GPU Kernels 16')
# plt.plot(GpuKwTimes32, cum_prob, label='GPU Kernels 32')
# plt.plot(GpuKwTimes64, cum_prob, label='GPU Kernels 64')
# plt.plot(GpuKwTimes128, cum_prob, label='GPU Kernels 128')
# plt.plot(GpuKwTimes256, cum_prob, label='GPU Kernels 256')
# plt.plot(GpuKwTimes512, cum_prob, label='GPU Kernels 512')
# plt.plot(GpuKwTimes1024, cum_prob, label='GPU Kernels 1024')
plt.plot(GpuKwTimesAnnealing, cum_prob, label='GPU Kernels Annealing')

plt.xscale('log')
plt.xlabel("Time, s")
plt.ylabel("Percent solved")

plt.legend()

plt.show()