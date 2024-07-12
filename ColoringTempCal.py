import numpy
import time
from matplotlib import pyplot as plt

import AllSolvers #my C++ and CUDA accelerated algorithms
from Tasks import GraphColoringTask as GC


gpu_iters = int(1e6)
nBreaks = 100
ntrials = 10
nColors = 5
nNodes = 200
nReplicates = 2048
p = 12.5/nNodes
T = 0.25


tasks = [GC.GraphColoring(nnodes=nNodes, p=p, ncolors=nColors) for i in range(ntrials)]
task = tasks[0]

# tstart = time.perf_counter()
plt.figure()


# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, nBreaks, 0.2, 0.2, int(1e6), -100)
# plt.plot(soln)

# soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, nBreaks, 0.05, 0.2, int(1e6), -100)
# plt.plot(soln)

soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, nBreaks, 0.1, 0.2, int(1e6), -100)
plt.plot(soln)

soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, nBreaks, 0.1, 0.3, int(1e6), -100)
plt.plot(soln)

soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, nBreaks, 0.1, 0.4, int(1e6), -100)
plt.plot(soln)

# times.append(time.perf_counter() - tstart)



plt.show()
exit()


plt.figure(figsize=(4., 2.5))
temps = [0.2, 0.25, 0.3]
cum_prob = numpy.linspace(0,1,ntrials)

for T in temps:
	times = []
	print("Testing temperature %.2f"%T)

	for i, task in enumerate(tasks):
		print("Solving task %i\r"%i, end="")

		tstart = time.perf_counter()
		soln = AllSolvers.PottsGpuKwSolve(task.kernels, task.kmap, task.qSizes, nReplicates, gpu_iters, T, 0.5)
		times.append(time.perf_counter() - tstart)

	times.sort()
	plt.plot(times, cum_prob, label='T=%.2f'%T)


plt.xscale('log')
plt.xlabel("Time, s")
plt.ylabel("Percent solved")

plt.legend()

plt.show()