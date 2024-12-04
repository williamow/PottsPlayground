import numpy
import pickle
import time
from matplotlib import pyplot as plt

import Annealing #my C++ and CUDA accelerated algorithms
from Tasks import GraphColoringTask as GC
from Tasks import TravelingSalesmanTask as TSP


def ArgsToFile(*args, **kwargs):
	with open("ArgsFile.pkl", 'wb') as f:
		pickle.dump({"args": args, "kwargs": kwargs}, f)


def check_energies(task, results, name):
	m = results['MinStates']
	e = results['MinEnergies']
	# print(e.shape)
	nchecks = e.shape[0]
	nMismatches = 0
	for i in range(nchecks):
		# print(e[i])
		# print(m[i,0,:])
		eval_cost = task.EvalCost(m[i,:])
		eps = 0.001 #epsilon value for float innacuracy
		if (e[i] - eps > eval_cost or e[i] + eps < eval_cost):
			print("Miscalculation at check %i: difference of %.5f"%(i, e[i]-eval_cost))
			nMismatches = nMismatches + 1;
	if (nMismatches > 0):
		print("Method %s had %i energy miscalculations"%(name, nMismatches))


T = 0.3
niters = int(1e4)
nReplicates = 32


task = GC.GraphColoring(nnodes=300, p=0.5, ncolors=30)
task = TSP.TravelingSalesman(25, 0.3)

PwlTemp = numpy.zeros([2, 2], dtype="float32")
PwlTemp[0,0] = 0.1
PwlTemp[0,1] = 0.001
PwlTemp[1,0] = 0
PwlTemp[1,1] = niters

task.e_th = 0.001
task.e_th = -1e14

# task.biases = task.biases + 1

# methods = [32, 64, 128]
methods = ["GPU", "CPU"]
# methods = [1, 2]
methods = ["Tsp", "Tsp", "TspLimited"]
methods = ["Tsp", "Tsp", "PottsJit"]


ArgsToFile(vars(task), PwlTemp, algorithm=0, backend="PottsPrecompute", substrate="CPU", nReplicates=nReplicates, nWorkers=1, nReports=10)

plt.figure()
for method in methods:
	# nWorkers = method
	nWorkers = 32
	be = method
	tstart = time.perf_counter()
	results = Annealing.Anneal(vars(task), PwlTemp, algorithm=2, backend=be, substrate="GPU", nReplicates=nReplicates, nWorkers=nWorkers, nReports=5)
	fps = niters/(time.perf_counter() - tstart)*1e-6
	final_best_soln = results['MinStates'][-1,:]
	
	# print(results['MinStates'])
	plt.plot(results['Iter'], results['AvgMinEnergies'], label="%s"%method)
	check_energies(task, results, method)
	print("Method %s. Solution cost: %.2f, Per instance FPS: %.6f MHz, Overall FPS: %.6f MHz"%(method, task.EvalCost(final_best_soln), fps, fps*nReplicates))

plt.show()

task.DisplayState(final_best_soln)

