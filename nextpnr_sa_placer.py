#script to be run as embedded python in nextpnr.
#saves the nextpnr "context" ctx to pickle for external analysis and use

# d = dict(locals())
# for key in d:
	# print(key, d[key])

# import pickle
# import networkx as nx
# import sys

from Tasks import Ice40PlacerTask
import numpy
import Annealing
from matplotlib import pyplot as plt
import time

# print(ctx.timing_result)
# for thing in dir(ctx.timing_result):
# 	print(thing)
# exit()
# def check_energies(task, results, name):
# 	m = results['MinStates']
# 	e = results['MinEnergies']
# 	# print(e.shape)
# 	nchecks = e.shape[0]
# 	nMismatches = 0
# 	for i in range(nchecks):
# 		# print(e[i])
# 		# print(m[i,0,:])
# 		eval_cost = task.EvalCost(m[i,:])
# 		eps = 0.001 #epsilon value for float innacuracy
# 		if (e[i] - eps > eval_cost or e[i] + eps < eval_cost):
# 			print("Miscalculation at check %i: difference of %.5f"%(i, e[i]-eval_cost))
# 			nMismatches = nMismatches + 1;
# 	if (nMismatches > 0):
# 		print("Method %s had %i energy miscalculations"%(name, nMismatches))


# for thing in dir(ctx):
	# print(thing)
# print(ctx.archId())
# for thing in dir(ctx.)
# placer = Ice40PlacerTask.Ice40PlacerTask(ctx, exclusion_factor=30)
# placer.MakeSparseKmap()
# exit()
# placer =
# placer.DisplayKernels()
# placer.MakeWeights()
# placer.weights = placer.kmap
# 
# placer.DisplayWeights()





plt.figure()
for excl in [15]:
	placer = Ice40PlacerTask.Ice40PlacerTask(ctx, exclusion_factor=excl)
		# exit()
	for temp in [9]:


		PwlTemp = numpy.zeros([2, 3], dtype="float32")
		PwlTemp[0,0] = temp
		PwlTemp[0,1] = temp-3
		PwlTemp[0,2] = 0.2
		PwlTemp[1,0] = 0
		PwlTemp[1,1] = 8e7
		PwlTemp[1,2] = 1e8

		
		placer.defaultPwlSchedule(niters=1e7, tmax=temp)
		placer.e_th = -1e14
		tstart = time.perf_counter()
		results = Annealing.Anneal(vars(placer), PwlTemp, OptsPerThrd=100, TakeAllOptions=True, backend="PottsJit", substrate="GPU", nReplicates=16, nWorkers=64, nReports=20)
		ttot = time.perf_counter() - tstart
		print("Annealing time is %.2f seconds"%ttot)
		final_best_soln = results['MinStates'][-1,:]

# check_energies(placer, results, "NA")

# plt.figure()
		plt.plot(results['Iter'], results['AvgMinEnergies'], label="Ex: %i, T=%i"%(excl, temp))


plt.legend()
plt.show()

placer.SetResultInContext(ctx, final_best_soln, STRENGTH_FIXED)
