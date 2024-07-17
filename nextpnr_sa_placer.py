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
import pickle

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





# plt.figure()
for excl in [15]:
	placer = Ice40PlacerTask.Ice40PlacerTask(ctx, exclusion_factor=excl)
		# exit()
	for temp in [5]:
		
		placer.defaultPwlSchedule(niters=1e5, tmax=10)
		placer.e_th = -1e14
		tstart = time.perf_counter()
		results = Annealing.Anneal(vars(placer), placer.PwlTemp, OptsPerThrd=10, TakeAllOptions=True, backend="PottsJit", substrate="CPU", nReplicates=4, nWorkers=1, nReports=1)
		# results = Annealing.Anneal(vars(placer), PwlTemp, OptsPerThrd=1, TakeAllOptions=False, backend="PottsJit", substrate="CPU", nReplicates=1, nWorkers=1, nReports=10)
		ttot = time.perf_counter() - tstart
		print("Annealing time is %.2f seconds"%ttot)

		with open("res.pkl", 'wb') as f:
			pickle.dump(results, f)
		# for i in range(16):
			# final_best_soln = results['AllMinStates'][-2, i, :]
			# cost = placer.EvalSparseCost(final_best_soln)
			# print(i, cost)
		final_best_soln = results['AllStates'][-1,0,:]
		# exit()
# check_energies(placer, results, "NA")

# plt.figure()
		# plt.plot(results['Iter'], results['AvgMinEnergies'], label="Ex: %i, T=%i"%(excl, temp))


# plt.legend()
# plt.show()

crit_path = [
"$auto$simplemap.cc:420:simplemap_dff$7449_DFFLC",
"$nextpnr_ICESTORM_LC_1",
"$auto$alumacc.cc:474:replace_alu$4204.slice[1].carry$CARRY",
"$auto$alumacc.cc:474:replace_alu$4204.slice[2].adder_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13526_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13522_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13516_LC",
"$auto$alumacc.cc:474:replace_alu$4213.slice[2].adder_LC",
"$auto$alumacc.cc:474:replace_alu$4213.slice[3].adder_LC",
"$auto$alumacc.cc:474:replace_alu$4213.slice[4].adder_LC",
"$auto$alumacc.cc:474:replace_alu$4213.slice[5].adder_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13688_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13687_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13686_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$13685_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$14012_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$14011_LC",
"$abc$13313$auto$blifparse.cc:492:parse_blif$14131_LC",
"ROM.rom.0.0.0_RAM"
]

# placer.examine_path(crit_path)

# for u, v, data in placer.G.edges("$nextpnr_ICESTORM_LC_1", data=True):
	# print(data)

placer.SetResultInContext(ctx, final_best_soln, STRENGTH_FIXED)
