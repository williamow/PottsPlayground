#run tests on the Ice40PlacerTask.
#this is only for testing the Potts model formulation itself,
#not for evaluating temperature schedules, annealing method, etc.

#this code relies on forking the process, so that the ctx (which lives in C land) can be duplicated.
#With duplication, we can run the place and route over and over to measure more data, without having to re-run the earlier steps.

# for thing in dir(ctx):
	# print(thing)
# d = dict(locals())
# for key in d:
	# print(key, d[key])
# for i in range(10):
	# print(ctx.getNetinfoRouteDelay(i))
# exit()

from Tasks import Ice40PlacerTask
import time
import os
import Annealing
import re
import numpy
from matplotlib import pyplot as plt
import pickle

def extract_float(s, pattern):
	match = re.search(pattern, s)
	if match:
		return float(match.group(1))
	return None

def cost_arch_scan():
	arches = []
	for i in [0, 0.25, 0.5, 1]:
		for j in [0, 0.25, 0.5, 1]:
			if (i+j) > 1:
				continue
			arches.append(   (15, i, j, (1-i-j)*0.1)   )
	return arches

def gather_metrics():
	cost_architectures = [
		# (15, 1, 0, 0),
		# (15, 0, 1, 0),
		# (15, 0, 0, 0.1),
		(15, 0., 1, 0.0)
	]
	# cost_architectures = cost_arch_scan()
	nArch = len(cost_architectures)
	print("Testing %i cost architectures."%nArch)
	arch_results = {}

	for arch in cost_architectures:
		placer = Ice40PlacerTask.Ice40PlacerTask(ctx, arch)

		potts_cost = []
		timing_cost = []
		fmax = []
		wirelen = []
		route_time = []

		def parent(pid, r):
			#parent process waits for child to finish and collects data from the child
			try:
				os.waitpid(pid, 0)
			except OSError:
				pass
			child_output = os.read(r, 1000000)
			child_output = child_output.decode('utf-8')
			if extract_float(child_output, pattern = r'nConflicts: (\d+)') > 0:
				print("ignoring data, placement had conflicts")
				return

			fmax.append(extract_float(child_output, pattern = r'Achieved FMAX: (\d+\.\d+)'))
			route_time.append(extract_float(child_output, pattern = r'Routing time is (\d+\.\d+)'))

			timing_cost.append(extract_float(child_output, pattern = r'timing cost = (\d+)'))
			wirelen.append(extract_float(child_output, pattern = r'wirelen = (\d+)'))

			potts_cost.append(extract_float(child_output, pattern = r'Potts cost: (\d+\.\d+)'))

			print(fmax[-1], potts_cost[-1], route_time[-1], timing_cost[-1], wirelen[-1])


		def child(final_best_soln, w):
			os.dup2(w, 1) #redirect child stderr and stdout to the pipe
			os.dup2(w, 2)

			#child process finishes the nextpnr run to extract performance metrics
			nConflicts = placer.SetResultInContext(ctx, final_best_soln)
			print("nConflicts:", nConflicts)
			if (nConflicts > 0):
				exit()

			#run built in place and route here to get some extra metric information
			ctx.place()

			tstart = time.perf_counter()
			ctx.route()
			ttot = time.perf_counter() - tstart
			print("Routing time is %.2f seconds"%ttot)

			for key in ctx.timing_result.clock_fmax:
				print("Achieved FMAX: %.2f"%ctx.timing_result.clock_fmax[key.first].achieved)

			print("Potts cost: %.2f"%placer.EvalCost(state))
			
			exit()

		nReplicates = 2
		for niters in [1e5, 3e5, 1e6, 3e6, 1e7]:
			tstart = time.perf_counter()
			results = Annealing.Anneal(vars(placer), placer.defaultTemp(niters, tmax=12), OptsPerThrd=10, TakeAllOptions=True, backend="PottsJit", substrate="CPU", nReplicates=nReplicates, nWorkers=1, nReports=1)
			ttot = time.perf_counter() - tstart
			print("Annealing time for %i iterations is %.2f seconds"%(niters, ttot))

			#nReplicates children are forked off all at once before the parent waits for all the children to finish
			r = [-1]*nReplicates
			w = [-1]*nReplicates
			pid = [-1]*nReplicates
			for i in range(nReplicates):			
				state = results["AllStates"][-1,i,:]
				r[i], w[i] = os.pipe()
				pid[i] = os.fork()
				if pid[i] == 0:
					child(state, w[i])
				
			for i in range(nReplicates):
				parent(pid[i], r[i])
				os.close(r[i])
				os.close(w[i])

		arch_results[arch] = {
			"potts_cost": potts_cost,
			"timing_cost": timing_cost,
			"fmax": fmax,
			"wirelen": wirelen,
			"route_time": route_time
		}

		#save the results after each new architecture has been tested:
		with open("Results/PlacerCostArch/PlacerCostArchData.pkl", 'wb') as f:
			pickle.dump(arch_results, f)


def plot_grid():
	from scipy import stats

	with open("Results/PlacerCostArch/PlacerCostArchData.pkl", 'rb') as f:
		arch_results = pickle.load(f)

	for i, arch_spec in enumerate(arch_results):
		res = arch_results[arch_spec]
		potts_cost = res["potts_cost"]
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 9))

		basis = numpy.ones([len(potts_cost), 2])
		basis[:,0] = potts_cost

		for ax, name in zip([ax1, ax2, ax3, ax4], ["timing_cost", "fmax", "wirelen", "route_time"]):
			ax.scatter(potts_cost, res[name])

			prank = stats.spearmanr(potts_cost, res[name])
			# print(prank)
			#add a linear regression:
			mb = numpy.matmul(numpy.linalg.pinv(basis), res[name])
			y = numpy.matmul(basis, mb)
			ax.plot(potts_cost, y)
			#calcuate R^2:
			R2 = 1 - (numpy.var(y-res[name])/numpy.var(res[name]))
			ax.set_title(name + " R^2: %.2f, rank: %.2f"%(R2, prank.statistic))

		plt.suptitle(repr(arch_spec))
		plt.tight_layout()
		plt.savefig("Results/PlacerCostArch/arch-%s.png"%repr(arch_spec))
		plt.close()

if __name__ == '__main__':
	if "ctx" in locals():
		gather_metrics()
	else:
		plot_grid()
	exit()