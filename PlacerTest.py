#run tests on the Ice40PlacerTask.
#this is only for testing the Potts model formulation itself,
#not for evaluating temperature schedules, annealing method, etc.

#this code relies on forking the process, so that the ctx (which lives in C land) can be duplicated.
#With duplication, we can run the place and route over and over to measure more data, without having to re-run the earlier steps.


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
		(15, 0, 0, 0.1),
		(15, 0.5, 1, 0.05)
	]
	cost_architectures = cost_arch_scan()
	nArch = len(cost_architectures)
	print("Testing %i cost architectures."%nArch)
	arch_results = {}

	r, w = os.pipe()

	for arch in cost_architectures:
		placer = Ice40PlacerTask.Ice40PlacerTask(ctx, arch)

		potts_cost = []
		timing_cost = []
		fmax = []
		wirelen = []
		route_time = []

		def parent(pid, state):
			#parent process waits for child to finish and collects data from the child
			os.waitpid(pid, 0)
			child_output = os.read(r, 1000000)
			child_output = child_output.decode('utf-8')
			if extract_float(child_output, pattern = r'nConflicts: (\d+)') > 0:
				print("ignoring data, placement had conflicts")
				return

			fmax.append(extract_float(child_output, pattern = r'Achieved FMAX: (\d+\.\d+)'))
			route_time.append(extract_float(child_output, pattern = r'Routing time is (\d+\.\d+)'))

			timing_cost.append(extract_float(child_output, pattern = r'timing cost = (\d+)'))
			wirelen.append(extract_float(child_output, pattern = r'wirelen = (\d+)'))

			potts_cost.append(placer.EvalCost(state))
			print(fmax[-1], potts_cost[-1], route_time[-1], timing_cost[-1], wirelen[-1])


		def child(final_best_soln):
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
			
			exit()

		nReplicates = 2
		for niters in [1e5, 3e5]:#, 1e6, 3e6, 1e7]:
			tstart = time.perf_counter()
			results = Annealing.Anneal(vars(placer), placer.defaultTemp(niters, tmax=12), OptsPerThrd=10, TakeAllOptions=True, backend="PottsJit", substrate="CPU", nReplicates=nReplicates, nWorkers=1, nReports=1)
			ttot = time.perf_counter() - tstart
			print("Annealing time for %i iterations is %.2f seconds"%(niters, ttot))

			for i in range(nReplicates):
				for res_type in ["AllStates"]:#, "AllMinStates"]:
					state = results[res_type][-1,i,:]
					pid = os.fork()
					if pid > 0:
						parent(pid, state)
					else:
						child(state)

		arch_results[arch] = {
			"potts_cost": potts_cost,
			"timing_cost": timing_cost,
			"fmax": fmax,
			"wirelen": wirelen,
			"route_time": route_time
		}

	with open("results/PlacerCostArchData.pkl", 'wb') as f:
		pickle.dump(arch_results, f)


def plot_grid():
	#plot a grid of
	fig, ax_list = plt.subplots(nArch, 4, figsize=(6.5, 1.5*nArch))

	ax1.scatter(potts_cost, timing_cost)
	ax1.set_title("timing cost")
	ax2.scatter(potts_cost, fmax)
	ax2.set_title("Fmax")
	ax3.scatter(potts_cost, wirelen)
	ax3.set_title("wirelen cost")
	ax4.scatter(potts_cost, route_time)
	ax4.set_title("routing time")
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	gather_metrics()
	exit()