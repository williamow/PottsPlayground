import PottsPlayground
import numpy
from matplotlib import pyplot as plt
import pickle
import dwave_networkx as dnx
import os
import re
import networkx as nx


def TimedPPAnneal(task, temp, IC, nOptions, nActions, model, device, nReplicates, nWorkers):
	#wraps the PP anneal function and does one extra run with a single replicate in order to provide both single-run time and statastical averaging
	main_results = PottsPlayground.Anneal(task, temp, IC=IC, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=nReplicates, nWorkers=nWorkers)
	time_results = PottsPlayground.Anneal(task, temp, IC=IC, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=1, nWorkers=nWorkers)
	main_results["Time"] = time_results["Time"]
	return main_results

def TspParallelism():
	nReplicates = 64
	nCities = 9

	PottsTask = PottsPlayground.TravelingSalesmanTask(ncities=nCities, ConstraintFactor=1.5, seed=0)
	# PottsTask = PottsPlayground.GraphColoringTask(3, p=0.05, nnodes=100)
	IsingTask = PottsPlayground.BinarizeTask(PottsTask, ConstraintFactor=1.5)
	t=0.2
	temp = PottsTask.SawtoothTempLog2Space(MaxTemp=1., MinTemp=0.05, nTeeth=40, nIters=1e6)
	# temp = PottsTask.constTempLog2Space(nReports=100, temp=t, nIters=1e6)
	# temp2 = PottsTask.constTempLog2Space(nReports=100, temp=t*2, nIters=1e6)

	results = {}

	ICpotts = numpy.linspace(0,nCities-1,nCities, dtype='int32')
	ICising = numpy.eye(nCities, dtype='int32')
	# ICising = numpy.zeros([100,3], dtype='int32')
	# ICising[:,0] = 1
	ICising = ICising.flatten()

	for nActions in [27,9,3,1]:
		for nOpts in [1,9,81]:
			if nActions > nOpts:
				continue
			test_name = "%i/%i"%(nOpts,nActions)
			if test_name in results:
				continue
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(IsingTask, temp, IC=ICising, nOptions=nOpts, 
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=4)

	with open("Results/TspParallelism.pkl", 'wb') as f:
		pickle.dump(results, f)

def CheckCpuSolutions():
	#makes sure that the ising and zephyr models are actually finding real soluions.
	#The energy graphs support this as well, but it is good to check the semantics
	#to make sure that the problem representations are correct.
	with open("Results/TspModelsPoints.pkl", 'rb') as f:
		results = pickle.load(f)

	with open("Results/TspModelTasks.pkl", 'rb') as f:
		tasks = pickle.load(f)

	for result in results:
		best_solution = results[result]["MinStates"][-1,:]
		task = tasks[result]
		print(result, "valid semantics?", task.IsValidSemantics(best_solution))
		if "Zephyr" in result:
			best_solution = task.FuzzyMinorToState(best_solution)
			best_solution = task.task.IsingToPottsSemantics(best_solution)
			task = task.task.PottsTask
		if "Ising" in result:
			best_solution = task.IsingToPottsSemantics(best_solution)
			task = task.PottsTask

		#task should be an instance of potts tsp task now:
		task.DisplayState(best_solution)

def TspModels():
	nReplicates = 32
	results = {}
	tasks = {}

	for nCities in [6, 7, 8, 9]:

		PottsTask = PottsPlayground.TravelingSalesmanTask(ncities=nCities, ConstraintFactor=1.5, seed=0)
		IsingTask = PottsPlayground.BinarizeTask(PottsTask, ConstraintFactor=1.5)
		zeph = dnx.zephyr_graph(10)
		ZephTask = PottsPlayground.MinorEmbeddedTask(IsingTask, zeph, ConstraintFactor=0.8)		

		t=0.2
		temp = PottsTask.SawtoothTempLog2Space(MaxTemp=1., MinTemp=0.05, nTeeth=40, nIters=1e6)
		tempz = PottsTask.SawtoothTempLog2Space(MaxTemp=1., MinTemp=0.05, nTeeth=40, nIters=10**nCities)

		IC = numpy.linspace(0,nCities-1,nCities, dtype='int32')
		print("Gathering data for natural representation, n=%i"%nCities)
		tasks["Natural n=%i"%nCities] = PottsTask
		results["Natural n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
			nActions=1, model="Tsp", device="CPU", nReplicates=nReplicates, nWorkers=4)

		print("Gathering data for Potts representation, n=%i"%nCities)
		tasks["Potts n=%i"%nCities] = PottsTask
		results["Potts n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
			nActions=1, model="PottsJit", device="CPU", nReplicates=nReplicates, nWorkers=4)

		IC = numpy.eye(nCities, dtype='int32')
		IC = IC.flatten()
		print("Gathering data for Ising representation, n=%i"%nCities)
		tasks["Ising n=%i"%nCities] = IsingTask
		results["Ising n=%i"%nCities] = TimedPPAnneal(IsingTask, temp, IC=IC, nOptions=1, 
			nActions=1, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=4)

		print("Gathering data for Zephyr representation, n=%i"%nCities)
		tasks["Zephyr n=%i"%nCities] = ZephTask
		ICzeph = ZephTask.StateToMinor(IC)
		results["Zephyr n=%i"%nCities] = TimedPPAnneal(ZephTask, tempz, IC=ICzeph, nOptions=1, 
			nActions=1, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=4)


	with open("Results/TspModels.pkl", 'wb') as f:
		pickle.dump(results, f)

	with open("Results/TspModelTasks.pkl", 'wb') as f:
		pickle.dump(tasks, f)

def IsingModels():
	nReplicates = 64
	nCities = 12

	PottsTask = PottsPlayground.TravelingSalesmanTask(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.BinarizeTask(PottsTask, ConstraintFactor=1.5)

	t=0.2
	temp = PottsTask.constTempLog2Space(nReports=100, temp=t, nIters=1e5)
	temp2 = PottsTask.constTempLog2Space(nReports=100, temp=t*2, nIters=1e5)

	results = {}


	IC = numpy.eye(nCities, dtype='int32')
	IC = IC.flatten()
	print("Gathering data for Ising representation")
	results["Ising - PottsJit-CPU"] = TimedPPAnneal(IsingTask, temp, IC=IC, OptsPerThrd=1, 
		TakeAllOptions=False, model="PottsJit", device="CPU", nReplicates=nReplicates, nWorkers=4)

	results["Ising - PottsJit-GPU"] = TimedPPAnneal(IsingTask, temp, IC=IC, OptsPerThrd=5, 
		TakeAllOptions=True, model="PottsJit", device="GPU", nReplicates=nReplicates, nWorkers=5)

	results["Ising - Ising-CPU"] = TimedPPAnneal(IsingTask, temp, IC=IC, OptsPerThrd=1, 
		TakeAllOptions=False, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=4)

	results["Ising - Ising-GPU"] = TimedPPAnneal(IsingTask, temp, IC=IC, OptsPerThrd=5, 
		TakeAllOptions=True, model="Ising", device="GPU", nReplicates=nReplicates, nWorkers=5)


	with open("Results/Isingmodels.pkl", 'wb') as f:
		pickle.dump(results, f)

def TspZephyrPoint():
	nCities = 8

	PottsTask = PottsPlayground.TravelingSalesmanTask(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.BinarizeTask(PottsTask, ConstraintFactor=1.5)
	zeph = dnx.zephyr_graph(10)
	ZephTask = PottsPlayground.MinorEmbeddedTask(IsingTask, zeph, ConstraintFactor=0.8)

	# with open("Results/TspZephyrTask.pkl", 'wb') as f:
			# pickle.dump(ZephTask, f)

	t=0.2
	# temp = PottsTask.constTempLog2Space(nReports=100, temp=t, nIters=1e5)
	temp = PottsTask.SawtoothTempLog2Space(MaxTemp=0.5, MinTemp=0.05, nTeeth=40, nIters=1e9)
	# temp2 = PottsTask.constTempLog2Space(nReports=100, temp=t*2, nIters=1e5)

	results = {}


	IC = numpy.eye(nCities, dtype='int32')
	IC = IC.flatten()
	IC = ZephTask.StateToMinor(IC)


	niters = 10
	ratio = 1.5

	while (niters <1e8):
		print("nIters=%i"%niters)
		temp = PottsTask.LinearTemp(niters=niters, temp=0.5, t2=0.05)
		niters=int(ratio*niters)

		inc_results = PottsPlayground.Anneal(ZephTask, temp, IC=IC, OptsPerThrd=10,
			TakeAllOptions=True, model="Ising", device="GPU", nReplicates=32, nWorkers=64)

		if len(results) == 0:
			results["Zephyr"] = inc_results
		else:
			#need to merge the incrimental results with the existing results.

			#most things can just be appended on the first dimension:
			for key in results["Zephyr"]:
				results["Zephyr"][key] = numpy.append(results["Zephyr"][key], inc_results[key], axis=0)

			#time and iters needs a little additional handling:
			results["Zephyr"]["Iter"][-1] = results["Zephyr"]["Iter"][-1] + results["Zephyr"]["Iter"][-2]
			results["Zephyr"]["Time"][-1] = results["Zephyr"]["Time"][-1] + results["Zephyr"]["Time"][-2]

		IC = inc_results["AllMinStates"][0,:,:] #copy old best states to new states for next run

		#save progress at each iteration
		with open("Results/TspZephyrPoint.pkl", 'wb') as f:
			pickle.dump(results, f)

		print("AvgMinEnergy: %.2f"%results["Zephyr"]["AvgMinEnergies"][-1])

def TspZephyrVariants():
	nCities = 12

	PottsTask = PottsPlayground.TravelingSalesmanTask(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.BinarizeTask(PottsTask, ConstraintFactor=1.5)
	zeph = dnx.zephyr_graph(10)
	ZephTask = PottsPlayground.MinorEmbeddedTask(IsingTask, zeph, ConstraintFactor=0.8)

	IC = numpy.eye(nCities, dtype='int32')
	IC = IC.flatten()
	ICz = ZephTask.StateToMinor(IC)

	results = {}

	for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
		temp = PottsTask.SawtoothTempLog2Space(MaxTemp=t, MinTemp=0.05, nTeeth=40, nIters=1e7)
		print("T=%.2f"%t)
		results["Zephyr - Tmax=%.2f"%t] = PottsPlayground.Anneal(ZephTask, temp, IC=ICz, OptsPerThrd=10,
			TakeAllOptions=True, model="Ising", device="GPU", nReplicates=32, nWorkers=64)

	# temp = PottsTask.SawtoothTempLog2Space(MaxTemp=0.5, MinTemp=0.05, nTeeth=40, nIters=1e7)
	# for cf in [0.8, 0.9, 1.0]:
		
	# 	print("cf=%.2f"%cf)
	# 	ZephTask = PottsPlayground.MinorEmbeddedTask(IsingTask, zeph, ConstraintFactor=0.8)
	# 	ICz = ZephTask.StateToMinor(IC)
	# 	results["Zephyr - Cf=%.2f"%cf] = PottsPlayground.Anneal(ZephTask, temp, IC=ICz, OptsPerThrd=10,
	# 		TakeAllOptions=True, model="Ising", device="GPU", nReplicates=32, nWorkers=64)

	with open("Results/TspZephyrVariants.pkl", 'wb') as f:
		pickle.dump(results, f)

def PlotMixingCurves(fname):
	with open(fname, 'rb') as f:
		results = pickle.load(f)


	plt.figure()
	for series in results:
		data = results[series]
		print(series, ": Real flips / possible flips %i/%i"%(data["RealFlips"][-1], data["Iter"][-1]))
		avg_energies = data["AllEnergies"]*data["DwellTimes"]
		avg_energies = numpy.sum(avg_energies, axis=1)
		avg_energies = avg_energies/numpy.sum(data["DwellTimes"], axis=1)

		avg_energies = data["AvgMinEnergies"]
		plt.plot(data['RealFlips'], avg_energies, label=series)
	plt.xscale('log')
	plt.xlabel("Annealing Iterations")
	plt.ylabel("Average energy")
	plt.legend()
	plt.tight_layout()
	imname = fname[:-3] + "png"
	plt.savefig(imname)
	plt.show()

def PlotPerformace(fname):
	with open(fname, 'rb') as f:
		results = pickle.load(f)

	#determine completion threshold by scanning over all series for the lowest found energy:
	Emin = 1e9
	Emax = -1e9
	for series in results:
		this_Emin = numpy.nanmin(results[series]["AvgMinEnergies"])
		this_Emax = numpy.nanmax(results[series]["AvgMinEnergies"])
		Emin = numpy.min([Emin, this_Emin])
		Emax = numpy.max([Emax, this_Emax])
	E50 = (Emin+Emax)/2.

	#process each "run" in results into an "fps" value and a work-per-flip value,
	#so that each "run" can be plotted as a single point and they can more easily be compared to each other on a single graph
	names = []
	wrkPerFlip = []
	fps = []
	for series in results:
		data = results[series]
		# avg_energies = data["AllEnergies"]*data["DwellTimes"]
		# avg_energies = numpy.sum(avg_energies, axis=1)
		# avg_energies = avg_energies/numpy.sum(data["DwellTimes"], axis=1)
		avg_energies = data["AvgMinEnergies"]

		#find how many iterations it took
		#for the average energy to drop half of the way from the starting energy to the final/equilibrium energy:
		E50 = (numpy.nanmax(avg_energies)+numpy.nanmin(avg_energies))/2.
		# max_point = numpy.nanargmax(avg_energies)
		# mask = numpy.zeros(avg_energies.shape)
		# mask[max_point:] = 1
		indx = numpy.argmax(avg_energies < E50) #argmax should return the first index of value 1

		#do a linear interpolation:
		flip_def = "Iter"
		flip_def = "RealFlips"
		E1 = avg_energies[indx-1]
		flips1 = data[flip_def][indx-1]
		E2 = avg_energies[indx]
		flips2 = data[flip_def][indx]
		lmbda = (E50-E2)/(E1-E2)
		flips = lmbda*flips1 + (1-lmbda)*flips2

		wrkPerFlip.append(1./flips)
		fps.append(data[flip_def][-1]/data['Time'][-1])
		# print(series, data['nSerialExecutions'])
		names.append(series)

	xmin = numpy.min(wrkPerFlip)*0.5
	xmax = numpy.max(wrkPerFlip)*2
	ymin = numpy.min(fps)*0.5
	ymax = numpy.max(fps)*2

	plt.figure(figsize=(7,7))
	nLines = 10
	for i in range(nLines):
		#create iso-performance lines.
		#Plot will be log-log, and I want equal visual spacing, so let's figure out what the ratio should be:
		perfmin = xmin*ymin
		perfmax = xmax*ymax
		ratio = (perfmax/perfmin)**(1./nLines)
		thisPerf = perfmin*ratio**i
		xdata = numpy.linspace(xmin, xmax, 50)
		ydata = thisPerf/xdata
		plt.plot(xdata, ydata, c='lightgray')

	plt.scatter(wrkPerFlip, fps)

	plt.xlim([xmin, xmax])
	plt.ylim([ymin, ymax])
	plt.xscale('log')
	plt.yscale('log')
	plt.ylabel("Flips per second")
	plt.xlabel("Work per flip")

	# for series in results:
	# 	data = results[series]
	# 	avg_energies = data["AllEnergies"]*data["DwellTimes"]
	# 	avg_energies = numpy.sum(avg_energies, axis=1)
	# 	avg_energies = avg_energies/numpy.sum(data["DwellTimes"], axis=1)
	# 	plt.plot(data['Iter'], avg_energies, label=series)
	# plt.xscale('log')
	# plt.xlabel("Annealing Iterations")
	# plt.ylabel("Average energy")
	# plt.legend()
	plt.tight_layout()
	# imname = fname[:-3] + "png"
	# plt.savefig(imname)

	plt.savefig("Results/Work-Speed tradeoff.svg")
	#use inkscape to convert to emf for visio import:
	os.system('cat "Results/Work-Speed tradeoff.svg" | inkscape --pipe --export-filename="Results/Work-Speed tradeoff.emf"')
	os.remove("Results/Work-Speed tradeoff.svg")

	#annotate after saving, so that annotations can be added and arranged manually
	for i, txt in enumerate(names):
		plt.annotate(txt, (wrkPerFlip[i], fps[i]))

	plt.show()

# Isingmodels()
# PlotMixingCurves("Results/Isingmodels.pkl")
# PlotPerformace(["Results/Isingmodels.pkl"])

# TspZephyrVariants()
# PlotMixingCurves("Results/TspZephyrVariants.pkl")

# TspModels()
# PlotMixingCurves("Results/TspModels.pkl")
# PlotPerformace("Results/TspModels.pkl")
# CheckCpuSolutions() #double-checks that the different model forms are correct and that their ground states actually correspond to the problem solution

# TspParallelism()
# PlotMixingCurves("Results/TspParallelism.pkl")
PlotPerformace("Results/TspParallelism.pkl")