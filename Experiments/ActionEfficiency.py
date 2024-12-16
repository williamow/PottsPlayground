import PottsPlayground
import numpy
from matplotlib import pyplot as plt
import pickle
import dwave_networkx as dnx
import os
import re
import networkx as nx
import time


def TimedPPAnneal(task, temp, IC, nOptions, nActions, model, device, nReplicates, nWorkers):
	#wraps the PP anneal function and does one extra run with a single replicate in order to provide both single-run time and statastical averaging
	main_results = PottsPlayground.Anneal(task, temp, IC=IC, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=nReplicates, nWorkers=nWorkers)
	start = time.time()
	time_results = PottsPlayground.Anneal(task, temp, IC=IC, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=1, nWorkers=nWorkers)
	t = time.time()-start
	main_results["Time"] = main_results["Iter"]/temp[-1,-1]*t
	return main_results

def TspParallelism(nReplicates=64, nCities=9):

	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
	t=0.2
	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=40, niters=1e6)

	results = {}

	ICpotts = numpy.linspace(0,nCities-1,nCities, dtype='int32')
	ICising = IsingTask.PottsToIsingSemantics(ICpotts)

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
	with open("Results/TspModels.pkl", 'rb') as f:
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

def TspModels(nReplicates=32, sizes=[6,7,8,9]):
	nReplicates = 32
	results = {}
	tasks = {}

	for nCities in sizes:

		PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
		IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
		zeph = dnx.zephyr_graph(10)
		ZephTask = PottsPlayground.Tasks.MinorEmbedded(IsingTask, zeph, ConstraintFactor=0.8)		

		t=0.2
		temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=40, niters=1e6)
		tempz = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=40, niters=10**nCities)

		IC = numpy.linspace(0,nCities-1,nCities, dtype='int32')
		print("Gathering data for natural representation, n=%i"%nCities)
		tasks["Natural n=%i"%nCities] = PottsTask
		results["Natural n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
			nActions=1, model="Tsp", device="CPU", nReplicates=nReplicates, nWorkers=4)

		print("Gathering data for Potts representation, n=%i"%nCities)
		tasks["Potts n=%i"%nCities] = PottsTask
		results["Potts n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
			nActions=1, model="PottsJit", device="CPU", nReplicates=nReplicates, nWorkers=4)

		print("Gathering data for Ising representation, n=%i"%nCities)
		IC = IsingTask.PottsToIsingSemantics(IC)
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

# TspModels(nReplicates=2, sizes=[6])
TspModels()
PlotMixingCurves("Results/TspModels.pkl")
PlotPerformace("Results/TspModels.pkl")
CheckCpuSolutions() #double-checks that the different model forms are correct and that their ground states actually correspond to the problem solution

# TspParallelism(nReplicates=2, nCities=6)
TspParallelism()
PlotMixingCurves("Results/TspParallelism.pkl")
PlotPerformace("Results/TspParallelism.pkl")