import PottsPlayground
import numpy
from matplotlib import pyplot as plt
import pickle
import dwave_networkx as dnx
import os
import re
import networkx as nx
import time


def TimedPPAnneal(task, temp, **kwargs):
	#wraps the PP anneal function and does one extra run with a single replicate in order to provide both single-run time and statastical averaging
	main_results = PottsPlayground.Anneal(task, temp, **kwargs)
	kwargs["nReplicates"] = 1
	start = time.time()
	time_results = PottsPlayground.Anneal(task, temp, **kwargs)
	t = time.time()-start
	main_results["Time"] = main_results["Iter"]/temp[-1,-1]*t
	return main_results

def MaxCut(nReplicates, nnodes, p):
	results = {}
	task = PottsPlayground.Tasks.GraphColoring(2, p=p, nnodes=nnodes)

	IsingTask = PottsPlayground.Tasks.Binarized(task) #to convert to binary quadratic form
	zeph = dnx.zephyr_graph(10)
	ZephTask = PottsPlayground.Tasks.MinorEmbedded(IsingTask, zeph, ConstraintFactor=2)

	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e6)

	for nOpts in [1, 1000000]:
		for nActions in [100, 50, 20, 10, 5, 2, 1]:
			if nActions > nOpts:
				continue
			test_name = "Ising %i/%i"%(nOpts,nActions)
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(IsingTask, temp, IC=None, nOptions=nOpts, 
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-1)

		for nActions in [200, 100, 50, 20, 10, 5, 2, 1]:
			if nActions > nOpts:
				continue
			test_name = "Zephyr %i/%i"%(nOpts,nActions)
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(ZephTask, temp, IC=None, nOptions=nOpts, 
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	with open("Results/MaxCut.pkl", 'wb') as f:
		pickle.dump(results, f)

def TspAllInOne(nReplicates, nCities):

	results = {}
	# tasks = {}

	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
	zeph = dnx.zephyr_graph(10)
	ZephTask = PottsPlayground.Tasks.MinorEmbedded(IsingTask, zeph, ConstraintFactor=0.8)

	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e6)
	tempz = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=10**nCities)

	ICpotts = numpy.linspace(0,nCities-1,nCities, dtype='int32')
	ICising = IsingTask.PottsToIsingSemantics(ICpotts)
	ICzephyr = ZephTask.StateToMinor(ICising)


	# print("Gathering data for natural representation, n=%i"%nCities)
	# # tasks["Natural"] = PottsTask
	# results["Natural"] = TimedPPAnneal(PottsTask, temp, IC=ICpotts, nOptions=1, 
	# 	nActions=1, model="Tsp", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	
	for nOpts in [1000000]:
		for nActions in [5]:
			if nActions > nOpts:
				continue
			test_name = "Potts %i/%i"%(nOpts,nActions)
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(PottsTask, temp, IC=ICpotts, nOptions=nOpts, 
				nActions=nActions, model="PottsJit", device="CPU", nReplicates=nReplicates, nWorkers=-1)

		for nActions in [20]:
			if nActions > nOpts:
				continue
			test_name = "Ising %i/%i"%(nOpts,nActions)
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(IsingTask, temp, IC=ICising, nOptions=nOpts, 
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-1)

		for nActions in []:
			if nActions > nOpts:
				continue
			test_name = "Zephyr %i/%i"%(nOpts,nActions)
			print("running test", test_name)
			results[test_name] = TimedPPAnneal(ZephTask, tempz, IC=ICzephyr, nOptions=nOpts, 
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-1)



	with open("Results/TspAllInOne.pkl", 'wb') as f:
		pickle.dump(results, f)

def PEvsDE(nReplicates, nCities):
	results = {}

	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)

	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e5)

	ICpotts = numpy.linspace(0,nCities-1,nCities, dtype='int32')

	results["PE"] = TimedPPAnneal(PottsTask, temp, 
		model="PottsPrecomputePE", algo="BirdsEye", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	results["DE"] = TimedPPAnneal(PottsTask, temp, 
		model="PottsPrecompute", algo="BirdsEye", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	with open("Results/PEvsDE.pkl", 'wb') as f:
		pickle.dump(results, f)

def TspNhpp(nReplicates, nCities):
	results = {}

	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
	# zeph = dnx.zephyr_graph(10)
	# ZephTask = PottsPlayground.Tasks.MinorEmbedded(IsingTask, zeph, ConstraintFactor=0.8)

	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e6)
	# tempz = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=10**nCities)

	ICpotts = numpy.linspace(0,nCities-1,nCities, dtype='int32')
	ICising = IsingTask.PottsToIsingSemantics(ICpotts)
	# ICzephyr = ZephTask.StateToMinor(ICising)

	for nActions in [10000, 20000, 50000, 100000, 1e6]:
		test_name = "Nhpp %i"%(nActions)
		print("running test", test_name)
		results[test_name] = TimedPPAnneal(IsingTask, temp, IC=ICising, nOptions=10000, 
			nActions=int(nActions), model="Ising-NHPP", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	for nActions in [20, 10, 5, 2, 1]:
		test_name = "Ising %i"%(nActions)
		print("running test", test_name)
		results[test_name] = TimedPPAnneal(IsingTask, temp, IC=ICising, nOptions=10000, 
			nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-1)

	with open("Results/TspNhpp.pkl", 'wb') as f:
		pickle.dump(results, f)

def TspParallelism(nReplicates, nCities):
	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e6)

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
				nActions=nActions, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-2)

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

def TspModels(nReplicates, nCities):
	results = {}
	tasks = {}

	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=nCities, ConstraintFactor=1.5, seed=0)
	IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)
	zeph = dnx.zephyr_graph(10)
	ZephTask = PottsPlayground.Tasks.MinorEmbedded(IsingTask, zeph, ConstraintFactor=0.8)

	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=1e6)
	tempz = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=20, niters=10**nCities)

	IC = numpy.linspace(0,nCities-1,nCities, dtype='int32')
	print("Gathering data for natural representation, n=%i"%nCities)
	tasks["Natural n=%i"%nCities] = PottsTask
	results["Natural n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
		nActions=1, model="Tsp", device="CPU", nReplicates=nReplicates, nWorkers=-2)

	print("Gathering data for Potts representation, n=%i"%nCities)
	tasks["Potts n=%i"%nCities] = PottsTask
	results["Potts n=%i"%nCities] = TimedPPAnneal(PottsTask, temp, IC=IC, nOptions=1, 
		nActions=1, model="PottsJit", device="CPU", nReplicates=nReplicates, nWorkers=-2)

	print("Gathering data for Ising representation, n=%i"%nCities)
	IC = IsingTask.PottsToIsingSemantics(IC)
	tasks["Ising n=%i"%nCities] = IsingTask
	results["Ising n=%i"%nCities] = TimedPPAnneal(IsingTask, temp, IC=IC, nOptions=1, 
		nActions=1, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-2)

	print("Gathering data for Zephyr representation, n=%i"%nCities)
	tasks["Zephyr n=%i"%nCities] = ZephTask
	ICzeph = ZephTask.StateToMinor(IC)
	results["Zephyr n=%i"%nCities] = TimedPPAnneal(ZephTask, tempz, IC=ICzeph, nOptions=1, 
		nActions=1, model="Ising", device="CPU", nReplicates=nReplicates, nWorkers=-2)


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
		# if series == "0/0":
		# 	print(data)
		print(series, ": Real flips / possible flips %i/%i"%(data["RealFlips"][-1], data["Iter"][-1]))
		avg_energies = data["AllEnergies"]*data["DwellTimes"]
		avg_energies = numpy.sum(avg_energies, axis=1)
		avg_energies = avg_energies/numpy.sum(data["DwellTimes"], axis=1)

		avg_energies = data["AvgMinEnergies"]
		plt.plot(data['Iter'], avg_energies, label=series)
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
	for series in results:
		this_Emin = numpy.nanmin(results[series]["MinEnergies"])
		Emin = numpy.min([Emin, this_Emin])
		print(this_Emin)
	print(Emin)

	#process each "run" in results into an "fps" value and a work-per-flip value,
	#so that each "run" can be plotted as a single point and they can more easily be compared to each other on a single graph
	names = []
	wrkPerFlip = []
	fps = []
	for series in results:
		print(series)

		data = results[series]

		successes = (data["AllMinEnergies"] < (Emin + 1e-2)) #add a little epsilon for float comparison

		pr_success = numpy.mean(successes, axis=1)
		# print(pr_success)
		if pr_success[0] == 1 or pr_success[-1] == 0:
			print("This series was either too fast or too slow for analysis")
			continue
		#set zero and 1 to nan since they are nonsensical to tts99 formula
		pr_success[pr_success == 0] = numpy.nan
		pr_success[pr_success == 1] = numpy.nan
		tts99 = data['Time']*numpy.log(0.01)/numpy.log(1-pr_success)
		# print(tts99)
		opt_indx = numpy.nanargmin(tts99)

		fps.append(data["RealFlips"][-1]/data['Time'][-1]) #use Iter or RealFlips, either way the tts99 is the same
		wrkPerFlip.append(1./tts99[opt_indx]/fps[-1])
		names.append(series)
		print(wrkPerFlip)
		print(fps)

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

# MaxCut(1024, 200, 0.02)
# PlotPerformace("Results/MaxCut.pkl")

# TspNhpp(nReplicates=256, nCities=8)
# PlotPerformace("Results/TspNhpp.pkl")

PEvsDE(nReplicates=256, nCities=12)
PlotPerformace("Results/PEvsDE.pkl")

# TspAllInOne(nReplicates=1024, nCities=8)
# PlotPerformace("Results/TspAllInOne.pkl")

# TspModels(nReplicates=1024, nCities=9)
# PlotPerformace("Results/TspModels.pkl")
# CheckCpuSolutions() #double-checks that the different model forms are correct and that their ground states actually correspond to the problem solution

# TspParallelism(nReplicates=1024, nCities=9)
# PlotPerformace("Results/TspParallelism.pkl")