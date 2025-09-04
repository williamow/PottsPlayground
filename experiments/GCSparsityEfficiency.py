import PottsPlayground
import numpy
from matplotlib import pyplot as plt
import pickle
import dwave_networkx as dnx
import os
import re
import time
import networkx as nx
import random


def generate_colored_graph(nnodes, ncolors, density):
    G = nx.Graph()
    nodes = list(range(nnodes))
    # random.shuffle(nodes)
    
    # Assign nodes to color groups
    # color_groups = {i: [] for i in range(ncolors)}
    for i, node in enumerate(nodes):
        # color_groups[i % ncolors].append(node)
        G.add_node(node, color=i % ncolors)
    
    max_edges = int(density * (nnodes ** 2)/2)
    added_edges = 0
    
    while added_edges < max_edges:
        node1, node2 = random.sample(nodes, 2)
        
        # Ensure the edge does not connect nodes of the same group
        if G.nodes[node1]['color'] != G.nodes[node2]['color'] and not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
            added_edges += 1
    
    return G

def TimedPPAnneal(task, temp, IC, nOptions, nActions, model, device, nReplicates, nWorkers):
	#wraps the PP anneal function and does one extra run with a single replicate in order to provide both single-run time and statastical averaging
	main_results = PottsPlayground.Anneal(task, temp, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=nReplicates, nWorkers=nWorkers)
	start = time.time()
	time_results = PottsPlayground.Anneal(task, temp, nOptions=nOptions, 
		nActions=nActions, model=model, device=device, nReplicates=1, nWorkers=nWorkers)
	t = time.time()-start
	main_results["Time"] = main_results["Iter"]/temp[-1,-1]*t
	return main_results

def calc_tts99(data, emin):
	successes = (data["AllMinEnergies"] < (emin + 1e-3)) #add a little epsilon for float comparison

	pr_success = numpy.mean(successes, axis=1)
	# print(pr_success)
	#set zero and 1 to nan since they are nonsensical to tts99 formula
	pr_success[pr_success == 1] = numpy.nan
	if numpy.all(numpy.isnan(pr_success)):
		return 0
	pr_success[pr_success == 0] = numpy.nan
	# print(pr_success == numpy.nan)
	if numpy.all(numpy.isnan(pr_success)):
		return 1e20 #arbitrary really large number

	tts99 = data['Time']*numpy.log(0.01)/numpy.log(1-pr_success)
	# print(tts99)
	opt_indx = numpy.nanargmin(tts99)

	return tts99[opt_indx]
	# fps.append(data["Iter"][-1]/data['Time'][-1]) #use Iter or RealFlips, either way the tts99 is the same
	# wrkPerFlip.append(1./tts99[opt_indx]/fps[-1])
	# names.append(series)


def GcSparsity():
	# temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=40, niters=1e6)
	results = {}
	ncolors = 3
	nReplicates = 1024
	nnodes = 150
	IC = numpy.random.randint(low=0, high=ncolors, size=[nnodes], dtype="int32")

	for density in [0.05, 0.1, 0.2, 0.5]:
		# tts99 = -1
		# counter = 0
		# density = 0.1
		# G = generate_colored_graph(nnodes=nnodes, ncolors=ncolors, density=density)
		# task = PottsPlayground.Tasks.GraphColoring(ncolors, G=G)
		# temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=density*nnodes*0.3, MinTemp=0.05, nTeeth=20, niters=1e6)
		task = PottsPlayground.Tasks.GraphColoring(2, p=density, nnodes=nnodes) #essentially, MaxCut
		temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=2, MinTemp=0.01, nTeeth=20, niters=1e6)
	
		# while tts99 < 0.1 or tts99 > 1:
		# 	counter = counter + 1
		# 	if tts99 < 0.1:
		# 		density = density * 1.05
		# 	else:
		# 		density = density * 0.97
		# 	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=2, MinTemp=0.01, nTeeth=20, niters=1e6)
		# 	# IC = numpy.random.randint(low=0, high=ncolors, size=[nReplicates, nnodes], dtype="int32")
		# 	# print(IC.shape)
		# 	# numpy.zeros([nnodes], dtype="int32")
		# 	print("Searching for problem of suitable difficulty, iter=%i                     "%counter, end='\r')
		# 	# G = generate_colored_graph(nnodes=nnodes, ncolors=ncolors, density=density)
		# 	task = PottsPlayground.Tasks.GraphColoring(ncolors, p=density, nnodes=nnodes)
		# 	task = PottsPlayground.Tasks.Binarized(task, ConstraintFactor=2)
		# 	r = TimedPPAnneal(task, temp, IC=IC, nOptions=1, 
		# 			nActions=1, model="PottsPrecompute", device="CPU", nReplicates=nReplicates, nWorkers=-1)
		# 	tts99 = calc_tts99(r, 0)
		# 	print(density, tts99, end=' ')

		# print("For density=%.2f, ncolors=%i                                                       "%(density, ncolors))

		for nActions in [1,2,5,10]:
			test_name = "density:%.2f-Actions:%i"%(density, nActions)
			print("running test", test_name, end='\r')
			results[test_name] = TimedPPAnneal(task, temp, IC=IC, nOptions=nnodes*ncolors*100, 
					nActions=nActions, model="PottsPrecompute", device="CPU", nReplicates=nReplicates, nWorkers=-1)
			best_soln = results[test_name]["MinEnergies"][-1]
			tts99 = calc_tts99(results[test_name], best_soln)
			print("Test", test_name, "soln is %.2f, tts99 is %.3f seconds"%(best_soln, tts99))

			# task.DisplayState(results[test_name]["MinStates"][-1,:])

		with open("Results/GcSparsity.pkl", 'wb') as f:
			pickle.dump(results, f)	

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
	Emin = 0
	# for series in results:
	# 	this_Emin = numpy.nanmin(results[series]["MinEnergies"])
	# 	Emin = numpy.min([Emin, this_Emin])
	# print(Emin)

	#process each "run" in results into an "fps" value and a work-per-flip value,
	#so that each "run" can be plotted as a single point and they can more easily be compared to each other on a single graph
	names = []
	wrkPerFlip = []
	fps = []
	for series in results:
		print(series)

		data = results[series]

		successes = (data["AllMinEnergies"] < (Emin + 1e-3)) #add a little epsilon for float comparison

		pr_success = numpy.mean(successes, axis=1)
		print(pr_success)
		#set zero and 1 to nan since they are nonsensical to tts99 formula
		pr_success[pr_success == 0] = numpy.nan
		pr_success[pr_success == 1] = numpy.nan
		tts99 = data['Time']*numpy.log(0.01)/numpy.log(1-pr_success)
		print(tts99)
		opt_indx = numpy.nanargmin(tts99)

		fps.append(data["Iter"][-1]/data['Time'][-1]) #use Iter or RealFlips, either way the tts99 is the same
		wrkPerFlip.append(1./tts99[opt_indx]/fps[-1])
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

GcSparsity()
# PlotPerformace("Results/GcSparsity.pkl")
# TspModels(nReplicates=1024, nCities=9)
# PlotPerformace("Results/TspModels.pkl")
# CheckCpuSolutions() #double-checks that the different model forms are correct and that their ground states actually correspond to the problem solution

# TspParallelism(nReplicates=1024, nCities=9)
# PlotPerformace("Results/TspParallelism.pkl")