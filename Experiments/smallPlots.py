import PottsPlayground
# from PottsPlayground.Tasks.MinorEmbeddedTask import ChimeraGraph
import itertools
import numpy
# import subprocess
import os

import dwave_networkx as dnx

tsp = PottsPlayground.TravelingSalesmanTask(ncities=10, ConstraintFactor=1.5)
ising = PottsPlayground.BinarizeTask(tsp, ConstraintFactor=1.5)
zeph = dnx.zephyr_graph(6)
peg = dnx.pegasus_graph(30)
chi = dnx.chimera_graph(40)
chimera = PottsPlayground.MinorEmbeddedTask(ising, zeph, ConstraintFactor=1.5)

exit()

# task = GraphColoringTask.GraphColoring(ncolors=3, p=0.5, nnodes=5, seed=1)
# state = [0,1,2,1,0]*3
# # state = [2,1,2,2,0]
# # state = [-1]*5
# task.qannotations = [['R', 'G', 'B']]*15
# task.qannotations = [['1', '2', '3']]*5
# task.sannotations = ["$m_{%i}$"%i for i in range(15)]
# task.DisplayModel(state=state, flags=['no-hl'])

# task = GraphColoringTask.GraphColoring(ncolors=7, p=0.5, nnodes=4, seed=1)
# state = [0,1,2,6]
# task.qannotations = [['1', '2', '3', '4', '5', '6', '7']]*15
# # task.qannotations = [['1', '2', '3']]*5
# task.sannotations = ["$m_{%i}$"%i for i in range(15)]
# task.DisplayModel(state=state, flags=['PE'])


# task.DisplayModel(state=state, flags=['PE'])

# task.DisplayState(state)


# state = [4,3,5,1,0,2]
# task.sannotations = ["visit #%i"%i for i in range(ncities)]
# task.qannotations = [["city #%i"%i for i in range(ncities)]]*ncities
# task.sAnnotatePad = 40
# task.DisplayModel(state=state)
# task.DisplayState(state)


# calculate full distribution of energy states

# task = GraphColoringTask.GraphColoring(ncolors=3, p=0.5, nnodes=6, seed=1)
# e = []
# for state in itertools.product([0,1,2], repeat=6):
# 	e.append(task.EvalCost(state))

# from matplotlib import pyplot as plt

# plt.figure(figsize=(5,3))
# plt.hist(e, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
# plt.xlabel("Energy/cost, # of invalid edges")
# plt.ylabel("Number of states")
# plt.tight_layout()
# plt.show()

# e, counts = numpy.unique(e, return_counts=True)
# T=2
# pr = numpy.exp(-e/T)
# z = numpy.sum(pr*counts)
# pr = pr/z
# pr_e = pr*counts


# plt.figure(figsize=(5,3))
# plt.scatter(e, pr, label='individual states')
# plt.scatter(e, pr_e, label='energy levels')
# plt.xlabel("Energy/cost, # of invalid edges")
# plt.yscale('log')
# plt.ylabel("probability")
# plt.legend()
# plt.title("T=%.2f"%T)
# plt.tight_layout()
# plt.show()

# task.DisplayState([0,0,1,1,2,2])