import numpy
import networkx as nx
from matplotlib import pyplot as plt
from Tasks import BaseTask

class GraphColoring(BaseTask.BaseTask):

	def __init__(self, ncolors, p=None, G=None):
		self.e_th = 0
		if G is not None:
			self.G = G
		else:
			self.G = nx.erdos_renyi_graph(nnodes, p)
		self.Partitions2Nodes = list(self.G)
		self.nnodes = len(self.Partitions2Nodes)
		self.ncolors = ncolors
		
		self.SetPartitions(numpy.zeros([self.nnodes])+ncolors)

		self.InitKernelManager()

		for i, node1 in enumerate(self.Partitions2Nodes):
			conn_nodes = [e[1] for e in self.G.edges(node1)]
			for node2 in conn_nodes:
				j = self.Partitions2Nodes.index(node2)
				if i != j: #graph may have self-loops. we want to ignore them
					self.AddKernel(lambda n: self.IdentityKernel(n), i, j)

		self.CompileKernels()

	def DisplayState(self, state):
		#graphically represents the solution given by the supplied state
		#map solution indices to colors for plotting:
		colors = ['tab:red', 'tab:green', 'tab:blue']
		coloring = [colors[c] for c in state]
		plt.figure(figsize=(7, 7))
		nx.draw_kamada_kawai(self.G, node_color=coloring)
		plt.show()

	def defaultTemp(self, niters):
		PwlTemp = numpy.zeros([2, 2], dtype="float32")
		PwlTemp[0,0] = 0.3
		PwlTemp[0,1] = 0.05
		PwlTemp[1,0] = 0
		PwlTemp[1,1] = niters
		return PwlTemp

	def IdentityKernel(self, n=False):
		if n:
			return "IdentityKernel"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax)