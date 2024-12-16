import numpy
import networkx as nx
from matplotlib import pyplot as plt
from PottsPlayground.Tasks.BaseTask import BaseTask

class GraphColoring(BaseTask):
	"""
	Models the Graph Node coloring problem as a Potts model.  Just about the simplest Potts mode design.
	Rather than asking how many colors are needed to color a graph, which is the typical framing of graph coloring,
	the simple Potts model form asks if a coloring with a set number of colors is possible.
	"""

	def __init__(self, ncolors, p=None, nnodes=None, G=None, seed=None):
		"""
		:param ncolors: How many colors are to be used.
		:type ncolors: int
		:param G: The graph to be colored.
		type G: undirected networkx graph
		"""
		self.e_th = 0
		if G is not None:
			self.G = G
		else:
			self.G = nx.erdos_renyi_graph(nnodes, p, seed=seed)
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
		"""
		graphically represents the solution given by the supplied state.

		"""
		#map solution indices to colors for plotting:
		colors = ['tab:red', 'tab:green', 'tab:blue']
		coloring = [colors[c] for c in state]
		plt.figure(figsize=(7, 7))
		nx.draw_kamada_kawai(self.G, node_color=coloring)
		plt.show()

	def IdentityKernel(self, n=False):
		if n:
			return "IdentityKernel"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax)