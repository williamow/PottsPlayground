import numpy
import networkx as nx
from matplotlib import pyplot as plt
from PottsPlayground.Tasks.BaseTask import BaseTask
from PottsPlayground.Kernels import Identity as IdentityKernel

class GraphColoring(BaseTask):
	"""
	Models the Graph Node coloring problem as a Potts model, which is just about the simplest possible model.
	Rather than asking how many colors are needed to color a graph, which is the typical framing of graph coloring,
	the simple Potts model form asks if a coloring with a set number of colors is possible.
	"""

	def __init__(self, ncolors, p=None, nnodes=None, G=None, seed=None):
		"""
		Either construct a random Erdos-Renyi graph, or use a graph given by G.

		Parameters for either initialization:

		:param ncolors: How many colors are to be used.
		:type ncolors: int
		:param seed: Optional seed for random graph construction, for repeatable results.
		:type seed: int

		Parameters for coloring a random Erdos-Renyi graph:

		:param p: Edge probability for random graph construction (i.e. edge density)
		:type p: float
		:param nnodes: Number of graph nodes for random graph construction
		:param nnodes: int

		Parameters for coloring a given graph:
		
		:param G: The graph to be colored.
		:type G: undirected networkx graph
		"""
		BaseTask.__init__(self)
		# self.e_th = -100
		if G is not None:
			self.G = G
		else:
			self.G = nx.erdos_renyi_graph(nnodes, p, seed=seed)
		# self.Partitions2Nodes = list(self.G)
		# self.nnodes = len(self.Partitions2Nodes)
		self.ncolors = ncolors

		ik = lambda n: IdentityKernel(ncolors, n)
		
		#copy spins and edges from G to the BaseTask class:
		[self.AddSpins([ncolors], [node_name]) for node_name in self.G.nodes]
		# for stuff in self.G.edges:
		# 	print(stuff)
		[self.AddWeight(ik, t[0], t[1]) for t in self.G.edges]
		

		# for i, node in enumerate(self.graph.nodes):
		# self.AddSpins(numpy.zeros([self.nnodes])+ncolors)	

		# for i, node1 in enumerate(self.Partitions2Nodes):
		# 	conn_nodes = [e[1] for e in self.G.edges(node1)]
		# 	for node2 in conn_nodes:
		# 		j = self.Partitions2Nodes.index(node2)
		# 		if i != j: #graph may have self-loops. we want to ignore them
		# 			self.AddKernel(lambda n: self.IdentityKernel(n), i, j)

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