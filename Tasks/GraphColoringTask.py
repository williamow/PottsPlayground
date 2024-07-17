import numpy
import networkx as nx
from matplotlib import pyplot as plt
from Tasks import BaseTask


class GraphColoring(BaseTask.BaseTask):

	def __init__(self, G, ncolors):
		self.G = G#nx.erdos_renyi_graph(nnodes, p)
		self.Partitions2Nodes = list(G)
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

	def InitKernels(self):
		#kernels has dimensions nKernels x qMax x qMax,
		#where qMax is the largest Potts node dimension.
		#to avoid branching, there is no default or "null" kernel;
		#a kernel of all zeros must be explicitly used instead.
		k = numpy.zeros([2,self.ncolors, self.ncolors], dtype="float32")

		#for graph coloring, the second kernel is just an identity matrix
		k[1,:,:] = numpy.eye(self.ncolors, dtype='float32')
		self.kernels = k
		return k

	def InitKernelMap(self):
		edges = numpy.zeros([self.nnodes, self.nnodes], dtype='int32')
		for i, node1 in enumerate(self.Partitions2Nodes):
			conn_nodes = [e[1] for e in self.G.edges(node1)]
			for node2 in conn_nodes:
				j = self.Partitions2Nodes.index(node2)
				edges[i,j] = 1
		self.kmap = edges
		return edges

	def ConnectivityList(self):
		#returns graph connectivity in a numpy array, in the following format:
		#each matrix row is a list of the nodes connected to the row's node.
		#the first value in the row is the number of edges+1, followed by a list of indexes to the connected nodes.
		G = self.G
		nnodes = G.number_of_nodes()
		degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
		dmax = max(degree_sequence)
		edges = numpy.zeros([nnodes, dmax+1], dtype=numpy.int32)
		for i in range(nnodes):

			conn_nodes = [e[1] for e in G.edges(i)]
			# print(conn_nodes)
			degree = len(conn_nodes)
			edges[i,0] = degree+1

			for j,node in enumerate(conn_nodes):
				edges[i,j+1] = node

		return edges

	def DisplayState(self, state):
		#graphically represents the solution given by the supplied state
		#map solution indices to colors for plotting:
		colors = ['tab:red', 'tab:green', 'tab:blue']
		coloring = [colors[c] for c in state]
		plt.figure(figsize=(7, 7))
		nx.draw_kamada_kawai(self.G, node_color=coloring)
		plt.show()

	def defaultPwlSchedule(self, niters):
		PwlTemp = numpy.zeros([2, 2], dtype="float32")
		PwlTemp[0,0] = 0.3
		PwlTemp[0,1] = 0.05
		PwlTemp[1,0] = 0
		PwlTemp[1,1] = niters
		self.PwlTemp = PwlTemp

		PwlLegalWeight = numpy.zeros([2, 2], dtype="float32")
		PwlLegalWeight[0,0] = 1
		PwlLegalWeight[0,1] = 1
		PwlLegalWeight[1,0] = 0
		PwlLegalWeight[1,1] = niters
		self.PwlLegalWeight = PwlLegalWeight

	def IdentityKernel(self, n=False):
		if n:
			return "IdentityKernel"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax)