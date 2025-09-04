import networkx as nx
from PottsPlayground.Tasks import BaseTask
from PottsPlayground.Kernels import BinaryQuadratic as BQK
import numpy

import dwave.embedding
import dimod
import minorminer


class MinorEmbedded(BaseTask.BaseTask):
	"""
	Uses utilities from d-wave ocean libraries to minor embed and map from a q=2 potts model with arbitrary connectivity
	to a fixed-hardware topology ising model. 
	"""

	def __init__(self, task, target_graph, ConstraintFactor=1):
		"""
		Creates the minor-embedded task.  May take a long time, and success is not guarenteed, since minor-embedding
		is itself an NP hard problem.

		:param task: An Ising-compatible Potts task (i.e. a Potts task with q=2 for all Spins)
		:param target_graph: A NetworkX graph of the allowed Ising model connectivity.
		:param ConstraintFactor: Energy penalty for disagreements within each physical chain representing a single logical spin.
		:type ConstraintFactor: float
		"""
		BaseTask.BaseTask.__init__(self)
		print("Finding embedding.")#this can easily take a very long time, so it is worthwhile to let user know what is up
		

		# self.source_label_mapping = {original_label: idx for idx, original_label in enumerate(G.nodes())}
		# self.source_embeddable_graph = nx.relabel_nodes(task.graph, self.source_label_mapping)

		embedding = minorminer.find_embedding(task.graph, target_graph)
		self.embedding = embedding
		self.target = target_graph
		self.e_th = -1e9
		self.task = task

		if len(embedding) == 0:
			print("Error: no embedding found.")
			return
		else:
			target_nodes = numpy.sum([len(embedding[src]) for src in embedding])
			source_nodes = numpy.sum([1 for src in embedding])
			available_nodes = target_graph.number_of_nodes()
			print("Minor embedding maps %i source nodes to %i of %i nodes available in the target graph"%(source_nodes, target_nodes, available_nodes))
		
		#convert source q=2 potts graph into format needed by dwave.embedding.embed_ising
		source_h = {}
		source_J = {}
		c = 0

		#assume/work with binary quadratic form, with variables having values of 0 or 1.
		for i, spin in enumerate(task.ListSpins()):
			h0 = task.biases[i,0]
			h1 = task.biases[i,1]	
			#treat h0 as a "background" contribution to the energy,
			#that should be moved from the local bias and rolled into a global offset
			h = h1-h0
			c = c + h0
			source_h[spin] = h
			for spin1, spin2, edge_data in task.graph.out_edges(spin, data=True):
				if (spin2, spin1) in source_J:
					continue #checks if the transpose weight has already been processed and added to the edge dictionary
				k = edge_data['kIndex']
				k = task.KernelList[k]*edge_data['weight']
				assert (k.shape == (2,2) and k[0,0] == 0. and k[0,1] == 0. and k[1,0] == 0.) #enforce that source is in binary quadratic form
				wij = k[1,1]
				source_J[(spin1, spin2)] = wij

		

		dwave_src_model = dimod.BinaryQuadraticModel(source_h, source_J, c, 'BINARY')  #the init func here is shit and you can't use keyword args
		dwave_embedded_model = dwave.embedding.embed_bqm(dwave_src_model, embedding, target_graph, chain_strength=ConstraintFactor)

		#convert dwave's returned ising format back into a PottsPlayground task:
		emb_h = dwave_embedded_model.linear
		emb_J = dwave_embedded_model.quadratic
		emb_c = dwave_embedded_model.offset

		qSizes = numpy.array([2]*target_nodes)
		self.SetPartitions(qSizes)
		self.nnodes = target_nodes

		#map between the target graph node numbers, and the indices of the MinorEmbedded PottsPlayground task
		self.Indx2TgtNode = [n for n in emb_h]
		self.TgtNode2Indx = {tgt:i for i, tgt in enumerate(self.Indx2TgtNode)}

		# self.biases = numpy.add(self.biases, emb_c/self.nnodes, dtype="float32") 
		for i in range(self.nnodes):
			TgtNode = self.Indx2TgtNode[i]
			self.AddBias(i, [emb_c/self.nnodes, emb_c/self.nnodes]) #redistribute the offset across all the nodes
			self.AddBias(i, [0, emb_h[TgtNode]])
			# self.biases[i,1] = self.biases[i,1] + emb_h[TgtNode]

		# self.InitKernelManager()

		for edge in emb_J:
			i = self.TgtNode2Indx[edge[0]]
			j = self.TgtNode2Indx[edge[1]]
			w = emb_J[edge]
			self.AddWeight(BQK, i, j, weight=w)
			# self.AddKernel(lambda n: self.BQMKernel(n), j, i, weight=w)

		self.CompileKernels()



	# def BQMKernel(self, n=False):
	# 	if n:
	# 		return "ik"
	# 	k = numpy.zeros([2,2])
	# 	k[1,1] = 1
	# 	return k

	def StateToMinor(self, state):
		"""
		Converts a state in the native Ising format to a state in the embedded format.
		This is useful for creating an initial condition for the embedded model that corresponds to 
		an equal initial condition in the original model.
		
		:param state: A vector of spin values, corresponding to spins in the source/logical Ising model.
		:return: A vector of spin values, corresponding to spins in the target/physical Ising model.
		"""
		minor_state = numpy.zeros([self.nnodes], dtype="int32")

		for src_name, embedded_spins in self.embedding.items():
			src_value = self.task.GetSpinFromState(src_name, state)
			for tgt_node in embedded_spins:
				i = self.TgtNode2Indx[tgt_node]
				minor_state[i] = src_value
		return minor_state

		# for i_src, value in enumerate(state):
		# 	print(self.embedding)
		# 	tgt_nodes = self.embedding[i_src]
		# 	for tgt_node in tgt_nodes:
		# 		i = self.TgtNode2Indx[tgt_node]
		# 		minor_state[i] = value
		# return minor_state


	def FuzzyMinorToState(self, minor_state):
		"""
		Uses majority-vote to collapse the minor-embedded representation back into the source ising model.

		:param minor_state: A vector of spin values, corresponding to spins in the target/physical Ising model.
		:return: A vector of spin values, corresponding to spins in the source/logical Ising model.
		"""
		state = numpy.zeros([self.task.nnodes], dtype="int32")
		for i in range(self.task.nnodes):
			minor_nodes = self.embedding[i]
			minor_node_values = [minor_state[self.TgtNode2Indx[ii]] for ii in minor_nodes]
			state[i] = 1*(numpy.mean(minor_node_values) > 0.5)
		return state

	def IsValidSemantics(self, minor_state):
		for i in range(self.task.nnodes):
			minor_nodes = self.embedding[i]
			minor_node_values = [minor_state[self.TgtNode2Indx[ii]] for ii in minor_nodes]
			chain_avg = numpy.mean(minor_node_values)
			if chain_avg > 0.0001 and chain_avg < 0.9999:
				return False
		#if here, state is valid from perspective of the minor embedding.
		#Still need to check if it is valid w.r.t. the source task:
		return self.task.IsValidSemantics(self.FuzzyMinorToState(minor_state))

