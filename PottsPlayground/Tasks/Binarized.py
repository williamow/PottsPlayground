import numpy
from PottsPlayground.Tasks.BaseTask import BaseTask
from PottsPlayground.Kernels import BinaryQuadratic as BQK
from matplotlib import pyplot as plt

#create an Ising model task out of any potts model task.
#It is still structurally a Potts model, but each n-way Potts nodes are split into n 2-way Potts nodes, equivalent to an Ising Model.
class Binarized(BaseTask):
	"""
	Maps any Potts model to an Ising model by splitting every q-dimensional Spin into q 2-dimensional spins and adding soft one-hot constraints 
	to each group of Ising spins.
	"""
	def __init__(self, PottsTask, ConstraintFactor=1):
		"""
		Turns a Potts model into an Ising model.  The resulting task is still a PottsTask, but all spins are binary.  The task can still
		be annealed as a Potts model, but can additionally be annealed using dedicated Ising-model code that runs a little faster.

		:param PottsTask: Any object correctly derived from the BaseTask class. Could even be another BinarizedTask, although that would not be productive.
		:param ConstraintFactor: Weight penalty for two Ising bits both being "hot" if they represent different values of the same Potts spin.
		:type ConstraintFactor: float
		"""
		self.PottsTask = PottsTask
		self.e_th = PottsTask.e_th
		self.SetPartitions(numpy.array([2]*numpy.sum(PottsTask.qSizes)))

		#copy biases first:
		for Pi, (q, qc) in enumerate(zip(PottsTask.qSizes, PottsTask.qCumulative)):
			m10 = qc-q #the P-bit number corresponding to the first state of Potts node i
			for Pmi in range(q): #Potts m_i value
				Ii = qc-q+Pmi #Ising i index
				self.biases[Ii, 1] = PottsTask.biases[Pi, Pmi]

		#iterate through the Potts Model's Kernel system,
		#copying weights over into ising model Kernel system
		self.InitKernelManager()
		nUnits = PottsTask.qSizes.shape[0]
		for i in range(nUnits): #iterates through rows of weight matrix
			for c, (i, j, edge_data) in enumerate(PottsTask.graph.out_edges([i], data=True)):
				# kmap_sparse[i,c+1,:] = [edge_data['kIndex'], edge_data['weight'], v]
			# for kspec in PottsTask.kMapLists[i]: #iterates through sparse format of column kernels
				#start by getting dimensions and positions of the particular kernel:
				# j = v#kspec[2]
				k = edge_data['kIndex']#kspec[0]
				w = edge_data['weight']
				sz_x = int(PottsTask.qSizes[i])
				sz_y = int(PottsTask.qSizes[j])
				pos_x = int(PottsTask.qCumulative[i]-sz_x)
				pos_y = int(PottsTask.qCumulative[j]-sz_y)
				k = PottsTask.kernels[k, 0:sz_x, 0:sz_y]*w
				for x in range(sz_x): #iterate through all the values in the kernel:
					for y in range(sz_y):
						self.AddKernel(BQK, pos_x+x, pos_y+y, weight=k[x,y])

		#add additional kernels for representing the one-hot inhibition within Potts nodes:
		for i, (q, qc) in enumerate(zip(PottsTask.qSizes, PottsTask.qCumulative)):
			#zero bias, one bias, and correlation penalty for enforcing one-hotness amoung Pbits in a Potts node.
			#Designed so that the energy is zero when the collection of P-bits is one-hot,
			#energy is E+=ConstraintFactor if there are two hot or zero hot, and E += >ConstraintFactor if more than two are hot.
			H0 = ConstraintFactor/q 
			H1 = -(q-1)*H0
			w_inhib = 2*q*H0
			m10 = qc-q #the P-bit number corresponding to the first state of Potts node i
			for m1 in range(q):
				#add biases. Add to existing biases, so Potts model biases to the energy are preserved
				self.biases[m10+m1, :] = self.biases[m10+m1, :] + [H0, H1]
				for m2 in range(q):
					if m1 == m2:
						continue
					else:
						self.AddKernel(BQK, m10+m1, m10+m2, weight=w_inhib)

		self.CompileKernels()


	def IsingToPottsSemantics(self, IsingState):
		"""
		if the Ising state is valid, this will return the Potts state for the associated Potts Model.
		Invalid states will also be processed into a valid Potts state, but the transformation is undefined.
		The returned Potts state can then be sent to the parent Potts Task for interpretation or visualization as a graph coloring solution, etc.

		:param IsingState: Vector of Ising bit states.
		:type IsingState: 1-D numpy int
		:return: A vector representing a Potts model state.
		:rtype: 1-D numpy int
		"""
		PottsState = []
		for i, (q, qc) in enumerate(zip(self.PottsTask.qSizes, self.PottsTask.qCumulative)):
			Potts_node_pbits = IsingState[qc-q:qc]
			PottsState.append(numpy.argmax(Potts_node_pbits))
		return numpy.array(PottsState, dtype='int32')

	def PottsToIsingSemantics(self, PottsState):
		"""
		Converts a Potts state from the source task into an eqivalent Ising state.

		:param PottsState: Vector of Potts spin states.
		:type PottsState: 1-D numpy int
		:return: A vector representing a Ising model state.
		:rtype: 1-D numpy int
		"""
		IsingState = []
		for i, q in enumerate(self.PottsTask.qSizes):
			partial_state = [0]*q
			partial_state[PottsState[i]] = 1
			IsingState = IsingState + partial_state
		return numpy.array(IsingState, dtype='int32')

	def IsValidSemantics(self, state):
		#checks if the state is valid, in the sense of whether or not the state follows the basic rules of the combintorial problem
		#valid meaning that the state represents a valid tour of cities,
		#and invalid represents a state that does not correspond to a valid tour of cites (cities visited more than once or not at all)
		for i, (q, qc) in enumerate(zip(self.PottsTask.qSizes, self.PottsTask.qCumulative)):
			Potts_node_pbits = state[qc-q:qc]
			if numpy.sum(Potts_node_pbits) != 1:
				return False

		#if we get here, it means this Ising model represents a valid state of the parent Potts model.
		PottsState = self.IsingToPottsSemantics(state)
		# if self.PottsTask.IsValidSemantics(PottsState):
			# print(state, PottsState)
		#finally, also check if the state is valid in the Potts model sense
		return self.PottsTask.IsValidSemantics(PottsState)
