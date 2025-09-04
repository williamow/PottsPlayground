import numpy
from PottsPlayground.Tasks.BaseTask import BaseTask
from PottsPlayground.Kernels import BinaryQuadratic as BQK
from matplotlib import pyplot as plt

class Binarized(BaseTask):
	"""
	Maps any Potts model to an Ising model by splitting every q-dimensional Spin
	into q 2-dimensional spins and adding soft one-hot constraints 
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
		BaseTask.__init__(self)
		self.PottsTask = PottsTask
		self.e_th = PottsTask.e_th

		PottsSpins = PottsTask.ListSpins()

		#create new partitions and set biases
		for i, spin in enumerate(PottsSpins):
			q = PottsTask.SpinSize(spin)
			biases = PottsTask.GetSpinBias(spin)
			subspins = ["%i-%i"%(i,j) for j in range(q)]
			if q == 2:
				self.AddSpins([2], [subspins[0]])
				self.SetBias(subspins[0], biases)
			else:
				self.AddSpins([2]*q, subspins)
				[self.SetBias(subspins[j], [0,biases[j]]) for j in range(q)]

				#add one-hot inhibitory weights.
				#zero bias, one bias, and correlation penalty for enforcing one-hotness amoung Pbits in a Potts node.
				#Designed so that the energy is zero when the collection of P-bits is one-hot,
				#energy is E+=ConstraintFactor if there are two hot or zero hot, and E += >ConstraintFactor if more than two are hot.
				H0 = ConstraintFactor/q 
				H1 = -(q-1)*H0
				w_inhib = 2*q*H0
				for j in range(q):
					self.AddBias(subspins[j], [H0, H1])
					for jj in range(j+1, q):
						self.AddWeight(BQK, subspins[j], subspins[jj], weight=w_inhib)

		for i, spin1 in enumerate(PottsSpins):
			for j, spin2 in enumerate(PottsSpins):
				if j <= i:
					continue #assume symetric, only iterate over half
				w = PottsTask.TotalWeight(spin1, spin2)
				if numpy.sum(w) == 0:
					continue #no weight here

				#if either Potts spin is only size 2,
				#simplify it to a weird sort of sudo-size-1 potts spin
				#by factoring some of the weights into the biases,
				#and then the remaining size-1 weight matrix can be processed
				#just like any size q weight matrix
				if w.shape[0] == 2:
					w[1,:] = w[1,:]-w[0,:]
					if w.shape[1] == 2:
						self.AddBias("%i-%i"%(j,0), w[0,:])
					else:
						[self.AddBias("%i-%i"%(j,y), [0, w[0,y]]) for y in range(w.shape[1])]
					w = w[1,:]
					w = numpy.expand_dims(w, 0)

				if w.shape[1] == 2:
					w[:,1] = w[:,1]-w[:,0]
					if w.shape[0] == 1:
						#this case is slightly different than above, 
						#since w would have been modified above
						self.AddBias("%i-%i"%(i,0), [0,w[0,0]])
					else:
						[self.AddBias("%i-%i"%(i,x), [0, w[x,0]]) for x in range(w.shape[0])]
					w = w[:,1]
					w = numpy.expand_dims(w, 1)

				q1, q2 = w.shape
				for x in range(q1): #iterate through all the values in the kernel:
					for y in range(q2):
						self.AddWeight(BQK, "%i-%i"%(i,x), "%i-%i"%(j,y), weight=w[x,y])


		# self.SetPartitions(numpy.array([2]*numpy.sum(PottsTask.qSizes)))

		# #copy biases first:
		# for Pi, (q, qc) in enumerate(zip(PottsTask.qSizes, PottsTask.qCumulative)):
		# 	m10 = qc-q #the P-bit number corresponding to the first state of Potts node i
		# 	for Pmi in range(q): #Potts m_i value
		# 		Ii = qc-q+Pmi #Ising i index
		# 		self.AddBias(Ii, [0,PottsTask.biases[Pi, Pmi]])
				# self.biases[Ii, 1] = PottsTask.biases[Pi, Pmi]

		#iterate through the Potts Model's Kernel system,
		#copying weights over into ising model Kernel system
		# self.InitKernelManager()
		# nUnits = PottsTask.qSizes.shape[0]
		# for i in range(nUnits): #iterates through rows of weight matrix
		# 	for c, (i, j, edge_data) in enumerate(PottsTask.graph.out_edges([i], data=True)):
		# 		# kmap_sparse[i,c+1,:] = [edge_data['kIndex'], edge_data['weight'], v]
		# 	# for kspec in PottsTask.kMapLists[i]: #iterates through sparse format of column kernels
		# 		#start by getting dimensions and positions of the particular kernel:
		# 		# j = v#kspec[2]
		# 		k = edge_data['kIndex']#kspec[0]
		# 		w = edge_data['weight']
		# 		sz_x = int(PottsTask.qSizes[i])
		# 		sz_y = int(PottsTask.qSizes[j])
		# 		pos_x = int(PottsTask.qCumulative[i]-sz_x)
		# 		pos_y = int(PottsTask.qCumulative[j]-sz_y)
		# 		k = PottsTask.kernels[k, 0:sz_x, 0:sz_y]*w
		# 		for x in range(sz_x): #iterate through all the values in the kernel:
		# 			for y in range(sz_y):
		# 				self.AddKernel(BQK, pos_x+x, pos_y+y, weight=k[x,y])

		#add additional kernels for representing the one-hot inhibition within Potts nodes:
		# for i, (q, qc) in enumerate(zip(PottsTask.qSizes, PottsTask.qCumulative)):
			
		self.PottsSpins = PottsSpins
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
		PottsState = numpy.zeros([len(self.PottsSpins)], dtype='int32')
		for i, PottsSpin in enumerate(self.PottsSpins):
			q = self.PottsTask.SpinSize(PottsSpin)
			subspins = ["%i-%i"%(i,j) for j in range(q)]
			if q == 2:
				s = self.GetSpinFromState(subspins[0], IsingState)
				self.PottsTask.SetSpinInState(PottsSpin, PottsState, s)
			else:
				s = [self.GetSpinFromState(subspins[j], IsingState) for j in range(q)]
				self.PottsTask.SetSpinInState(PottsSpin, PottsState, numpy.argmax(s))
		return PottsState

		# for i, (q, qc) in enumerate(zip(self.PottsTask.qSizes, self.PottsTask.qCumulative)):
		# 	Potts_node_pbits = IsingState[qc-q:qc]
		# 	PottsState.append(numpy.argmax(Potts_node_pbits))
		# return numpy.array(PottsState, dtype='int32')

	def PottsToIsingSemantics(self, PottsState):
		"""
		Converts a Potts state from the source task into an eqivalent Ising state.

		:param PottsState: Vector of Potts spin states.
		:type PottsState: 1-D numpy int
		:return: A vector representing a Ising model state.
		:rtype: 1-D numpy int
		"""
		IsingState = numpy.zeros([len(self.ListSpins())], dtype='int32')
		for i, PottsSpin in enumerate(self.PottsSpins):
			s = self.PottsTask.GetSpinFromState(PottsSpin, PottsState)
			q = self.PottsTask.SpinSize(PottsSpin)
			subspins = ["%i-%i"%(i,j) for j in range(q)]
			if q == 2:
				self.SetSpinInState("%i-%i"%(i,0), IsingState, s)
			else:
				self.SetSpinInState("%i-%i"%(i,s), IsingState, 1)
		return IsingState

		# for i, q in enumerate(self.PottsTask.qSizes):
		# 	partial_state = [0]*q
		# 	partial_state[PottsState[i]] = 1
		# 	IsingState = IsingState + partial_state
		# return numpy.array(IsingState, dtype='int32')

	def IsValidSemantics(self, IsingState):
		#checks if the state is valid, in the sense of whether or not the state follows the basic rules of the combintorial problem
		#valid meaning that the state represents a valid tour of cities,
		#and invalid represents a state that does not correspond to a valid tour of cites (cities visited more than once or not at all)
		
		# for i, (q, qc) in enumerate(zip(self.PottsTask.qSizes, self.PottsTask.qCumulative)):
		# 	Potts_node_pbits = state[qc-q:qc]
		# 	if numpy.sum(Potts_node_pbits) != 1:
		# 		return False

		PottsState = numpy.zeros([len(self.PottsSpins)], dtype='int32')
		for i, PottsSpin in enumerate(self.PottsSpins):
			q = self.PottsTask.SpinSize(PottsSpin)
			subspins = ["%i-%i"%(i,j) for j in range(q)]
			if q == 2:
				s = self.GetSpinFromState(subspins[0], IsingState)
				self.PottsTask.SetSpinInState(PottsSpin, PottsState, s)
			else:
				s = [self.GetSpinFromState(subspins[j], IsingState) for j in range(q)]
				if numpy.sum(s) != 1:
					return False
				self.PottsTask.SetSpinInState(PottsSpin, PottsState, numpy.argmax(s))
		
		#finally, also check if the state is valid in the Potts model sense
		return self.PottsTask.IsValidSemantics(PottsState)
