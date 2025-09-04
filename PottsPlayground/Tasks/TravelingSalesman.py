import numpy
from PottsPlayground.Tasks import BaseTask
from matplotlib import pyplot as plt


class TravelingSalesman(BaseTask.BaseTask):
	"""
	Potts model representation of the Traveling Salesman Problem (TSP).
	"""

	def __init__(self, ncities=None, costs=None, ConstraintFactor=1, seed=None):
		"""
		Two initializations are possible.  Either a regular euclidean traveling salesman on a plane
		is randomly generated, bounded in the coordinate box (0,1)x(0,1); or a matrix of segment costs is passed.

		For generating a regular euclidean TSP instance:

		:param ncities: The number of cities to randomly generate.
		:type ncities: int
		:param seed: Optional, define a seed to make the process repeatable.

		For using an arbitrary set of costs:

		:param costs: Matrix where each element i,j is the cost to travel from i to j.
		:type costs: NxN Numpy float
		"""

		#set up task-specific variables =========================================
		BaseTask.BaseTask.__init__(self)
		if seed is not None:
			numpy.random.seed(seed)

		if costs is None:
			#set up distances here, so that it only needs to be set up once:
			self.coords = numpy.random.random([2,ncities])
			self.distances = numpy.zeros([ncities,ncities], dtype="float32")
			for i in range(ncities):
				for j in range(i+1,ncities):
					d = ((self.coords[0,i]-self.coords[0,j])**2+(self.coords[1,i]-self.coords[1,j])**2)**0.5
					self.distances[i,j] = d
					self.distances[j,i] = d
		else:
			assert costs.shape[0] == costs.shape[1]
			self.distances = costs #allow an external distance/cost matrix to be used
			ncities = costs.shape[0]

		self.e_th = -1e14
		self.ncities = ncities
		self.nnodes = ncities
		self.ConstraintFactor = ConstraintFactor

		#create actual Potts formulation of the task =============================
		#set up partitioning - BaseTask function
		self.SetPartitions(numpy.zeros([ncities]) + ncities)

		for i in range(self.ncities):
			for j in range(self.ncities):
				if (i==j):
					continue
				self.AddKernel(lambda n: self.constrain(n), i, j, weight=1)

		for i in range(self.ncities):
			self.AddKernel(lambda n: self.distance(n), i, (i+1)%self.ncities, weight=1)
			self.AddKernel(lambda n: self.distance(n), (i+1)%self.ncities, i, weight=1)

		self.CompileKernels()


	#Kernel definitions:
	def constrain(self, n=False):
		if n:
			return "constrain"
		return numpy.eye(self.ncities) * self.ConstraintFactor

	def distance(self, n=False):
		if n:
			return "distance"
		return self.distances


	def DisplayState(self, state):
		"""
		If the regular euclidean TSP was used, this plots the city locations
		and connects them with line segments to show the tour represented by the given state.

		:param state: Vector of spin magnetizations.
		:type state: 1-D Numpy int
		"""
		plt.figure()

		#draw cities:
		plt.scatter(self.coords[0,:], self.coords[1,:])

		#draw route:
		for leg in range(self.ncities):
			start_x = self.coords[0,state[leg]]
			start_y = self.coords[1,state[leg]]
			end_x = self.coords[0,state[(leg+1)%self.ncities]]
			end_y = self.coords[1,state[(leg+1)%self.ncities]]
			x = [start_x, end_x]
			y = [start_y, end_y]
			plt.plot(x,y,c='black')

		plt.gca().set_aspect('equal')
		plt.show()

	# def defaultTemp(self, niters, tmax=10):
	# 	PwlTemp = numpy.zeros([2, 3], dtype="float32")
	# 	PwlTemp[0,0] = tmax
	# 	PwlTemp[0,1] = tmax/2
	# 	PwlTemp[0,2] = 0.02
	# 	PwlTemp[1,0] = 0
	# 	PwlTemp[1,1] = niters*0.8
	# 	PwlTemp[1,2] = niters
	# 	self.PwlTemp = PwlTemp


	def StateDict(self, temp):
		import itertools
		#calculates the ground-truth distribution of states for the given temperature.
		#Obviously, only feasible for a small number of cities.
		state = [i for i in range(self.ncities)]
		energies = []
		perms = list(itertools.permutations(state))
		for perm in perms:
			energies.append(self.EvalCost(perm))
		print("There are %i different tours of %i cities"%(len(energies), self.ncities))
		energies = numpy.array(energies)
		probs = numpy.exp(-energies/temp)
		z = numpy.sum(probs)
		probs = probs/z
		print("Avg E: %.3f"%numpy.sum(probs*energies))
		
		#collect energies and states and probs into a dict:
		d = {}
		for st, pr, e in zip(perms, probs, energies):
			d[st] = (pr, e)
		return d


	# I don't think this is needed since valid and invalid sets are distinguished in the main energy calculating code
	# def AvgE(self, temp):
	# 	import itertools
	# 	#calculates the ground-truth distribution of states for the given temperature.
	# 	#Obviously, only feasible for a small number of cities.
	# 	state = [i for i in range(self.ncities)]
	# 	energies = []
	# 	for perm in itertools.permutations(state):
	# 		energies.append(self.EvalCost(perm))
	# 	print("There are %i different tours of %i cities"%(len(energies), self.ncities))
	# 	energies = numpy.array(energies)
	# 	probs = numpy.exp(-energies/temp)
	# 	z = numpy.sum(probs)
	# 	probs = probs/z
	# 	return numpy.sum(probs*energies)

	def IsValidSemantics(self, state):
		#checks if the state is valid, in the sense of whether or not the state follows the basic rules of the combintorial problem
		#valid meaning that the state represents a valid tour of cities,
		#and invalid represents a state that does not correspond to a valid tour of cites (cities visited more than once or not at all)
		if numpy.unique(state).shape[0] == self.ncities:
			return True
		return False
