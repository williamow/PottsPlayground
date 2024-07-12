import numpy
from matplotlib import pyplot as plt

#base class for problems solvable as Potts models.
#also includes a python/numpy native Potts solver, as a slow-end reference.

class BaseTask:

	# ====================================================================================== 
	def SetPartitions(self, qSizes):
		self.qSizes = numpy.zeros(qSizes.shape, dtype='int32')
		self.nnodes = self.qSizes.shape[0]
		numpy.copyto(self.qSizes, qSizes, casting='unsafe')
		self.qCumulative = numpy.cumsum(self.qSizes, dtype="int32")
		self.Partitions = numpy.zeros([int(self.qCumulative[-1])], dtype="int32")
		self.Partition_states = numpy.zeros([int(self.qCumulative[-1])], dtype="int32")
		for i in range(qSizes.shape[0]):
			self.Partitions[int(self.qCumulative[i]-qSizes[i]):int(self.qCumulative[i])] = i
			self.Partition_states[int(self.qCumulative[i]-qSizes[i]):int(self.qCumulative[i])] = numpy.linspace(0,qSizes[i]-1,int(qSizes[i]))

		self.biases = numpy.zeros([self.nnodes, numpy.max(self.qSizes)], dtype="float32")

	#makes a full, big weight matrix out of kernel weights.  Use sparingly.
	#Very easy for this function to request memory allocations much bigger than the system limit.
	def MakeWeights(self):
		#this function assumes that partition information is already set.
		dim = int(self.qCumulative[-1])
		nUnits = self.qSizes.shape[0]
		weights = numpy.zeros([dim, dim], dtype="float32")
		for i in range(nUnits):
			for j in range(nUnits):
				#start by getting dimensions and positions of the particular kernel:
				sz_x = int(self.qSizes[i])
				sz_y = int(self.qSizes[j])
				pos_x = int(self.qCumulative[i]-sz_x)
				pos_y = int(self.qCumulative[j]-sz_y)
				k = self.kernels[self.kmap[i,j], 0:sz_x, 0:sz_y]
				weights[pos_x:pos_x+sz_x, pos_y:pos_y+sz_y] = k
		self.weights = weights
	
	#python-domain calculation of the energy of a state, based on kernels.
	def EvalCost(self, state):
		cost = 0
		for i,m in enumerate(state):
			cost = cost + 2*self.biases[i,m]
			for j,n in enumerate(state):
				cost = cost + self.kernels[int(self.kmap[i,j]), int(m), int(n)]
		return cost/2.

	

	def DisplayWeights(self):
		plt.figure() #show each different kernel sequentially
		k = -self.weights #negate, so darker means stronger weight
		plt.imshow(k, cmap='gray', vmin=numpy.min(k), vmax=numpy.max(k))
		plt.title("full weight matrix")
		plt.show()

	# ================================================================================ Potts kernel/weight construction utilities
	def DisplayKernels(self):
		for name in self.KernelDict:
			plt.figure() #show each different kernel sequentially
			k = -self.KernelList[self.KernelDict[name]] #negate, so darker means stronger weight
			print("Kernel %s has dimensions %i,%i"%(name, k.shape[0], k.shape[1]))
			plt.imshow(k, cmap='gray', vmin=numpy.min(k), vmax=numpy.max(k))
			plt.title(name)
			plt.show()


	#==========================kernel manager system
	def InitKernelManager(self):
		self.KernelList = []
		self.KernelDict = {}
		self.kMapLists = {}

	def AddKernel(self, creator, i, j, weight=1, scale=0):
		#first, see if kernel has already been created, and if not, create it:
		kName = creator(True)
		if kName not in self.KernelDict:
			self.KernelList.append(creator(False))
			self.KernelDict[kName] = len(self.KernelList)-1
		kIndex = self.KernelDict[kName]

		#add the kernel to the dict:
		if i not in self.kMapLists:
			self.kMapLists[i] = []
		self.kMapLists[i].append((kIndex, weight, j, scale))

	def CompileKernels(self):
		#compiles both the kernels and the kernel map.
		#dense kernels:
		maxQ = numpy.max(self.qSizes)
		nKernels = len(self.KernelList)
		self.kernels = numpy.zeros([nKernels, maxQ, maxQ], dtype="float32")
		for i, kernel in enumerate(self.KernelList):
			self.kernels[i, :kernel.shape[0], :kernel.shape[1]] = kernel

		#sparse kmap:
		nPartitions = self.qSizes.shape[0]
		maxDensity = numpy.max([len(self.kMapLists[i]) for i in self.kMapLists])
		total_count = 0
		self.kmap_sparse = numpy.zeros([nPartitions, maxDensity+1,4], dtype="float32")-1 #initialize all to minus 1
		for i in self.kMapLists:
			connections = self.kMapLists[i] 
			self.kmap_sparse[i,0,0] = len(connections)+1 #+1, so that it directly tells the stop index, rather than the number of elements
			for c, conn in enumerate(connections):
				self.kmap_sparse[i,c+1,:] = conn
			total_count = total_count + c
		print("Done making sparse kernel map, density = %.3f"%(total_count/nPartitions**2))