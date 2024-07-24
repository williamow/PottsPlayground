import numpy
from Tasks import BaseTask
from matplotlib import pyplot as plt

#regular euclidean traveling salesman on a plane, bounded in the coordinate box (0,1)x(0,1)
class TravelingSalesman(BaseTask.BaseTask):

	def __init__(self, ncities, costs=None, ConstraintFactor=1):
		#set up task-specific variables ==========================================
		self.e_th = -1e14
		self.ncities = ncities
		self.nnodes = ncities
		self.ConstraintFactor = ConstraintFactor

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
			self.distances = costs #allow an external distance/cost matrix to be used

		#create actual Potts formulation of the task =============================
		#set up partitioning - BaseTask function
		self.SetPartitions(numpy.zeros([ncities]) + ncities)

		self.InitKernelManager()
		for i in range(self.ncities):
			for j in range(self.ncities):
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

		plt.show()

	def defaultTemp(self, niters, tmax=10):
		PwlTemp = numpy.zeros([2, 3], dtype="float32")
		PwlTemp[0,0] = tmax
		PwlTemp[0,1] = tmax/2
		PwlTemp[0,2] = 0.02
		PwlTemp[1,0] = 0
		PwlTemp[1,1] = niters*0.8
		PwlTemp[1,2] = niters
		self.PwlTemp = PwlTemp