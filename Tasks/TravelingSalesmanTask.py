import numpy
from Tasks import BaseTask
from matplotlib import pyplot as plt

#regular euclidean traveling salesman on a plane, bounded in the coordinate box (0,1)x(0,1)
class TravelingSalesman(BaseTask.BaseTask):

	def __init__(self, ncities, ConstraintFactor=1):
		#set up task-specific variables ==========================================
		self.ncities = ncities
		self.nnodes = ncities
		self.coords = numpy.random.random([2,ncities])
		self.ConstraintFactor = ConstraintFactor

		#set up distances here, so that it only needs to be set up once:
		self.distances = numpy.zeros([ncities,ncities], dtype="float32")
		for i in range(ncities):
			for j in range(i+1,ncities):
				d = ((self.coords[0,i]-self.coords[0,j])**2+(self.coords[1,i]-self.coords[1,j])**2)**0.5
				self.distances[i,j] = d
				self.distances[j,i] = d

		#create actual Potts formulation of the task =============================
		#set up partitioning - BaseTask function
		self.SetPartitions(numpy.zeros([ncities]) + ncities)
		self.InitKernels()
		self.InitKernelMap()

	def InitKernels(self):
		k = numpy.zeros([3, self.ncities, self.ncities], dtype='float32')
		k[1,:,:] = numpy.eye(self.ncities, dtype='float32')*self.ConstraintFactor
		k[2,:,:] = numpy.eye(self.ncities, dtype='float32')*self.ConstraintFactor + self.distances
		self.kernels = k
		return k

	def InitKernelMap(self):
		kmap = numpy.zeros([self.ncities, self.ncities], dtype="int32")+1 #default is kernel no. 1
		kmap = kmap - numpy.eye(self.ncities, dtype="int32") #the diagonal should point to kernel no. 0
		#the off-diagonals should point to kernel no. 2:
		for i in range(self.ncities):
			kmap[i, (i+1)%self.ncities] = 2
			kmap[(i+1)%self.ncities, i] = 2
		self.kmap = kmap
		return kmap

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