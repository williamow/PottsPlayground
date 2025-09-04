import numpy
import PottsPlayground

class Lattice(PottsPlayground.PottsModel):

	def __init__(self, q, sx, sy=None, pattern=None, wrap=True):
		"""
		Generates a sterotypical lattice model on a rectangular grid.
		Default connectivity is top/left/right/bottom, correlated with weight -1.
		A different repeating connectivity pattern can be used by specifying the pattern arguement.

		:param q: Dimension of each spin.
		:type q: int
		:param sx: Size of the lattice.
		:type sx: int
		:param sy: Optional, height of lattice.  Square lattice assumed if not specified.
		:type sy: int
		"""

		PottsPlayground.PottsModel.__init__(self)

		k = lambda n: PottsPlayground.Kernels.Identity(dim=q, n=n)

		if sy is None:
			sy = sx
		sn = numpy.linspace(0, sx*sy-1, sx*sy, dtype=int)
		self.AddSpins([q]*sx*sy, sn)


		sn = numpy.reshape(sn, [sx, sy])

		for x in range(sx):
			for y in range(sy):
				if x > 0:
					self.AddKernel(k, sn[x,y], sn[x-1,y], weight=-1)
				if x < sx-1:
					self.AddKernel(k, sn[x,y], sn[x+1,y], weight=-1)
				if y > 0:
					self.AddKernel(k, sn[x,y], sn[x,y-1], weight=-1)
				if y < sy-1:
					self.AddKernel(k, sn[x,y], sn[x,y+1], weight=-1)

		self.Compile()
		self.sn = sn
		self.q = q
		self.sx = sx
		self.sy = sy


	def StateToMatrix(self, state):
		m = numpy.zeros([self.sx, self.sy])
		for x in range(self.sx):
			for y in range(self.sy):
				m[x,y] = self.GetSpinFromState(self.sn[x,y], state)
		return m


if __name__ == '__main__':
	from matplotlib import pyplot as plt
	import PottsPlayground

	lattice_model = Lattice(5, 100)

	temp = PottsPlayground.Schedules.LinearTemp(1e7, 2.2, 1.6)

	res = PottsPlayground.Anneal(lattice_model, temp, algo="BirdsEye", model="PottsPrecompute", nReports=10)
	# res = PottsPlayground.Anneal(lattice_model, temp, nReports=10)

	for i in range(10):
		state = res["AllStates"][i,0,:]
		img = lattice_model.StateToMatrix(state)
		plt.figure()
		plt.imshow(img)
		plt.show()