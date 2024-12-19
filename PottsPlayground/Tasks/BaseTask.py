import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch
from matplotlib import cm
import matplotlib
import math
import types
import itertools
import networkx

class BaseTask:
	"""
	Provides a framework for creating, storing, and manipulating Potts models.  Provides a bridge between the semantic structure
	of various combinatorial problems and a Potts-model format used by the Potts annealing engine.  Usually instantiated as a base class,
	but can also be used alone to create arbitrary Potts model tasks.
	"""

	def SetPartitions(self, qSizes):
		"""
		First step in creating a Potts model. The qSizes vector describes the structure of the Potts Model state, including how many spins there are and
		how many states each spin may take.  The spins do not need to all have the same number of states.

		:param qSizes: The number of possibilities for each magnetization state of the Potts model.
   		:type qSizes: 1-D Numpy array of ints or list[int]
   		:return: None
		"""
		if type(qSizes) == list:
			self.nnodes = len(qSizes)
		else:
			self.nnodes = qSizes.shape[0]
		self.qSizes = numpy.zeros([self.nnodes], dtype='int32')
		numpy.copyto(self.qSizes, qSizes, casting='unsafe')
		self.qCumulative = numpy.cumsum(self.qSizes, dtype="int32")
		self.Partitions = numpy.zeros([int(self.qCumulative[-1])], dtype="int32")
		self.Partition_states = numpy.zeros([int(self.qCumulative[-1])], dtype="int32")
		for i in range(self.nnodes):
			self.Partitions[int(self.qCumulative[i]-qSizes[i]):int(self.qCumulative[i])] = i
			self.Partition_states[int(self.qCumulative[i]-qSizes[i]):int(self.qCumulative[i])] = numpy.linspace(0,qSizes[i]-1,int(qSizes[i]))

		self.biases = numpy.zeros([self.nnodes, numpy.max(self.qSizes)], dtype="float32")

		self.e_th = -1e9

	def InitKernelManager(self):
		"""
		Initializes the kernel manager system, which is used for building up weight matrices piece by piece,
		by initializing a few empty variables.  SetPartitions(...) must be called beforehand.

		:return: None
		"""
		self.KernelList = []
		self.KernelDict = {}
		self.graph = networkx.MultiDiGraph()
		#directed graph, so that the kernel in each direction can be different (usually just a transpose.)
		#MultiGraph, so that multiple kernels can be applied to the same edge.
		#fill the graph with all nodes here in advance:
		for i in range(self.nnodes):
			self.graph.add_node(i)

	def AddKernel(self, creator, i, j, weight=1):
		"""
		Adds a connection between spin i and spin j.  The connection is a weight matrix unto itself,
		with size [qSizes[i], qSizes[j]].  'creator' is a function that accepts a single boolean arguement;
		when true, it returns a string identifier for the weight kernel between spins i and j.  If false, 
		it returns the actual weight kernel.  This is designed to balance ease of use and efficiency:
		if the same weight kernel is used between many pairs of Potts spins, the weight kernel will only be
		generated once (which may not be trivial depending on how it is generated), but the user does not need
		to keep track of how many times a given weight kernel was used.

		:param creator: A lambda function that takes a single boolean arguement.
		:param i: Index of first spin.
		:type i: int
		:param j: Index of second spin.
		:type j: int
		:param weight: Scales the entire weight kernel for this connection only.
		:type weight: float
		:return: None
		"""

		#first, see if kernel has already been created, and if not, create it:
		if weight == 0:
			return
		kName = creator(True)
		if kName not in self.KernelDict:
			self.KernelList.append(creator(False))
			self.KernelDict[kName] = len(self.KernelList)-1
		kIndex = self.KernelDict[kName]

		#add the connection to the graph:
		self.graph.add_edge(i, j, weight=weight, kIndex=kIndex)

	def CompileKernels(self):
		"""
		After all of the connections in a model have been added via AddKernel, the weight kernels are assembled into
		Numpy arrays with specific formats, tailored for use in C++/CUDA sampling code. The results are stored in the Task object.

		The weight kernels are stored in a dense format in a 3-D Numpy array.  The first dimension is the kernel index,
		while the second and third dimensions correspond to the magnetization states of the connected spins.

		A second 3-D NUmpy array, the kernel map, specifies which kernel connects any given pair of spins.  It is stored in
		a sparse format.

		:return: None
		"""

		#dense kernels
		maxQ = numpy.max(self.qSizes)
		nKernels = len(self.KernelList)
		kernels = numpy.zeros([nKernels, maxQ, maxQ], dtype="float16")
		for i, kernel in enumerate(self.KernelList):
			kernels[i, :kernel.shape[0], :kernel.shape[1]] = kernel

		#sparse kmap:
		nPartitions = self.qSizes.shape[0]
		assert nPartitions == self.graph.number_of_nodes()
		maxDensity = numpy.max([self.graph.out_degree(i) for i in range(nPartitions)])
		total_count = 0
		kmap_sparse = numpy.zeros([nPartitions, maxDensity+1,3], dtype="float16")
		for i in range(nPartitions): #this could break if there any completely unconnected nodes in the graph.
			kmap_sparse[i,0,0] = self.graph.out_degree(i)+1 #+1, so that it directly tells the stop index, rather than the number of elements
			for c, (u, v, edge_data) in enumerate(self.graph.out_edges(i, data=True)):
				kmap_sparse[i,c+1,:] = [edge_data['kIndex'], edge_data['weight'], v]
			total_count = total_count + len(self.graph.out_edges(i))
		self.sparse_kmap_density = (total_count/(nPartitions**2))
		# print("Done making sparse kernel map, density = %.3f"%)

		#cast back to float 32.  By starting with float 16 and copying to float32,
		#hopefully the floats will be truncated so as to avoid some cumulative errors in 32 bit FP math
		self.kernels = numpy.zeros([nKernels, maxQ, maxQ], dtype="float32")
		numpy.copyto(self.kernels, kernels)
		self.kmap_sparse = numpy.zeros([nPartitions, maxDensity+1,3], dtype="float32")
		numpy.copyto(self.kmap_sparse, kmap_sparse)
	
	# ====================================================================================
	#
	def EvalCost(self, state):
		"""
		Python-domain calculation of the energy of a state, based on compiled kernels and sparse kernel map.

		:param state: Magnetization values for each of the spins in the model.
		:type state: 1-D numpy array of ints.
		:return: Energy/Cost of the model when in the given state.
		:rtype: float
		"""
		cost = 0
		for i,m in enumerate(state):
			cost = cost + 2*self.biases[i,m]
			for c in range(1, int(self.kmap_sparse[i,0,0])):
				j = int(self.kmap_sparse[i,c,2])
				w = self.kmap_sparse[i,c,1]
				k = int(self.kmap_sparse[i,c,0])
				n = int(state[j])
				cost = cost + w*self.kernels[k,m,n]
		return cost/2

	def EvalPE(self, state, i, m):
		"""
		Calculates the potential energy of state m of node i, given state.  

		:param state: Magnetization values for each of the spins in the model.
		:type state: 1-D numpy array of ints.
		:param i: Spin index of interest.
		:type i: int
		:param m: value of spin i
		:type m: int
		:return: The energy that would potentially be added to the total if spin i was in state m.
		:rtype: float
		"""

		PE = self.biases[i,m]
		for c in range(1, int(self.kmap_sparse[i,0,0])):
			j = int(self.kmap_sparse[i,c,2])
			w = self.kmap_sparse[i,c,1]
			k = int(self.kmap_sparse[i,c,0])
			n = int(state[j])
			PE = PE + w*self.kernels[k,m,n]
		return PE

	def EvalDE(self, state, i, m_new):
		"""
		Calculates how much the total energy would change if spin i had magnetization m_new.

		:param state: Magnetization values for each of the spins in the model.
		:type state: 1-D numpy array of ints.
		:param i: Spin index of interest.
		:type i: int
		:param m_new: new value of spin i
		:type m_new: int
		:return: Change in energy if spin i had magnetization m_new.
		:rtype: float
		"""
		m_old = state[i]
		DE = self.biases[i, m_new]-self.biases[i, m_old]
		for c in range(1, int(self.kmap_sparse[i,0,0])):
			j = int(self.kmap_sparse[i,c,2])
			w = self.kmap_sparse[i,c,1]
			k = int(self.kmap_sparse[i,c,0])
			n = int(state[j])
			DE = DE + w*(self.kernels[k,m_new,n]-self.kernels[k,m_old,n])
		return DE

	def EvalDDE(self, i, m_old, m_new):
		"""
		Calculates the delta-delta-energy of all possible magnetizations of all spins.
		DDE is how much the DE of each action changes, when an action is taken.

		:param i: Index of changing spin that creates the DDE.
		:type i: int
		:param m_old: new value of spin i
		:type m_old: int
		:param m_new: new value of spin i
		:type m_new: int
		:return: DDE value for the all magnetizations of all spins.
		:rtype: 2-D Numpy float array, N_{spins}X N_{magnetizations}
		"""
		DDE = numpy.zeros([self.qSizes.shape[0], numpy.max(self.qSizes)])

		for c in range(1, int(self.kmap_sparse[i,0,0])):
			j = int(self.kmap_sparse[i,c,2])
			w = self.kmap_sparse[i,c,1]
			k = int(self.kmap_sparse[i,c,0])
			DDE[j] = DDE[j] + w*(self.kernels[k,m_new,:]-self.kernels[k,m_old,:])

		return DDE

	# def MakeWeights(self): 
	# 	#actually I think this might not work any more.  Not only that, but I don't need to keep it
	# 	#this function assumes that partition information is already set.
	# 	dim = int(self.qCumulative[-1])
	# 	nUnits = self.qSizes.shape[0]
	# 	weights = numpy.zeros([dim, dim], dtype="float32")
	# 	for i in range(nUnits):
	# 		for kspec in self.kMapLists[i]:
	# 			#start by getting dimensions and positions of the particular kernel:
	# 			j = kspec[2]
	# 			k = kspec[0]
	# 			sz_x = int(self.qSizes[i])
	# 			sz_y = int(self.qSizes[j])
	# 			pos_x = int(self.qCumulative[i]-sz_x)
	# 			pos_y = int(self.qCumulative[j]-sz_y)
	# 			k = self.kernels[k, 0:sz_x, 0:sz_y]
	# 			weights[pos_x:pos_x+sz_x, pos_y:pos_y+sz_y] = k*kspec[1]

	# 	self.weights = weights

	def EnergyBands(self):
		"""
		Calculates the energy of every possible state.  Since the number of states grows eponentially,
		this can only be used with very small problem instances, and has been used just for illustrative purposes.

		:return valid_energies: A list of energies that correspond to valid model states.  Validity is defined by child classes.
		:return invalid_energies: A list of energies that correspond to invalid model states.
		"""
		state = [i for i in range(numpy.max(self.qSizes))]
		valid_energies = []
		invalid_energies = []
		for perm in itertools.product(state, repeat=self.qSizes.shape[0]):
			#just from how this product method works, we need to check to make sure that 
			#the permutation is a valid state, in case the Potts model has unequal qSizes:
			if not numpy.all(perm < self.qSizes):
				continue #skipping this perm since it is a non-existant state
			if self.IsValidSemantics(perm): #IsValidSemantics is defined by the child class
				valid_energies.append(self.EvalCost(perm))
			else:
				invalid_energies.append(self.EvalCost(perm))
		return valid_energies, invalid_energies


	#creates a graphic of the weights and states of the model, with semantic annotations
	qannotations = None
	sannotations = None
	sAnnotatePad = 15
	cellgap=0.5
	partspace = 1 #space from the partition units to the weight matrix
	qfont = matplotlib.font_manager.FontProperties(family='Arial', style='normal', size=8, weight='normal', stretch='normal')
	sfont = matplotlib.font_manager.FontProperties(family='Arial', style='normal', size=14, weight='normal', stretch='normal')
	

	def DisplayModel(self, m1=None, m2=None, state=None, flags=[]):
		#m1 and m2 allow subsets of the model to be plotted (otherwise this might be too big of a plot to see anything)
		if m1 is None:
			m1 = [i for i in range(len(self.qSizes))]
		if m2 is None:
			m2 = [i for i in range(len(self.qSizes))]
		qs1 = [self.qSizes[i] for i in m1]
		qs2 = [self.qSizes[i] for i in m2]
		qc1 = numpy.cumsum(qs1, dtype="int32")
		qc2 = numpy.cumsum(qs2, dtype="int32")

		local_org_x = qc1 - qs1 + numpy.linspace(0, self.cellgap*(len(qs1)-1), len(qs1))
		local_org_y = qc2 - qs2 + numpy.linspace(0, self.cellgap*(len(qs2)-1), len(qs2))

		fig = plt.figure()
		gs = fig.add_gridspec(1, 3)#, hspace=0., wspace=0.3)
		ax1 = fig.add_subplot(gs[0, 0:2])
		ax2 = fig.add_subplot(gs[0, 2])

		plt.sca(ax1)
		ax1 = plt.gca()
		ymax = local_org_y[-1]+qs2[-1]+0.5
		xmax = local_org_x[-1]+qs1[-1]+0.5
		# plt.gca().add_patch(Rectangle((-0.5, -0.5), qc1[-1]+0.5+0.5*(len(qs1)), qc2[-1]+0.5+0.5*(len(qs2)), fc='lightgray', zorder=0))

		#build the weight matrix:
		weights = numpy.zeros([qc1[-1], qc2[-1]], dtype="float32")
		for ii, i in enumerate(m1):
			if not self.graph.has_node(i):# not in self.kMapLists:
				continue
			for u, v, edge_data in self.graph.out_edges(i, data=True):
				
				#start by getting dimensions and positions of the particular kernel:
				j = v
				k = edge_data['kIndex']
				if j not in m2:
					continue
				jj = m2.index(j)
				sz_x = int(self.qSizes[i])
				sz_y = int(self.qSizes[j])
				pos_x = int(qc1[ii]-sz_x)
				pos_y = int(qc2[jj]-sz_y)
				k = self.kernels[k, 0:sz_x, 0:sz_y]
				
				weights[pos_x:pos_x+sz_x, pos_y:pos_y+sz_y] = weights[pos_x:pos_x+sz_x, pos_y:pos_y+sz_y] + k*edge_data['weight']

		#if a state is available, also calculate PE and DE values
		if state is not None:
			pe_vals = numpy.zeros([len(m1), numpy.max(qs1)])
			de_vals = numpy.zeros([len(m1), numpy.max(qs1)])
			for ii, i in enumerate(m1):
				for q in range(qs1[ii]):
					pe_vals[ii, q] = self.EvalPE(state, i, q)
					de_vals[ii, q] = self.EvalDE(state, i, q)

		
		#set up colorbars to indicate weight meaning
		def custom_cmap(positive_color, negative_color=None):
			zero_color = 'lightgrey'
			if negative_color is not None:
				return matplotlib.colors.LinearSegmentedColormap.from_list('NA name', [negative_color, zero_color, positive_color])
			else:
				return matplotlib.colors.LinearSegmentedColormap.from_list('NA name', [zero_color, positive_color])

		cmapWeights = custom_cmap('black')
		normWeights = matplotlib.colors.Normalize(vmin=numpy.min(weights), vmax=numpy.max(weights))
		cmapWeights = matplotlib.cm.ScalarMappable(norm=normWeights, cmap=cmapWeights)

		cmapActiveWeights = custom_cmap('salmon')
		cmapActiveWeights = matplotlib.cm.ScalarMappable(norm=normWeights, cmap=cmapActiveWeights)

		if 'DE' in flags:
			cmapEnergy = custom_cmap('green', 'blue')
			absmax = numpy.max([numpy.max(numpy.abs(pe_vals)), numpy.max(numpy.abs(de_vals))])
			normEnergy = matplotlib.colors.Normalize(vmin=-absmax, vmax=absmax)
		elif 'PE' in flags:
			cmapEnergy = custom_cmap('green')
			normEnergy = matplotlib.colors.Normalize(vmin=-numpy.min(pe_vals), vmax=numpy.max(pe_vals))
			cmapEnergy = matplotlib.cm.ScalarMappable(norm=normEnergy, cmap=cmapEnergy)


		#separate back out into kernels for plotting:
		for ii, i in enumerate(m1):
			for jj, j in enumerate(m2):
				sz_x = int(self.qSizes[i])
				sz_y = int(self.qSizes[j])
				pos_x = int(qc1[ii]-sz_x)
				pos_y = int(qc2[jj]-sz_y)
				k = weights[pos_x:pos_x+sz_x, pos_y:pos_y+sz_y]
				img = cmapWeights.to_rgba(k)
				if state is not None and 'PE' in flags:
					img[:, state[j], :] = cmapActiveWeights.to_rgba(k[:, state[j]])
				elif state is not None and 'no-hl' not in flags:
					img[state[i], state[j], :] = cmapActiveWeights.to_rgba(k[state[i], state[j]])
				img = numpy.transpose(img, axes=[1,0,2])

				x_ofst = local_org_x[ii]
				y_ofst = local_org_y[jj]
				plt.imshow(img, extent=[x_ofst, x_ofst+sz_x, y_ofst+sz_y, y_ofst])

		plt.colorbar(mappable=cmapWeights, ax=ax2, orientation='horizontal', label='weight strength', aspect=10)
		plt.colorbar(mappable=cmapActiveWeights, ax=ax2, orientation='horizontal', label='active weight strength', aspect=10)
		if 'PE' in flags or 'DE' in flags:
			plt.colorbar(mappable=cmapEnergy, ax=ax2, orientation='horizontal', label='partial energy', aspect=10)

		#making the legend.  Oh my god why am I writing so much complicated code just for these stupid plots...
		class HandlerEllipse(HandlerPatch):
			def create_artists(self, legend, orig_handle,
							   xdescent, ydescent, width, height, fontsize, trans):
				center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
				p = Ellipse(xy=center, width=height + xdescent,
									 height=height + ydescent)
				self.update_prop(p, orig_handle, legend)
				p.set_transform(trans)
				return [p]

		state_option_artist = Ellipse((0, 0), 0.7, 0.7, fc='gray')
		state_active_artist = Ellipse((0, 0), 0.7, 0.7, fc='salmon')
		ax2.legend([state_option_artist, state_active_artist], ['Magnetization Option', 'Current Magnetization'], handler_map={Ellipse: HandlerEllipse()})
		ax2.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
		ax2.set_xticks([])
		ax2.set_yticks([])

		#========================================================================================================================
		#start adding partition embelishments.
		topspace = 0
		leftspace = 0

		if 'PE' in flags or 'DE' in flags:
			topspace = topspace + 1
			for ii, i in enumerate(m1):
				for q in range(qs1[ii]):
					flt_indx = local_org_x[ii] + q + 0.5
					plt.arrow(flt_indx, 0, 0, -1, length_includes_head=True, head_width=0.4, head_length=0.4, color='salmon')

		if 'PE' in flags:
			topspace = topspace + 1
			for ii, i in enumerate(m1):
				for q in range(qs1[ii]):
					flt_indx = local_org_x[ii] + q + 0.5
					color = cmapEnergy.to_rgba(pe_vals[i,q])
					plt.gca().add_patch(Rectangle((flt_indx-0.5, -topspace), 1, 1, fc=color))

		if 'DE' in flags:
			topspace = topspace + 1.3
			for ii, i in enumerate(m1):
				for q in range(qs1[ii]):
					flt_indx = local_org_x[ii] + q + 0.5
					color = cmapEnergy.to_rgba(de_vals[i,q])
					plt.gca().add_patch(Rectangle((flt_indx-0.5, -topspace), 1, 1, fc=color))



		if state is not None:
			topspace = topspace + 1
			leftspace = leftspace + 1

			#add circles to indicate state options/actual state.
			for ii, i in enumerate(m2):
				#but first, add a partition group indicator, using a rectangle and circles at the ends:
				center = local_org_y[ii]+0.5*qs2[ii]
				h = qs2[ii]-1
				plt.gca().add_patch(Ellipse((-leftspace, center+h/2), 0.9, 0.9, fc='lightgray'))
				plt.gca().add_patch(Ellipse((-leftspace, center-h/2), 0.9, 0.9, fc='lightgray'))
				plt.gca().add_patch(Rectangle((-leftspace-0.45, center-h/2), 0.9, h, fc='lightgray'))
				for q in range(qs2[ii]):
					flt_indx = local_org_y[ii] + q + 0.5
					if q == state[i]:
						plt.gca().add_patch(Ellipse((-leftspace, flt_indx), 0.7, 0.7, fc='salmon'))
						if 'no-hl' in flags:
							continue
						if 'PE' in flags: 
							#different visual styles to distinguish potential energy diagrams from total energy diagrams
							plt.arrow(-leftspace, flt_indx, leftspace, 0, length_includes_head=True, head_width=0.4, head_length=0.4, color='salmon')
							plt.plot([0, 0, xmax-0.5, xmax-0.5, 0], [flt_indx-0.5, flt_indx+0.5, flt_indx+0.5, flt_indx-0.5, flt_indx-0.5], c='salmon')
						else:
							plt.plot([-leftspace, xmax], [flt_indx, flt_indx], c='salmon')
					else:
						plt.gca().add_patch(Ellipse((-leftspace, flt_indx), 0.7, 0.7, fc='gray'))


			for ii, i in enumerate(m1):
				#but first, add a partition group indicator, using a rectangle and circles at the ends:
				center = local_org_x[ii]+0.5*qs1[ii]
				h = qs1[ii]-1
				plt.gca().add_patch(Ellipse((center+h/2, -topspace), 0.9, 0.9, fc='lightgray'))
				plt.gca().add_patch(Ellipse((center-h/2, -topspace), 0.9, 0.9, fc='lightgray'))
				plt.gca().add_patch(Rectangle((center-h/2, -topspace-0.45), h, 0.9, fc='lightgray'))
				for q in range(qs1[ii]):
					flt_indx = local_org_x[ii] + q + 0.5
					if q == state[i]:
						plt.gca().add_patch(Ellipse((flt_indx, -topspace), 0.7, 0.7, fc='salmon'))
						if 'no-hl' in flags:
							continue
						if 'PE' not in flags:
							plt.plot([flt_indx, flt_indx], [xmax, -topspace], c='salmon')
					else:
						plt.gca().add_patch(Ellipse((flt_indx, -topspace), 0.7, 0.7, fc='gray'))



		if self.qannotations is not None:
			xlbls = []
			xlblpos = []
			for ii, i in enumerate(m1):
				for q in range(qs1[ii]):
					flt_indx = local_org_x[ii] + q + 0.5
					xlblpos.append(flt_indx)
					xlbls.append(self.qannotations[i][q])
			plt.xticks(xlblpos, xlbls)
			
			ylbls = []
			ylblpos = []
			for ii, i in enumerate(m2):
				for q in range(qs2[ii]):
					flt_indx = local_org_y[ii] + q + 0.5
					ylblpos.append(flt_indx)
					ylbls.append(self.qannotations[i][q])
			plt.yticks(ylblpos, ylbls)
		else:
			plt.xticks([])
			plt.yticks([])

		axx2 = ax1.secondary_xaxis('top')
		axy2 = ax1.secondary_yaxis('left')
		if self.sannotations is not None:
			pos = [local_org_x[i] + 0.5*qs1[i] for i in range(len(qs1))]
			ann = [self.sannotations[i] for i in m1]
			axx2.set_xticks(pos, ann)
			pos = [local_org_y[i] + 0.5*qs2[i] for i in range(len(qs2))]
			ann = [self.sannotations[i] for i in m2]
			axy2.set_yticks(pos, ann)
		

		ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
		

		for ax, font in zip([ax1, axx2, axy2], [self.qfont, self.sfont, self.sfont]):
			ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
			for label in ax.get_xticklabels() :
				label.set_fontproperties(font)
			for label in ax.get_yticklabels() :
				label.set_fontproperties(font)

		
		axx2.tick_params(axis='x', pad=self.sAnnotatePad, length=0)
		axy2.tick_params(axis='y', pad=self.sAnnotatePad, length=0)
		plt.setp(axx2.xaxis.get_majorticklabels(), rotation=45, ha="left", rotation_mode="anchor")
		plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="left", rotation_mode="anchor") 
		# for ax in [axx2, axy2]:


		for ax in [ax1, axx2]:
			ax.set_xlim([-leftspace-0.5, xmax])
		for ax in [ax1, axy2]:
			ax.set_ylim([ymax, -topspace-0.5])

		plt.tight_layout()
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

	def PrintCombinatorics(self):
		nstates = 1.
		for q in self.qSizes:
			nstates = nstates*q
		print("Total number of states:", nstates)