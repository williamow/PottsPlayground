import numpy
import networkx as nx
from matplotlib import pyplot as plt
from Tasks import BaseTask
from Tasks import GraphColoringTask
import Annealing
import math

def GetCellPortNetName(cell, name):
	if cell.ports[name].net is not None:
		return cell.ports[name].net.name
	else:
		return None

def RecurseDownCarryChain(ctx, cell):
		#recursively finds the next cell in a carry chain, returning a list of all cells downwind of the given cell
		nn = GetCellPortNetName(cell, 'COUT')
		if nn is None:
			return [cell] #last cell in the carry chain
		net = ctx.nets[nn]
		# print(len(net.users))
		# for user in net.users:
			# print(user.cell.name) #ugh. some cell has two users listed, but it's the same cell, twice.
		# assert len(net.users) == 1 #the carry output should only go to a single carry input
		return [cell] + RecurseDownCarryChain(ctx, net.users[0].cell)

def DffIncompatible(cell1, cell2):
	if (cell1.params['DFF_ENABLE']=='1') and (cell2.params['DFF_ENABLE']=='1'):
		if ((GetCellPortNetName(cell1, 'CEN') != GetCellPortNetName(cell2, 'CEN')) or
			(GetCellPortNetName(cell1, 'CLK') != GetCellPortNetName(cell2, 'CLK')) or
			(GetCellPortNetName(cell1, 'SR')  != GetCellPortNetName(cell2, 'SR'))  or
			(cell1.params['NEG_CLK'] != cell2.params['NEG_CLK'])):
				return True
	return False

def FollowArcs(ctx, G, cell_name, ArcsBefore=0, names_before=[]):
	#graph G contains connectivity information.
	#this function adds attributes to the edges of G to approximate timing criticality.
	#ArcsAfter should be the maximum number of Arcs after the Dest node before the next Dff.
	#should be zero if the next node is a Dff.
	#similarly, attribute ArcsBefore keeps track in the other direction.
	#ArcsBefore+ArcsAfter+1 is the total number of routes between two DFFs.
	#names_before is to prevent against infinite recursion.
	if cell_name in names_before:
		print("Hit an infinite recursion! oops!")
		for name in names_before:
			cell = ctx.cells[name]
			print(name, cell.params['DFF_ENABLE'])
			# print
			# for edge in G.edges(name, data=True):
				# print(edge)
		exit()
		return 0
	else:
		names_before.append(cell_name)

	MaxArcsAfter = 0
	for u, v, data in G.edges(cell_name, data=True):
		if data["d2u"] == False:
			continue
		if data["src"] != cell_name:#since the graph edge recall is undirected...
			continue
		if 0: #"ArcsAfter" in data:
			#this arc has already been followed.
			if data["ArcsBefore"] < ArcsBefore:
				data["ArcsBefore"] = ArcsBefore
			if data["ArcsAfter"] > MaxArcsAfter:
				MaxArcsAfter = data["ArcsAfter"]
		else:
			if "ArcsBefore" in data and data["ArcsBefore"] < ArcsBefore or "ArcsBefore" not in data:
				data["ArcsBefore"] = ArcsBefore
			dest_cell = ctx.cells[data["dest"]]
			if 'DFF_ENABLE' not in dest_cell.params:
				# print(dest_cell.name, "does not have a DFF enable option")
				continue #not a logic cell.  Consider as a dead end.
			if dest_cell.params['DFF_ENABLE'] == "1" and GetCellPortNetName(dest_cell, 'COUT') is None:
				#a cell can have a dff enabled, but still have a non-clocked output via cout or lout
				
				data["ArcsAfter"] = 0
			elif dest_cell.params['DFF_ENABLE'] == "1" and GetCellPortNetName(dest_cell, 'COUT') is not None:
				print("Both conditions met!")
			else:
				data["ArcsAfter"] = FollowArcs(ctx, G, data["dest"], ArcsBefore=ArcsBefore+1, names_before=names_before)
				if data["ArcsAfter"] > MaxArcsAfter:
					MaxArcsAfter = data["ArcsAfter"]
	names_before.remove(cell_name)

	return MaxArcsAfter + 1



class Ice40PlacerTask(BaseTask.BaseTask):

	def __init__(self, ctx, exclusion_factor=10):
		#======================================================construct friendlier formats of the design specifications
		#get lists of bel types in the architecture:
		self.exclusion_factor = exclusion_factor
		self.BelTypes = {}
		self.GetBelType = {} #replicate ctx functionality, so that I can easily, temporarily keep track of articial bel type changes
		for bel in ctx.getBels():
			bel_type = ctx.getBelType(bel)
			self.GetBelType[bel] = bel_type
			if bel_type in self.BelTypes:
				self.BelTypes[bel_type].append(bel)
			else:
				self.BelTypes[bel_type] = [bel]

		self.CellTypes = {}
		self.GetCellType = {}
		for key, cell in ctx.cells:
			self.GetCellType[cell.name] = cell.type
			if cell.type in self.CellTypes:
				self.CellTypes[cell.type].append(cell.name)
			else:
				self.CellTypes[cell.type] = [cell.name]

		G = nx.Graph()
		for key, net in ctx.nets:
			if (net.driver.cell is None or net.driver.cell.type == "SB_GB"):
				#net from a global buffer to cells should not contribute in the distance matrix, since a GB distributes a signal evenly
				continue;
			# print(net.name, len(net.users))
			for dest in net.users:
				# if net.driver.cell.name != dest.cell.name: #not sure why I need to exclude like this, but here it is
				G.add_edge(net.driver.cell.name, dest.cell.name, w=len(net.users), d2u=True, src=net.driver.cell.name, dest=dest.cell.name)
				for dest2 in net.users:
					if not G.has_edge(dest2.cell.name, dest.cell.name):
						# print("Edge already exists: ", G[dest2.cell.name][dest.cell.name])
					# else:
						G.add_edge(dest2.cell.name, dest.cell.name, w=len(net.users), d2u=False) #the more users on a net, the less each individual distance is weighed

		

		#======================================================================  split lut cells into subcategories.
		#========================= first, carry chains need to be pulled out and turned into a new type of "cell":


		#Find the start of carry chains and follow them to collect all carry chain cells together
		self.chains = {}
		for i, cell_name in enumerate(self.CellTypes['ICESTORM_LC']):
			cell = ctx.cells[cell_name]
			if GetCellPortNetName(cell, 'COUT') is not None and GetCellPortNetName(cell, 'CIN') is None:
				#then this is the start of a chain!
				chain = RecurseDownCarryChain(ctx, cell)
				# print("Carry chain len =", len(chain))
				chain = [cell.name for cell in chain]
				#split chains into 8-cell max blocks, so that each chain in the SA algorithm only takes up one logic tile:
				# while len(chain) > 8:
				# 	new_chain = chain[:7]
				# 	chain = chain[8:]
				# 	# self.CellTypes['CC'].append(new_chain)
				# 	self.chains[new_chain[0]] = new_chain
				self.chains[chain[0]] = chain
				# self.CellTypes['CC'].append(chain)
				
		#need to remove the chain cells from the list of logic cells.
		self.CellTypes["CC"] = [] #CC, for carry chains
		for chain_start in self.chains:
			chain = self.chains[chain_start]
			for cell_name in chain:
				self.CellTypes['ICESTORM_LC'].remove(cell_name)
			#instead of storing the whole chain, just keep the name of the first cell.
			#Why? by having a name instead of a list, code machinery lower down will still work,
			#and by using the name of the first cell, we can easily find the whole carry chain again later.
			self.CellTypes['CC'].append(chain_start)
			# self.chains[chain[0]] = chain #move actual chain to separate data structure
			self.GetCellType[chain_start] = 'CC'
			#in the edge connectivity graph, merge all of the chain cells into one:
			for cell_name in chain[1:]:
				#manually merge each cell into the root cell
				for u, v, data in G.edges(cell_name, data=True):
					if u != cell_name:
						print("Guess that's not the right order")
					if v == chain_start:
						continue #don't need self loops
					if data["d2u"] or not G.has_edge(chain_start, v):
						G.add_edge(chain_start, v, **data)
						# G[chain_start][v] = data
				G.remove_node(cell_name)
				# G = nx.contracted_nodes(G, chain_start, cell_name, False, False)

		#build timing arc stuff into the graph:
		for node in G:
			FollowArcs(ctx, G, node)

		# print(G["$nextpnr_ICESTORM_LC_0"]["$abc$13313$auto$blifparse.cc:492:parse_blif$13641_LC"])
		# exit()

		# next, the remaining logic cells need to be colored by what logic cell index (within each tile) they will be constrained to.
		#this should significantly reduce the size of the problem, and speed up computation, without significantly limiting the minimum.
		random_coloring = False
		colors = []
		if random_coloring:
			for i, cell_name in enumerate(self.CellTypes['ICESTORM_LC']):
				#if a LUT has been constrained, we need to respect that when assigning it to its index:
				if "BEL" in cell.attrs:
					bel = cell.attrs["BEL"]
					loc = ctx.getBelLocation(bel)
					colors.append(loc.z)
					# new_cell_type = "LC%i"%loc.z
				else:
					colors.append(i%8)
					# new_cell_type = "LC%i"%(i%8)
		else:
			#use graph coloring to assign each cell to a z index.
			#cells that are connected should have different colors, so that they have a chance to live in the same tile.
			#build NX graph of the free logic cells only, for coloring.
			GcG = nx.Graph()
			GcG.add_nodes_from(self.CellTypes["ICESTORM_LC"])
			for key, net in ctx.nets:
				if (net.driver.cell is None or net.driver.cell.type != "ICESTORM_LC" or net.driver.cell.name not in self.CellTypes["ICESTORM_LC"]):
					#net from a global buffer to cells should not contribute in the distance matrix, since a GB distributes a signal evenly
					continue;
				lc_users = []
				for dest in net.users:
					if dest.cell.name in self.CellTypes["ICESTORM_LC"]:
						lc_users.append(dest)
				for dest in lc_users:
					GcG.add_edge(net.driver.cell.name, dest.cell.name)
					#also add edges between the destinations, since destinations that are closer together can use some of the same wire routing:
					for dest2 in lc_users:
						GcG.add_edge(dest2.cell.name, dest.cell.name)

			#now this is fun.  We get to use the graph coloring task to solve this little problem so that we can solve the placement task better
			GcTask = GraphColoringTask.GraphColoring(GcG, 8)
			GcTask.defaultPwlSchedule(niters=100000)
			GcTask.e_th = 0
			results = Annealing.Anneal(GcTask, GcTask.PwlTemp, OptsPerThrd=10, TakeAllOptions=False, backend="PottsJit", substrate="GPU", nReplicates=1, nWorkers=16, nReports=5)
			colors = results['MinStates'][-1,:]
			#reset the list in self.CellTypes to reflect the ordering that the graph coloring task used:
			print(len(self.CellTypes["ICESTORM_LC"]))
			self.CellTypes["ICESTORM_LC"] = GcTask.Partitions2Nodes
			print(len(self.CellTypes["ICESTORM_LC"]))
			print("First-stage graph coloring completed, with %i conflicting edges remaining"%results["MinEnergies"][-1])


		for i in range(8):
			self.CellTypes["LC%i"%i] = []

		for color, cell_name in zip(colors, self.CellTypes["ICESTORM_LC"]):
			self.CellTypes["LC%i"%color].append(cell_name)
			self.GetCellType[cell_name] = "LC%i"%color


		self.CellTypes["ICESTORM_LC"] = [] #all logic cells have been split into subtypes, so it should be an empty list

		# also artificially partition bels into their categories:
		for i in range(8):
			self.BelTypes["LC%i"%i] = []
		for i, bel in enumerate(self.BelTypes['ICESTORM_LC']):
			loc = ctx.getBelLocation(bel)
			self.BelTypes["LC%i"%(loc.z)].append(bel)
			self.GetBelType[bel] = "LC%i"%loc.z
		self.BelTypes["ICESTORM_LC"] = []
		self.BelTypes['CC'] = self.BelTypes['LC0'] #artificial mapping, so code below can apply more uniformly

		for ty in self.CellTypes:
			print(ty, len(self.CellTypes[ty]))

		
		#======================================================= partition construction
		qSizes = []
		self.Partitions2CellNames = []
		self.CellNames2Partitions = {}
		for CellType in self.CellTypes:
			qSize = len(self.BelTypes[CellType])
			for cell_name in self.CellTypes[CellType]:
				self.Partitions2CellNames.append(cell_name)
				self.CellNames2Partitions[cell_name] = len(self.Partitions2CellNames) - 1
				qSizes.append(qSize)
				# print(CellType, cell_name)
		nPartitions = len(qSizes)
		self.SetPartitions(numpy.array(qSizes))

		#iterate through cells again, assigning biases to indicate that a cell has previously been assigned a fixed position:
		self.used_bels = []
		self.biases = numpy.zeros([nPartitions, numpy.max(qSizes)], dtype="float32")
		for i, cell_name in enumerate(self.Partitions2CellNames):
			cell = ctx.cells[cell_name]
			cell_type = self.GetCellType[cell_name]
			if "BEL" in cell.attrs:
				bel = cell.attrs["BEL"]
				self.used_bels.append(bel)
				bel_list = self.BelTypes[cell_type]
				bel_position = bel_list.index(bel)
				self.biases[i,:] = 100 #all non-selected positions get a large penalty
				self.biases[i, bel_position] = 0

		#assign biases to some global buffer cells,
		#since there are limits to which global buffers can be used for reset and enable signals:
		def drivesPins(gb_cell, pin_name):
			users = cell.ports["GLOBAL_BUFFER_OUTPUT"].net.users
			user_ports = [user.port for user in users]
			return pin_name in user_ports

		#make bias vectors to constrain a global buffer cell to either even or odd global buffer bels
		constrain2even = numpy.zeros([len(self.BelTypes["SB_GB"])]) + 1e6
		constrain2odd = numpy.zeros([len(self.BelTypes["SB_GB"])]) + 1e6
		for i, bel in enumerate(self.BelTypes["SB_GB"]):
			wire_name = ctx.getBelPinWire(bel, "GLOBAL_BUFFER_OUTPUT")#[2];
			# print(wire_name) 
			netnum = int(wire_name[-1])
			if netnum%2 == 0:
				constrain2even[i] = 0
			else:
				constrain2odd[i] = 0

		for i, cell_name in enumerate(self.CellTypes["SB_GB"]):
			cell = ctx.cells[cell_name]
			if drivesPins(cell, "SR"): #SR might not be the right keyword here.
				print("this is a reset signal")
			if drivesPins(cell, "CEN"):
				# print("this is a enable signal")
				partition = self.CellNames2Partitions[cell_name]
				self.biases[partition, :constrain2odd.shape[0]] = constrain2odd

			# print("")
			# for thing in dir(cell):
				# print(thing)
			# for k, v in cell.ports:
				# print(k, v)
				# for thing in v.net.users:
					# print(thing.port)
					# for thing2 in dir(thing.port):
						# print(thing2)
			# for k, v in cell.attrs:
				# print(k, v)

		# exit()

		
		# ============================================================================= kernel map construction

		self.InitKernelManager() #sets up for a host of BaseTask functions used for kernel management
		self.kmap = numpy.zeros([nPartitions, nPartitions], dtype="int32")

		narchist = numpy.zeros([25])
		maxnarc = 1
		for u, v, data in G.edges(data=True):
			if "ArcsAfter" in data and "ArcsBefore" in data:
				narcs = data["ArcsAfter"]+data["ArcsBefore"]+1
				narchist[narcs] = narchist[narcs] + 1
				if narcs > maxnarc:
					maxnarc = narcs
		# print(narchist)
		print("max number arcs between dffs:", maxnarc)


		total_weight_strength = numpy.zeros([nPartitions])
		for i, cell1_name in enumerate(self.Partitions2CellNames):
			for j, cell2_name in enumerate(self.Partitions2CellNames):
				if G.has_edge(cell1_name, cell2_name):
					data = G[cell1_name][cell2_name]
					w = data["w"]
					total_weight_strength[i] = total_weight_strength[i] + 1/w
					d2u = data["d2u"]
					if "ArcsAfter" in data and "ArcsBefore" in data:
						narcs = data["ArcsAfter"]+data["ArcsBefore"]+1
						total_weight_strength[i] = total_weight_strength[i] + 1.5**(narcs-maxnarc)
		# print(total_weight_strength)

		for i, cell1_name in enumerate(self.Partitions2CellNames):
			for j, cell2_name in enumerate(self.Partitions2CellNames):

				type1 = self.GetCellType[cell1_name]
				type2 = self.GetCellType[cell2_name]
				cell1 = ctx.cells[cell1_name]
				cell2 = ctx.cells[cell2_name]

				#two logic cells with different D flip flop configurations cannot share the same logic tile
				dff_incompatible = False
				dff_check_types = ["LC%i"%i for i in range(8)] + ["CC"]
				if type1.startswith("LC") and type2.startswith("LC") and DffIncompatible(cell1, cell2):
					dff_incompatible = True

				#determine if there should be a structural penalty for having cell1 and cell2 in the same slot:

				#if a cell is a carry chain, it may span across more than one tile,
				#in which case we should have a multi-tile exclusion.
				#we must also account for when two multi-length carry chains exclude each other.
				ntiles1 = 1
				if type1 == "CC":
					ntiles1 = math.ceil(len(self.chains[cell1_name])/8)

				ntiles2 = 1
				if type2 == "CC":
					ntiles2 = math.ceil(len(self.chains[cell2_name])/8)

				#create a structural penalty to prevent cells from sharing the same FPGA resources
				#(applies to the mutual exclusion of carry chains and logic cells too)
				simple_exclude = False
				if (type1 == type2 or dff_incompatible):
					simple_exclude = True
				if (type1 == "CC" and type2.startswith("LC")):
					simple_exclude = True
				if (type2 == "CC" and type1.startswith("LC")):
					simple_exclude = True


				#determine the wiring connectivity between the two cells,
				connected = False
				if G.has_edge(cell1_name, cell2_name):
					connected = True #the heavy lifting and case handling has been done elsewhere, in the construction of G
					data = G[cell1_name][cell2_name]
					w = data["w"]
					d2u = data["d2u"]
					if "ArcsAfter" in data and "ArcsBefore" in data:
						narcs = data["ArcsAfter"]+data["ArcsBefore"]+1

				#a geometric mean of how constrained each cell is determines how much constraint weight there should be
				wgt = 1+total_weight_strength[i]*total_weight_strength[j]/(total_weight_strength[i] + total_weight_strength[j])

				# =====================================================main if-else to decide which kernel, if any, relates each pair of cells
				if i==j:
					#the assigned position of one cell should not feed back upon itself
					continue

				if connected:
					self.AddKernel(lambda n: self.WirelenKernel(ctx, type1, type2, n), i, j, weight=1/w)

				if connected and d2u:
					self.AddKernel(lambda n: self.TimingKernel(ctx, type1, type2, n), i, j, weight=1.5**(narcs-maxnarc))

				if simple_exclude and ntiles1 == 1 and ntiles2 == 1:
					self.AddKernel(lambda n: self.IdentityKernel(n), i, j, weight=wgt)

				elif simple_exclude:
					self.AddKernel(lambda n: self.MultiTileExclusion(ctx, ntiles1, ntiles2, n), i, j, weight=wgt)

		self.CompileKernels() #a baseTask function


	def SetResultInContext(self, ctx, state, strength):
		#takes the final state of an annealing run and configures that result into the nextpnr context.

		for i, q in enumerate(state):
			# if i > 5:
				# return
			cell_name = self.Partitions2CellNames[i]
			cell = ctx.cells[cell_name]
			if "BEL" in cell.attrs:
				#this cell was previously constrained, so we don't set it:
				continue
			cell_type = self.GetCellType[cell_name]
			if cell_type == "CC":
				print("constraining a carry chain in context")
				chain = [ctx.cells[cn] for cn in self.chains[cell_name]]
				for j, cell in enumerate(chain):
					LCtype = "LC%i"%(j%8)
					bel = self.BelTypes[LCtype][q+int(j/8)]#move to the next tile for each 8 logic cells
					if bel not in self.used_bels:
						cell.setAttr("BEL", bel)
						self.used_bels.append(bel)
					else:
						print("Error, cell %s cannot be assigned to bel %s which is already assigned"%(cell_name, bel))
			else: #all other things that are not carry chains
				bel = self.BelTypes[cell_type][q]
				# print(cell.name, bel)
				# if (ctx.checkBelAvail(bel)):
				if bel not in self.used_bels:
					# ctx.bindBel(bel, cell, strength)
					cell.setAttr("BEL", bel)
					self.used_bels.append(bel)
					# cell.attrs["BEL"] = bel
				else:
					print("Error, cell %s cannot be assigned to bel %s which is already assigned"%(cell_name, bel))
					# return


	# =============================================================================================== kernel generators
	# =================================================================================================================
	def TimingKernel(self, ctx, type1, type2, n=False):
		#LC and carry chain types all use the same resource type, and have the same kernel size, so they share the same wirelen kernels
		if type1.startswith("LC") or type1 == "CC":
			type1 = "LC0"
		if type2.startswith("LC") or type2 == "CC":
			type2 = "LC0" #has to be set to a specific type, so that it can find the right bel distances

		name = "Timing between types " + type1 + " and " + type2

		if n:
			return name
		elif name in self.KernelDict:
			#prevent redundant construction (possibly caused by constructing compound kernels)
			return self.KernelList[self.KernelDict[name]]

		kernel = numpy.zeros([len(self.BelTypes[type1]), len(self.BelTypes[type2])])
		for i, bel1 in enumerate(self.BelTypes[type1]):
			for j, bel2 in enumerate(self.BelTypes[type2]):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				dx = abs(loc1.x - loc2.x)
				dy = abs(loc1.y - loc2.y)
				#the cost between a driver and a user is different than the cost between two users.

				if dx <= 1 and dy <= 1:
					kernel[i, j] = 0 #can't do better than this, so why assign a penalty to it?
				elif dx > 0 and dy == 0:
					kernel[i, j] = 4*math.ceil(dx/4)**2 #this "2" is a minor heuristic.  Milage may vary.
				elif dy > 0 and dx == 0:
					kernel[i, j] = 4*math.ceil(dy/4)**2
				else:
					kernel[i, j] = 100 #having to change both row and column is expensive, and should(?) be easy to optimize away

		return kernel


	def WirelenKernel(self, ctx, type1, type2, n=False):
		#LC and carry chain types all use the same resource type, and have the same kernel size, so they share the same wirelen kernels
		if type1.startswith("LC") or type1 == "CC":
			type1 = "LC0"
		if type2.startswith("LC") or type2 == "CC":
			type2 = "LC0" #has to be set to a specific type, so that it can find the right bel distances

		name = "Wirelength between types " + type1 + " and " + type2

		if n:
			return name
		elif name in self.KernelDict:
			#prevent redundant construction (possibly caused by constructing compound kernels)
			return self.KernelList[self.KernelDict[name]]

		kernel = numpy.zeros([len(self.BelTypes[type1]), len(self.BelTypes[type2])])
		for i, bel1 in enumerate(self.BelTypes[type1]):
			for j, bel2 in enumerate(self.BelTypes[type2]):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				dx = abs(loc1.x - loc2.x)
				dy = abs(loc1.y - loc2.y)

				if dx > 0: #wirelen cost designed so that sticking to the same row or column is desirable 
					kernel[i, j] = kernel[i, j] + (dx + 2) #this "2" is a minor heuristic.  Milage may vary.
				if dy > 0: #horizontal runs only connect on one side, so dy should be 0 to get shared benefits .. (? am I making this up?)
					kernel[i, j] = kernel[i, j] + (dy + 2)

		return kernel


	def MultiTileExclusion(self, ctx, ntiles1, ntiles2, n=False):
		if n:
			return "MultiTileExclusion-%i-%i"%(ntiles1, ntiles2)

		bels = self.BelTypes["LC0"]
		kernel = numpy.zeros([len(bels), len(bels)])
		for i, bel1 in enumerate(bels):
			for j, bel2 in enumerate(bels):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				kernel[i, j] = self.exclusion_factor*(loc1.y-ntiles1 < loc2.y and loc2.y < loc1.y+ntiles2)

		return kernel


	def IdentityKernel(self, n=False):
		if n:
			return "IdentityKernel"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax) * self.exclusion_factor

	def defaultPwlSchedule(self, niters, tmax=5):
		PwlTemp = numpy.zeros([2, 2], dtype="float32")
		PwlTemp[0,0] = tmax
		PwlTemp[0,1] = 0.2
		PwlTemp[1,0] = 0
		PwlTemp[1,1] = niters
		self.PwlTemp = PwlTemp