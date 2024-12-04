import numpy
import networkx as nx
from matplotlib import pyplot as plt
import PottsPlayground #import includes self. What will happen? No obvious issues.  Maybe the interpreter knows not to make this a circular thing.
from PottsPlayground.Tasks import BaseTask
from PottsPlayground.Tasks import GraphColoringTask
import math
import time
import pickle

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
		if data["d2u"] == False or data["src"] != cell_name:
			continue

		if "ArcsAfter" in data and data["ArcsBefore"] >= ArcsBefore:
			#this arc has already been followed, and we don't need to follow it again:
			if data["ArcsAfter"] > MaxArcsAfter:
				MaxArcsAfter = data["ArcsAfter"]
		else:
			if "ArcsBefore" in data and data["ArcsBefore"] < ArcsBefore or "ArcsBefore" not in data:
				data["ArcsBefore"] = ArcsBefore
			dest_cell = ctx.cells[data["dest"]]
			if 'DFF_ENABLE' not in dest_cell.params:
				# print(dest_cell.name, "does not have a DFF enable option")
				data["ArcsAfter"] = 0
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
	"""
	Potts Model that corresponds to placing logic elements in an Ice40 FPGA.  Ties in with the NextPNR tool;
	is not 100% functional, but can be used to place many FPGA designs.  Supports LUTs, carry chains, and IO.
	"""

	def __init__(self, ctx, cost_balancing=(15, 0.5, 1, 0), split_cells=True, verbose=False):
		"""
		Creates a model of FPGA placement given a NextPNR context.  The context is available when running a Python script inside of
		the NextPNR tool.  The context includes both information about the specific FPGA (availability and locations of physical Basic ELements i.e. BELs)
		and about the design that needs to be placed (logical BELs and their connectivity).  There are several different optimization objectives that
		can be balanced.  The first objective, 'exclusion', is mandatory, since it enforces that multiple logical BELs are not assigned to
		the same physical BEL.  'wirelen' tries to minimize overall distance between connected logical BELs; 'timing' weighs critical path connections
		more to keep the critical path as fast as possible; and 'jogs' tries to assign connected BELs to the same row or column so that 
		only a single horizontal or vertical routing channel is needed to connect them.

		:param ctx: A NextPNR context opbject.
		:param cost_balancing: Specifies relative weight of different optimization objectives. Objectives are (exclusion, wirelen, timing, jogs)
		:type cost_balancing: tuple
		:param split_cells: FPGA LUTs are grouped in blocks of eight; each of the eight have nearly similar connectivity.  If split_cells is True,
		each logical LUT is pre-constrained to one of the eight LUT positions.  This reduces the optimization space for faster results.
		:type split_cells: boolean
		:type verbose: boolean
		"""

		#======================================================construct friendlier formats of the design specifications
		#get lists of bel types in the architecture:
		exclusion_factor = cost_balancing[0]
		w_wirelen = cost_balancing[1]
		w_timing = cost_balancing[2]
		w_jogs = cost_balancing[3]

		self.e_th = -1e14
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
				self.chains[chain[0]] = chain
				
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
			self.GetCellType[chain_start] = 'CC'
			#in the edge connectivity graph, merge all of the chain cells into one:
			for cell_name in chain[1:]:
				#manually merge each cell into the root cell, so that the edge attributes can be correctly accounted for
				for u, v, data in G.edges(cell_name, data=True):
					if u != cell_name:
						print("Guess that's not the right order")
					if v == chain_start:
						continue #don't need self loops
					if data["d2u"] or not G.has_edge(chain_start, v):
						if data['d2u'] and data['src'] == cell_name:
							data['src'] = chain_start
						elif data['d2u'] and data['dest'] == cell_name:
							data['dest'] = chain_start
						G.add_edge(chain_start, v, **data)
						# G[chain_start][v] = data
				G.remove_node(cell_name)

		#build timing arc stuff into the graph:
		for node in G:
			FollowArcs(ctx, G, node)

		self.G = G

		# next, the remaining logic cells need to be colored by what logic cell index (within each tile) they will be constrained to.
		#this should significantly reduce the size of the problem, and speed up computation, without significantly limiting the minimum.
		if split_cells:
			self.SplitCellsBySlot(ctx)
		self.SplitBelsBySlot(ctx)
		self.BelTypes['CC'] = self.BelTypes['LC0'] #artificial mapping, so code below can apply more uniformly
			

		if verbose:
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

		#prevent carry chains that are longer than 1 tile from being placed at the top of the grid
		constrain2lowerY = numpy.zeros([len(self.BelTypes['CC'])])
		ymax = 0
		for i, bel in enumerate(self.BelTypes['CC']):
			loc = ctx.getBelLocation(bel)
			if loc.y > ymax:
				ymax = loc.y
		for i, bel in enumerate(self.BelTypes['CC']):
			loc = ctx.getBelLocation(bel)
			if loc.y == ymax:
				constrain2lowerY[i] = 1e6
		for i, cell_name in enumerate(self.CellTypes["CC"]):
			if len(self.chains[cell_name]) > 1:
				partition = self.CellNames2Partitions[cell_name]
				self.biases[partition, :constrain2lowerY.shape[0]] = constrain2lowerY


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
					nEndpoints = data["w"]
					total_weight_strength[i] = total_weight_strength[i] + w_wirelen/nEndpoints
					d2u = data["d2u"]
					if "ArcsAfter" in data and "ArcsBefore" in data:
						narcs = data["ArcsAfter"]+data["ArcsBefore"]+1
						total_weight_strength[i] = total_weight_strength[i] + w_timing*2**(narcs-maxnarc)
						total_weight_strength[i] = total_weight_strength[i] + w_jogs

		# print(total_weight_strength)

		lc_check_types = ["LC%i"%i for i in range(8)] + ["CC", "ICESTORM_LC"]

		#set up ntiles variables, which tells if a cell occupies 1 or more than 1 logic tiles (if it is a carry chain)
		ntiles = numpy.ones([nPartitions])
		for i, cell_name in enumerate(self.Partitions2CellNames):
			type1 = self.GetCellType[cell_name]
			if type1 == "CC":
				ntiles[i] = math.ceil(len(self.chains[cell_name])/8)

		self.InitKernelManager() #sets up for a host of BaseTask functions used for kernel management

		for i, cell1_name in enumerate(self.Partitions2CellNames):
			for j, cell2_name in enumerate(self.Partitions2CellNames):

				type1 = self.GetCellType[cell1_name]
				type2 = self.GetCellType[cell2_name]
				cell1 = ctx.cells[cell1_name]
				cell2 = ctx.cells[cell2_name]

				#two logic cells with different D flip flop configurations cannot share the same logic tile
				tile_exclude = False
				
				if type1 in lc_check_types and type2 in lc_check_types:
					tile_exclude = DffIncompatible(cell1, cell2)

				#if a cell is a carry chain, it may span across more than one tile,
				#in which case we should have a multi-tile exclusion.
				#we must also account for when two multi-length carry chains exclude each other.
				ntiles1 = ntiles[i]
				ntiles2 = ntiles[j]

				#create a structural penalty to prevent cells from sharing the same FPGA resources
				#(applies to the mutual exclusion of carry chains and logic cells too)
				if (type1 == "CC" and type2 in lc_check_types):
					tile_exclude = True
				if (type2 == "CC" and type1 in lc_check_types):
					tile_exclude = True


				#determine the wiring connectivity between the two cells,
				connected = False
				if G.has_edge(cell1_name, cell2_name):
					connected = True #the heavy lifting and case handling has been done elsewhere, in the construction of G
					data = G[cell1_name][cell2_name]
					nEndpoints = data["w"]
					d2u = data["d2u"]
					if "ArcsAfter" in data and "ArcsBefore" in data:
						narcs = data["ArcsAfter"]+data["ArcsBefore"]+1

				#a geometric mean of how constrained each cell is determines how much constraint weight there should be
				wgt = 1+total_weight_strength[i]*total_weight_strength[j]/(total_weight_strength[i] + total_weight_strength[j])

				# =====================================================main if-else to decide which kernel, if any, relates each pair of cells
				if i==j:
					#the assigned position of one cell should not feed back upon itself
					continue

				if connected and d2u:
					self.AddKernel(lambda n: self.NoJogsKernel(ctx, type1, type2, n), i, j, weight=w_jogs)
					self.AddKernel(lambda n: self.TimingKernel(ctx, type1, type2, n), i, j, weight=w_timing*2**(narcs-maxnarc))

				elif connected:
					self.AddKernel(lambda n: self.WirelenKernel(ctx, type1, type2, n), i, j, weight=w_wirelen/nEndpoints)

				if tile_exclude:
					self.AddKernel(lambda n: self.TileExclusion(ctx, type1, type2, ntiles1, ntiles2, n), i, j, weight=wgt)

				elif type1 == type2:
					self.AddKernel(lambda n: self.BelExclusion(n), i, j, weight=wgt)


		self.CompileKernels() #a baseTask function

	def SplitCellsBySlot(self, ctx):
		random_coloring = False
		colors = []
		if random_coloring:
			for i, cell_name in enumerate(self.CellTypes['ICESTORM_LC']):
				#if a LUT has been constrained, we need to respect that when assigning it to its index:
				cell = ctx.cells[cell_name]
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
			GcTask = GraphColoringTask.GraphColoring(8, G=GcG)
			# GcTask.defaultPwlSchedule()
			GcTask.e_th = 0
			results = PottsPlayground.Anneal(GcTask, GcTask.defaultTemp(niters=10*GcTask.nnodes), OptsPerThrd=100, TakeAllOptions=False, backend="PottsPrecompute", substrate="CPU", nReplicates=1, nWorkers=1, nReports=10)
			colors = results['MinStates'][-1,:]
			#reset the list in self.CellTypes to reflect the ordering that the graph coloring task used:
			print(len(self.CellTypes["ICESTORM_LC"]))
			self.CellTypes["ICESTORM_LC"] = GcTask.Partitions2Nodes
			print(len(self.CellTypes["ICESTORM_LC"]))
			print("First-stage graph coloring completed, with %i conflicting edges remaining"%results["MinEnergies"][-1])
			# print("Exiting on purpose.")
			# exit()

		for i in range(8):
			self.CellTypes["LC%i"%i] = []

		for color, cell_name in zip(colors, self.CellTypes["ICESTORM_LC"]):
			self.CellTypes["LC%i"%color].append(cell_name)
			self.GetCellType[cell_name] = "LC%i"%color

		self.CellTypes["ICESTORM_LC"] = [] #all logic cells have been split into subtypes, so it should be an empty list

	def SplitBelsBySlot(self, ctx):
		#create new BelType lists depending on which of the 8 LCs in each tile each bel is.
		for i in range(8):
			self.BelTypes["LC%i"%i] = []
		for i, bel in enumerate(self.BelTypes['ICESTORM_LC']):
			loc = ctx.getBelLocation(bel)
			self.BelTypes["LC%i"%(loc.z)].append(bel)
			self.GetBelType[bel] = "LC%i"%loc.z

	def SetResultInContext(self, ctx, state):
		"""
		Takes the final state of an annealing run and configures that result into the nextpnr context.
		The context is not returned but afterwards should contain added constraints corresponding to the given state.

		:param ctx: NextPNR context corresponding to the PlacerTask.
		:param state: A Potts model state.
		:type state: 1-D Numpy int array
		:return: Number of conflicts, i.e. number of BELs assigned to incompatible physical locations.  Should be 0 for a successful run.
		:rtype: int
		"""
		nConflicts = 0
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
				chain = [ctx.cells[cn] for cn in self.chains[cell_name]]
				root_bel = self.BelTypes['LC0'][q]
				print("constraining length", len(chain), "carry chain with cell", cell_name,  "rooted at bel", root_bel)
				for j, cell in enumerate(chain):
					LCtype = "LC%i"%(j%8)
					bel = self.BelTypes[LCtype][q+int(j/8)]#move to the next tile for each 8 logic cells
					if bel not in self.used_bels:
						cell.setAttr("BEL", bel)
						# ctx.bindBel(bel, cell, STRENGTH_STRONG)
						self.used_bels.append(bel)
					else:
						print("Error, cell %s cannot be assigned to bel %s which is already assigned"%(cell_name, bel))
						nConflicts = nConflicts + 1
			else: #all other things that are not carry chains
				bel = self.BelTypes[cell_type][q]
				# print(cell.name, bel)
				# if (ctx.checkBelAvail(bel)):
				if bel not in self.used_bels:
					# ctx.bindBel(bel, cell, strength)
					cell.setAttr("BEL", bel)
					# ctx.bindBel(bel, cell, STRENGTH_STRONG)
					self.used_bels.append(bel)
					# cell.attrs["BEL"] = bel
				else:
					print("Error, cell %s cannot be assigned to bel %s which is already assigned"%(cell_name, bel))
					nConflicts = nConflicts + 1

		return nConflicts

	# =============================================================================================== kernel generators
	# =================================================================================================================
	def get_type_locs(self, ctx, type1):
		locs1 = numpy.zeros([len(self.BelTypes[type1]), 2])
		for i, bel1 in enumerate(self.BelTypes[type1]):
			loc1 = ctx.getBelLocation(bel1)
			locs1[i, 0] = loc1.x
			locs1[i, 1] = loc1.y
		return locs1

	def NoJogsKernel(self, ctx, type1, type2, n=False):
		if type1.startswith("LC") or type1 == "CC":
			type1 = "LC0"
		if type2.startswith("LC") or type2 == "CC":
			type2 = "LC0" #has to be set to a specific type, so that it can find the right bel distances

		name = "NoJogs between types " + type1 + " and " + type2

		if n:
			return name
		elif name in self.KernelDict:
			#prevent redundant construction (possibly caused by constructing compound kernels)
			return self.KernelList[self.KernelDict[name]]

		locs1 = self.get_type_locs(ctx, type1)
		locs2 = self.get_type_locs(ctx, type2)
		locs1 = numpy.expand_dims(locs1, 1)
		locs2 = numpy.expand_dims(locs2, 0)

		dx = numpy.abs(locs1[:,:,0]-locs2[:,:,0])
		dy = numpy.abs(locs1[:,:,1]-locs2[:,:,1])

		local_conn = (dx <= 1)*(dy<=1)*1
		dx_conn = (dx > 1)*(dy == 0) * 1
		dy_conn = (dy > 1)*(dx == 0) * 1
		distant_conn = (local_conn == 0)*(dx_conn == 0)*(dy_conn == 0)*(10+(dx+dy))

		return distant_conn

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

		locs1 = self.get_type_locs(ctx, type1)
		locs2 = self.get_type_locs(ctx, type2)
		locs1 = numpy.expand_dims(locs1, 1)
		locs2 = numpy.expand_dims(locs2, 0)

		dx = numpy.abs(locs1[:,:,0]-locs2[:,:,0])
		dy = numpy.abs(locs1[:,:,1]-locs2[:,:,1])

		local_conn = (dx <= 1)*(dy<=1)*0.6
		dx_conn = (dx > 1)*(dy == 0) * (dx + 5)
		dy_conn = (dy > 1)*(dx == 0) * (dy + 5)
		distant_conn = (local_conn == 0)*(dx_conn == 0)*(dy_conn == 0)*3*(3+dx+dy)

		return dx_conn + dy_conn + distant_conn

	def WirelenKernel(self, ctx, type1, type2, n=False):
		#LC and carry chain type all use the same resource type, and have the same kernel size, so they share the same wirelen kernels
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

		locs1 = self.get_type_locs(ctx, type1)
		locs2 = self.get_type_locs(ctx, type2)
		locs1 = numpy.expand_dims(locs1, 1)
		locs2 = numpy.expand_dims(locs2, 0)

		dx = numpy.abs(locs1[:,:,0]-locs2[:,:,0])
		dy = numpy.abs(locs1[:,:,1]-locs2[:,:,1])

		return dx + dy

	def TileExclusion(self, ctx, type1, type2, ntiles2, ntiles1, n=False):
		#LC and carry chain types all use the same resource type, and have the same kernel size, so they share the same wirelen kernels
		if type1.startswith("LC") or type1 == "CC":
			type1 = "LC0"
		if type2.startswith("LC") or type2 == "CC":
			type2 = "LC0" #has to be set to a specific type, so that it can find the right bel distances

		if n:
			return "TileExclusion-%s-%s-%i-%i"%(type1, type2, ntiles1, ntiles2)

		#numpy-vectorized kernel construction for improved speed
		locs1 = self.get_type_locs(ctx, type1)
		locs2 = self.get_type_locs(ctx, type2)
		locs1 = numpy.expand_dims(locs1, 1)
		locs2 = numpy.expand_dims(locs2, 0)

		loc1y = locs1[:,:,1]
		loc1x = locs1[:,:,0]
		loc2y = locs2[:,:,1]
		loc2x = locs2[:,:,0]
		return self.exclusion_factor*(loc1y-ntiles1 < loc2y)*(loc2y < loc1y+ntiles2)*(loc1x == loc2x)

	def BelExclusion(self, n=False):
		if n:
			return "BelExclusion"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax) * self.exclusion_factor


	def examine_path(self, path_cell_names):

		for i in range(1, len(path_cell_names)):
			cell1 = path_cell_names[i-1]
			cell2 = path_cell_names[i]
			if cell1 in self.G and cell2 in self.G:
				print(self.G[cell1][cell2])


if __name__ == '__main__':
	placer = Ice40PlacerTask(ctx, cost_balancing=(15, 0.5, 1, 0.1))
	tstart = time.perf_counter()
	results = Annealing.Anneal(vars(placer), placer.defaultTemp(niters=5e6, tmax=12), OptsPerThrd=1, TakeAllOptions=True, backend="PottsJit", substrate="CPU", nReplicates=1, nWorkers=1, nReports=1)
	ttot = time.perf_counter() - tstart
	print("Annealing time is %.2f seconds"%ttot) 

	placer.SetResultInContext(ctx, results['MinStates'][-1,:])