import numpy
import networkx as nx
from matplotlib import pyplot as plt
import PottsPlayground #import includes self. What will happen? No obvious issues.  Maybe the interpreter knows not to make this a circular thing.
from PottsPlayground.Tasks import BaseTask
from PottsPlayground.Tasks.GraphColoring import GraphColoring
import math
import time
import pickle
import copy
from collections import defaultdict

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
		return [cell] + RecurseDownCarryChain(ctx, net.users[0].cell)

# def DffFlavor(cell):

LcInpCounts = {}
# DffFlavors = set()
# FlavorNums = {}
def DffIncompatible(cell1, cell2):
	#the DFFs in each tile share several inputs, which must be the same within each tile:
	# for cell in [cell1, cell2]:
		# if cell.name not in DffFlavors:
	if (cell1.params['DFF_ENABLE']=='1') and (cell2.params['DFF_ENABLE']=='1'):
		if ((GetCellPortNetName(cell1, 'CEN') != GetCellPortNetName(cell2, 'CEN')) or
			(GetCellPortNetName(cell1, 'CLK') != GetCellPortNetName(cell2, 'CLK')) or
			(GetCellPortNetName(cell1, 'SR')  != GetCellPortNetName(cell2, 'SR'))  or
			(cell1.params['NEG_CLK'] != cell2.params['NEG_CLK'])):
				return True
	
	#there is also a limit that no more than 32 local input signals can be used across all 8 luts.
	#I can't enforce that directly with pairwise constraints,
	#but if we ensure that each pair has no more than 8 in total,
	#the 32-input limit will be met.
	for cell in [cell1, cell2]:
		if cell.name not in LcInpCounts:
			count = 0
			for name, port in cell.ports:
				if port.net is None:
					continue
				if port.net.driver.cell.name == cell.name:
					continue #i.e. ignore outputs
				if port.net.driver.cell.type == 'SB_GB':
					continue #global signals don't count towards the total
				count = count + 1
			LcInpCounts[cell.name] = count
	
	if LcInpCounts[cell1.name] + LcInpCounts[cell2.name] > 8:
		return True

	return False

def old_timing_paths(ctx):
	LC_in_ports = ["I0", "I1", "I2", "I3", "CIN"]

	#first, create directed graph where each node is a net,
	#and edges are non-dff cells between nets.
	G = nx.DiGraph()
	for key, net in ctx.nets:
		G.add_node(net.name, depth=1)

	for key, cell in ctx.cells:
		if cell.type != "ICESTORM_LC":
			continue #timing only for logic cells

		input_net_names = [(port, GetCellPortNetName(cell, port)) for port in LC_in_ports]
		input_net_names = [info for info in input_net_names if info[1] != None]

		output_net_names = []
		if cell.params['DFF_ENABLE'] == "0" and GetCellPortNetName(cell, 'O') is not None:
			output_net_names.append(('O', GetCellPortNetName(cell, 'O')))
		if GetCellPortNetName(cell, 'COUT') is not None:
			output_net_names.append(('COUT', GetCellPortNetName(cell, 'COUT')))

		for inp in input_net_names:
			for out in output_net_names:
				w = 0.1 if (inp[0] == 'CIN' and out == 'COUT') else 1.
				#carry connections are automatically very fast,
				#so should not be fully counted towards timing arc length
				G.add_edge(inp[1], out[1], weight=w)

	#delete two special nets that nextpnr uses:
	for special_node in ["$PACKER_VCC_NET", "$PACKER_GND_NET"]:
		if special_node in G:
			G.remove_node(special_node)

	# print(sorted(nx.simple_cycles(G)))
	assert len(sorted(nx.simple_cycles(G))) == 0

	#to figure out how important each net is to timing, traverse the graph
	#forwards and backwards, adding up the lengths each way.
	#after adding up, nodes in G_forward will know how the maximum depth
	#of nets before them, and G_reverse will know the maximum depth after.
	#adding the two gives the length of the longest pathway that passes through each net.
	#To make sure that each net really knows its maximum before or after length,
	#graph edges are deleted as they are followed and edges are only followed once all the 
	#preceding edges have been processed and deleted.
	G_forward = copy.deepcopy(G)
	G_reverse = copy.deepcopy(G)

	while (G_forward.number_of_edges() > 0):
		# print(G_forward.number_of_edges())
		for net in G_forward.nodes:
			if len(G_forward.in_edges(net)) == 0:
				to_remove = []
				for u, v, edge_data in G_forward.out_edges(net, data=True):
					G_forward.nodes[v]['depth'] = max(G_forward.nodes[v]['depth'], G_forward.nodes[u]['depth']+edge_data['weight'])
					# print(G.nodes[v]['depth'], G.nodes[u]['depth'], edge_data['weight'])
					to_remove.append((u, v))
				G_forward.remove_edges_from(to_remove)

	while (G_reverse.number_of_edges() > 0):
		# print(G_reverse.number_of_edges(), G_reverse.number_of_nodes())
		for net in G_reverse.nodes:
			if len(G_reverse.out_edges(net)) == 0:
				to_remove = []
				for u, v, edge_data in G_reverse.in_edges(net, data=True):
					G_reverse.nodes[u]['depth'] = max(G_reverse.nodes[u]['depth'], G_reverse.nodes[v]['depth']+edge_data['weight'])
					to_remove.append((u, v))
				G_reverse.remove_edges_from(to_remove)

	#combine forward and reverse counts:
	arc_lengths = defaultdict(lambda:1)
	for net in G.nodes:
		# print(G_reverse[net], G_forward[net])
		combined = G_reverse.nodes[net]['depth'] + G_forward.nodes[net]['depth'] - 1
		arc_lengths[net] = combined

	return arc_lengths

def timing_paths(ctx):
	LC_in_ports = ["I0", "I1", "I2", "I3", "CIN"]
	LC_out_ports = ["COUT", "LO"]
	ignore_nets = ["$PACKER_VCC_NET", "$PACKER_GND_NET"] #two special nets that nextpnr uses

	#first, create directed graph where each node is an arc,
	#and edges are non-dff cells between arcs.
	G = nx.DiGraph()
	arc_lookup = defaultdict(list)
	for key, net in ctx.nets:
		if net in ignore_nets or net.driver.cell is None:
			continue

		for user in net.users:
			arc = (net.driver.cell.name, user.cell.name)
			G.add_node(arc, depth=1)
			arc_lookup[net.name].append(arc)
			
	
		
	for key, cell in ctx.cells:
		if cell.type != "ICESTORM_LC":
			continue #timing only for logic cells

		input_net_names = [(port, GetCellPortNetName(cell, port)) for port in LC_in_ports]
		input_net_names = [info for info in input_net_names if info[1] != None]

		output_net_names = [(port, GetCellPortNetName(cell, port)) for port in LC_out_ports]
		output_net_names = [info for info in output_net_names if info[1] != None]

		if cell.params['DFF_ENABLE'] == "0" and GetCellPortNetName(cell, 'O') is not None:
			output_net_names.append(('O', GetCellPortNetName(cell, 'O')))


		for inp in input_net_names:
			for out in output_net_names:
				if inp[1] in ignore_nets or out[1] in ignore_nets:
					continue
				w = 0.1 if (inp[0] == 'CIN' and out[0] == 'COUT') else 1.
				#carry connections are automatically very fast,
				#so should not be fully counted towards timing arc length
				for inp_arc in arc_lookup[inp[1]]:
					for out_arc in arc_lookup[out[1]]:
						G.add_edge(inp_arc, out_arc, weight=w)


	for cycle in nx.simple_cycles(G):
		print("Error, there is as least one combinatorial loop.")
		print("Arcs in loop:", cycle)
		#This might arise from packing LUTs into carry chains.  
		#Since two logical cells effectively occupy one phyiscal cell,
		#it can sometimes look as though a logical pathway feeds back onto itself,
		#when in fact the signal is going to the other logical cell located in the same physical cell.
		exit()

	assert len(sorted(nx.simple_cycles(G))) == 0

	#to figure out how important each net is to timing, traverse the graph
	#forwards and backwards, adding up the lengths each way.
	#after adding up, nodes in G_forward will know how the maximum depth
	#of nets before them, and G_reverse will know the maximum depth after.
	#adding the two gives the length of the longest pathway that passes through each net.
	#To make sure that each net really knows its maximum before or after length,
	#graph edges are deleted as they are followed and edges are only followed once all the 
	#preceding edges have been processed and deleted.
	G_forward = copy.deepcopy(G)
	G_reverse = copy.deepcopy(G)

	while (G_forward.number_of_edges() > 0):
		# print(G_forward.number_of_edges())
		for arc in G_forward.nodes:
			if len(G_forward.in_edges(arc)) == 0:
				to_remove = []
				for u, v, edge_data in G_forward.out_edges(arc, data=True):
					G_forward.nodes[v]['depth'] = max(G_forward.nodes[v]['depth'], G_forward.nodes[u]['depth']+edge_data['weight'])
					# print(G.nodes[v]['depth'], G.nodes[u]['depth'], edge_data['weight'])
					to_remove.append((u, v))
				G_forward.remove_edges_from(to_remove)

	while (G_reverse.number_of_edges() > 0):
		# print(G_reverse.number_of_edges(), G_reverse.number_of_nodes())
		for arc in G_reverse.nodes:
			if len(G_reverse.out_edges(arc)) == 0:
				to_remove = []
				for u, v, edge_data in G_reverse.in_edges(arc, data=True):
					G_reverse.nodes[u]['depth'] = max(G_reverse.nodes[u]['depth'], G_reverse.nodes[v]['depth']+edge_data['weight'])
					to_remove.append((u, v))
				G_reverse.remove_edges_from(to_remove)

	#combine forward and reverse counts:
	arc_lengths = defaultdict(lambda:1)
	for arc in G.nodes:
		# print(G_reverse[net], G_forward[net])
		combined = G_reverse.nodes[arc]['depth'] + G_forward.nodes[arc]['depth'] - 1
		arc_lengths[arc] = combined

	return arc_lengths

def label_carry_chains(ctx):
	cc_lookup = {} #lookup has a flat structure, i.e. all chain cells at top level regardless of what chain they belong to
	for key, cell in ctx.cells:
		if cell.type == "ICESTORM_LC" and GetCellPortNetName(cell, 'COUT') is not None and GetCellPortNetName(cell, 'CIN') is None:
			#then this is the start of a chain!
			chain = RecurseDownCarryChain(ctx, cell)
			for i, chain_cell in enumerate(chain):
				#re-type cells to indicate they are in a carry chain, and if they are root or not.
				#also store carry chain cells into a dict, for looking up their info more easily
				if i == 0:
					chain_cell.type = "CC"
					cc_lookup[chain_cell.name] = len(chain) #for knowing how many tiles will be used
				else:
					chain_cell.type = "CCchild"
					cc_lookup[chain_cell.name] = (chain[0], i) #for extracting root cell placement after annealing
	return cc_lookup

def label_lc_subtypes(ctx):
	for i, (key, cell) in enumerate(ctx.cells):
		if cell.type != "ICESTORM_LC":
			continue
		if "BEL" in cell.attrs:
			bel = cell.attrs["BEL"]
			loc = ctx.getBelLocation(bel)
			# cell.setAttr("type", "LC%i"%loc.z)
			cell.type = "LC%i"%loc.z
		else:
			# cell.setAttr("type", "LC%i"%(i%8))
			cell.type = "LC%i"%(i%8)
		# print(cell.type)

def gather_bel_types(ctx, restrict):
	BelTypes = defaultdict(list)
	
	if type(restrict) == tuple or type(restrict) == list:
		xlim = restrict[0]
		ylim = restrict[1]
	elif restrict is not None:
		xlim = restrict
		ylim = restrict

	for bel in ctx.getBels():
		bel_type = ctx.getBelType(bel)
		if restrict is not None:
			loc = ctx.getBelLocation(bel)
			if loc.x > xlim or loc.y > ylim:
				continue
		BelTypes[bel_type].append(bel)

	for i, bel in enumerate(BelTypes['ICESTORM_LC']):
		loc = ctx.getBelLocation(bel)
		BelTypes["LC%i"%(loc.z)].append(bel)

	BelTypes['CC'] = BelTypes['LC0'] #artificial mapping, so code below can apply more uniformly

	# print(BelTypes)
	return BelTypes


class Ice40Placer(BaseTask.BaseTask):
	"""
	Potts Model that corresponds to placing logic elements in an Ice40 FPGA.  Ties in with the NextPNR tool;
	is not 100% functional, but can be used to place many FPGA designs.  Supports LUTs, carry chains, and IO.
	"""

	def __init__(self, ctx, cost_balancing=(15, 0.5, 1, 0), split_cells=True, verbose=False, restrict=None):
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
		:param restrict: Limits placement of logic cells to tiles with x < restrict, y < restrict.
		:type restrict: int
		"""

		#======================================================construct friendlier formats of the design specifications
		#get lists of bel types in the architecture:
		BaseTask.BaseTask.__init__(self)
		exclusion_factor = cost_balancing[0]
		w_wirelen = cost_balancing[1]
		w_timing = cost_balancing[2]
		w_jogs = cost_balancing[3]

		arc_timing_info = timing_paths(ctx)
		# arc_lengths = [value for key, value in arc_timing_info.items()]
		# print("arc lengths", numpy.unique(arc_lengths, return_counts=True))
		# longest_arc = numpy.max(arc_lengths)

		# for arc in arc_timing_info:
		# 	print(arc, arc_timing_info[arc])
		# exit()
		# print("Longest arc", longest_arc)

		self.cc_lookup = label_carry_chains(ctx)


		#filter timing arcs - remove mentions of carry chain child cells
		for arc in list(arc_timing_info.keys()):
			src_cell = arc[0]
			if ctx.cells[src_cell].type == "CCchild":
				src_cell = self.cc_lookup[src_cell][0].name

			dest_cell = arc[1]
			if ctx.cells[dest_cell].type == "CCchild":
				dest_cell = self.cc_lookup[dest_cell][0].name

			if src_cell == dest_cell:
				del arc_timing_info[arc]

			elif src_cell != arc[0] or dest_cell != arc[1]:
				arc_timing_info[(src_cell, dest_cell)] = arc_timing_info[arc]
				del arc_timing_info[arc]


		
		#this always also generates split LC categories, but we don't always use them
		self.BelTypes = gather_bel_types(ctx, restrict)


		if split_cells:
			label_lc_subtypes(ctx)
			del self.BelTypes['ICESTORM_LC']

		
			

		#subdivide global buffer bels based on even-oddness, since they have slightly different features
		wire_names = [ctx.getBelPinWire(bel, "GLOBAL_BUFFER_OUTPUT") for bel in self.BelTypes["SB_GB"]]
		net_nums = [int(wire_name[-1]) for wire_name in wire_names]
		even_global_buffers = [i for i, netnum in enumerate(net_nums) if netnum%2 == 0]
		odd_global_buffers =  [i for i, netnum in enumerate(net_nums) if netnum%2 == 1]

		#if an IO cell is unset, only use index zero cells (they come in pairs):
		usable_SB_IO = [i for i, sb_io_bel in enumerate(self.BelTypes['SB_IO']) if sb_io_bel.endswith("io0")]

		#get a sub-list of logic blocks that are not at the top,
		#i.e. ones that are not limited to length-8 carry chains
		bel_loc = [ctx.getBelLocation(bel) for bel in self.BelTypes['CC']]
		bel_y = [loc.y for loc in bel_loc]
		ymax = numpy.max(bel_y)
		constrain2lowerY = [i for i, y in enumerate(bel_y) if y != ymax]
		
		#======================================================= partition construction
		# self.used_bels = []
		for key, cell in ctx.cells:
			if cell.type == "CCchild":
				continue #these are not explicitly included in the optimization

			self.AddSpins([len(self.BelTypes[cell.type])], [cell.name], [self.BelTypes[cell.type]])

			# not all global buffers can drive all signal types.
			# Must restrict certain global buffer cells to a subset of physical global buffers:
			if cell.type == "SB_GB":
				users = cell.ports["GLOBAL_BUFFER_OUTPUT"].net.users
				user_ports = [user.port for user in users]
				if "CEN" in user_ports:
					self.PinSpin(cell.name, odd_global_buffers)
				elif "SR" in user_ports:
					self.PinSpin(cell.name, even_global_buffers)

			if cell.type == "CC":
				self.PinSpin(cell.name, constrain2lowerY)

			if cell.type == "SB_IO":
				self.PinSpin(cell.name, usable_SB_IO)

			#if a cell has already been fixed for some reason (like IO),
			#fix it in the Potts model here too.
			#do it last, since PinSpin is an override
			if "BEL" in cell.attrs:
				bel = cell.attrs["BEL"]
				print("constraining", cell.name, bel)
				bel_position = self.BelTypes[cell.type].index(bel)
				self.PinSpin(cell.name, [bel_position])

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

		#total_weight_strength keeps track of optimization weights,
		#so legalization constraints can be scaled appropriately on a per-cell basis
		total_weight_strength = defaultdict(lambda: 1) 
		maxnarc = numpy.max([v for k, v in arc_timing_info.items()])
		for arc in arc_timing_info:
			arc_weights = w_wirelen + w_timing*(arc_timing_info[arc]/maxnarc)**3
			type1 = ctx.cells[arc[0]].type
			type2 = ctx.cells[arc[1]].type
			kernel = lambda n: self.TimingKernel(ctx, type1, type2, n)
			self.AddWeight(kernel, arc[0], arc[1], weight=arc_weights)
			total_weight_strength[arc[0]] = total_weight_strength[arc[0]] + arc_weights
			total_weight_strength[arc[1]] = total_weight_strength[arc[1]] + arc_weights


		lc_check_types = ["LC%i"%i for i in range(8)] + ["CC", "ICESTORM_LC"]

		#set up ntiles variables, which tells if a cell occupies 1 or more than 1 logic tiles (if it is a carry chain)
		ntiles = {}
		for key, cell in ctx.cells:
			if cell.type == "CC":
				ntiles[cell.name] = math.ceil(self.cc_lookup[cell.name]/8)
			else:
				ntiles[cell.name] = 1

		for key1, cell1 in ctx.cells:
			if cell1.type == "CCchild":
				continue #these are not explicitly included in the optimization
			for key2, cell2 in ctx.cells:
				if cell2.type == "CCchild":
					continue #these are not explicitly included in the optimization
				if cell1.name == cell2.name:
					continue

				#there are two types of hard constraints, bel exclusion and tile exclusion.
				#Bel exclusion means that two cells have the same type, and therefore cannot
				#be in the same place.  Simple.
				#Tile exclusion means that two cells cannot even be in the same tile,
				#due to various factors.  A carry chain in one tile might even exclude
				#other cells in the tile above it, if the chain is longer than 8.

				
				tile_exclude = False
				
				#two logic cells with different D flip flop configurations cannot share the same logic tile
				if cell1.type in lc_check_types and cell2.type in lc_check_types:
					tile_exclude = DffIncompatible(cell1, cell2)
					# print(tile_exclude)

				#if a cell is a carry chain, it may span across more than one tile,
				#in which case we should have a multi-tile exclusion.
				#we must also account for when two multi-length carry chains exclude each other.
				ntiles1 = ntiles[cell1.name]
				ntiles2 = ntiles[cell2.name]

				#create a structural penalty to prevent cells from sharing the same FPGA resources
				#(applies to the mutual exclusion of carry chains and logic cells too)
				if (cell1.type == "CC" and cell2.type in lc_check_types):
					tile_exclude = True
				if (cell2.type == "CC" and cell1.type in lc_check_types):
					tile_exclude = True

				#a geometric mean of each cell's optimization weights determines how much constraint weight there should be
				wgt = total_weight_strength[cell1.name]*total_weight_strength[cell2.name]
				wgt = wgt/(total_weight_strength[cell1.name] + total_weight_strength[cell2.name])
				wgt = wgt*exclusion_factor

				if tile_exclude:
					self.AddKernel(lambda n: self.TileExclusion(ctx, cell1.type, cell2.type, ntiles1, ntiles2, n), cell1.name, cell2.name, weight=wgt)

				elif cell1.type == cell2.type:
					self.AddKernel(lambda n: self.BelExclusion(n), cell1.name, cell2.name, weight=wgt)

		

		#now loop through nets, to add net constraints.
				#determine the wiring connectivity between the two cells,
				# connected = False
				# if G.has_edge(cell1_name, cell2_name):
				# 	connected = True #the heavy lifting and case handling has been done elsewhere, in the construction of G
				# 	data = G[cell1_name][cell2_name]
				# 	nEndpoints = data["w"]
				# 	d2u = data["d2u"]
				# 	if "ArcsAfter" in data and "ArcsBefore" in data:
				# 		narcs = data["ArcsAfter"]+data["ArcsBefore"]+1

				# #a geometric mean of how constrained each cell is determines how much constraint weight there should be
				# wgt = 1+total_weight_strength[i]*total_weight_strength[j]/(total_weight_strength[i] + total_weight_strength[j])

				# # =====================================================main if-else to decide which kernel, if any, relates each pair of cells
				# if i==j:
				# 	#the assigned position of one cell should not feed back upon itself
				# 	continue

				# if connected and d2u:
				# 	self.AddKernel(lambda n: self.NoJogsKernel(ctx, type1, type2, n), i, j, weight=w_jogs)
				# 	self.AddKernel(lambda n: self.TimingKernel(ctx, type1, type2, n), i, j, weight=w_timing*2**(narcs-maxnarc))

				# elif connected:
				# 	self.AddKernel(lambda n: self.WirelenKernel(ctx, type1, type2, n), i, j, weight=w_wirelen/nEndpoints)


		self.CompileKernels()


	def RevertContext(self, ctx):
		"""
		Since the task makes some 
		revert altered types, needs to be called after all placer task is finished and results have been set
		"""
		for key, cell in ctx.cells:
			if cell.type.startswith("CC") or cell.type.startswith("LC"):
				# cell.setAttr("type", "ICESTORM_LC")
				cell.type = "ICESTORM_LC"
			# print(cell.type)

	

	def SetResultInContext(self, ctx, state, verbose=False):
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
		used_bels = {}
		for key, cell in ctx.cells:
			if cell.type == "SB_IO":
				continue #if IO is unconstrained, I can't tell if the pin exists or not from python

			if "BEL" in cell.attrs:
				used_bels[cell.bel] = cell
				continue #this cell was previously constrained, so we don't set it

			if cell.type == "CCchild":
				root_cell, i = self.cc_lookup[cell.name]
				root_bel_indx = self.GetSpinFromState(root_cell.name, state)
				LCtype = "LC%i"%(i%8)
				bel = self.BelTypes[LCtype][root_bel_indx+int(i/8)]#move to the next tile for each 8 logic cells

			else:
				bel_indx = self.GetSpinFromState(cell.name, state)
				bel = self.BelTypes[cell.type][bel_indx]

			if bel not in used_bels:
				cell.setAttr("BEL", bel)
				used_bels[bel] = cell
				if verbose:
					print("cell %s <=> bel %s"%(cell.name, bel))
			else:
				print("ERROR: BEL assignment conflict.  cell %s (type: %s) cannot use bel %s becuase it is already assigned to cell %s (type: %s)"%(cell.name, cell.type, bel, used_bels[bel].name, used_bels[bel].type))
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

		local_conn = (dx <= 1)*(dy<=1)*0.6 #not actually added into the kernel
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
		return (loc1y-ntiles1 < loc2y)*(loc2y < loc1y+ntiles2)*(loc1x == loc2x)

	def BelExclusion(self, n=False):
		if n:
			return "BelExclusion"

		qMax = numpy.max([len(self.BelTypes[bels]) for bels in self.BelTypes])
		return numpy.eye(qMax)


	def examine_path(self, path_cell_names):

		for i in range(1, len(path_cell_names)):
			cell1 = path_cell_names[i-1]
			cell2 = path_cell_names[i]
			if cell1 in self.G and cell2 in self.G:
				print(self.G[cell1][cell2])


if __name__ == '__main__':
	#alternate version that makes direct edits to the ctx instead of creating a parallel accounting structure:
	for key, cell in ctx.cells:
		if GetCellPortNetName(cell, 'COUT') is not None and GetCellPortNetName(cell, 'CIN') is None:
			#then this is the start of a chain!
			chain = RecurseDownCarryChain(ctx, cell)
			[chain_cell.setAttr("type", "CC") for chain_cell in chain] #re-type cells to indicate they are in a carry chain
			ctx.setAttr("type", "CCroot") #re-type root cells

	exit()

	placer = Ice40Placer(ctx, cost_balancing=(15, 0.5, 1, 0.1))
	tstart = time.perf_counter()
	results = Annealing.Anneal(vars(placer), placer.defaultTemp(niters=5e6, tmax=12), OptsPerThrd=1, TakeAllOptions=True, backend="PottsJit", substrate="CPU", nReplicates=1, nWorkers=1, nReports=1)
	ttot = time.perf_counter() - tstart
	print("Annealing time is %.2f seconds"%ttot) 

	placer.SetResultInContext(ctx, results['MinStates'][-1,:])