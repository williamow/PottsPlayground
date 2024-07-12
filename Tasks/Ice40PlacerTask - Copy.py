import numpy
import networkx as nx
from matplotlib import pyplot as plt
from Tasks import BaseTask

def GetCellPortNetName(cell, name):
	if cell.ports[name].net is not None:
		return cell.ports[name].net.name
	else:
		return None

class Ice40PlacerTask(BaseTask.BaseTask):

	def __init__(self, ctx, exclusion_factor=10):
		#======================================================construct friendlier formats of the design specifications
		#get lists of bel types in the architecture:
		self.exclusion_factor = exclusion_factor
		self.BelTypes = {}
		for bel in ctx.getBels():
			# if ctx.getBelHidden(bel): #not sure about the importance of this check, but the C code version has it
				# continue
			bel_type = ctx.getBelType(bel)
			# if bel_type == "SB_IO":
				# print(dir(bel))
			if bel_type in self.BelTypes:
				self.BelTypes[bel_type].append(bel)
			else:
				self.BelTypes[bel_type] = [bel]

		#get cells in the design:
		# self.CellTypes = {}
		# for key, cell in ctx.cells:
		# 	if cell.type in self.CellTypes:
		# 		self.CellTypes[cell.type].append(cell)
		# 	else:
		# 		self.CellTypes[cell.type] = [cell]
			


		#=======================================================kernel construction - depends only on FPGA architecture
		# self.NamedKernels = {}
		
		# for type1 in self.BelTypes:
		# 	for type2 in self.BelTypes:
		# 		kname = type1 + "-" + type2 + "-"
		# 		print("Generating kernel", kname, end="\r")
		# 		self.NamedKernels[kname+"wirelen"] = self.WirelenKernel(ctx, type1, type2)
		# 		if type1 == type2:
		# 			exclusion_weights = numpy.eye(len(self.BelTypes[type1]))*exclusion_factor
		# 			self.NamedKernels[kname+"wirelen_exclusion"] = self.NamedKernels[kname+"wirelen"] + exclusion_weights
		# 			self.NamedKernels[kname+"exclusion"] = exclusion_weights
		# 			if type1 == "ICESTORM_LC" or type1 == "SB_IO":
		# 				blk_ex_w = self.CompatibilityKernel(ctx, type1)*exclusion_factor
		# 				self.NamedKernels[kname+"wirelen_block_exclusion"] = self.NamedKernels[kname+"wirelen"] + blk_ex_w
		# 				self.NamedKernels[kname+"block_exclusion"] = blk_ex_w

		

		#======================================================= kernel map construction - topology depends on logic design
		qSizes = []
		self.Partitions2Cells = []
		self.CellNames2Partitions = {}
		for key, cell in ctx.cells:
			if cell.type == "SB_GB":
				continue #try ignoring these
			self.Partitions2Cells.append(cell)
			self.CellNames2Partitions[cell.name] = len(self.Partitions2Cells) - 1
			qSizes.append(len(self.BelTypes[cell.type]))
		nPartitions = len(qSizes)
		self.SetPartitions(numpy.array(qSizes))

		self.kmap = numpy.zeros([nPartitions, nPartitions], dtype="int32")

		# self.NamedKernels2MatrixKernels()

		#build a graph of the design connectivity:
		G = nx.Graph()
		for key, net in ctx.nets:
			if (net.driver.cell is None):
				continue;
			for dest in net.users:
				G.add_edge(net.driver.cell.name, dest.cell.name, net=net.name)
		self.G = G

		#iterate through cells again, assigning biases to indicate that a cell has previously been assigned a fixed position:
		self.used_bels = []
		for i, cell1 in enumerate(self.Partitions2Cells):
			if "BEL" in cell1.attrs:
				bel = cell1.attrs["BEL"]
				self.used_bels.append(bel)
				bel_list = self.BelTypes[cell1.type]
				bel_position = bel_list.index(bel)
				self.biases[i, bel_position] = -100 #set as a very low energy to fix the value in this position

		#also create and assign negative biases to constrain cells that are part of carry chains to particular locations.

		#create templates for constraining to a particular z loc:
		cc_biases = numpy.zeros([8, len(self.BelTypes["ICESTORM_LC"])])
		for i, bel in enumerate(self.BelTypes["ICESTORM_LC"]):
			loc = ctx.getBelLocation(bel)
			cc_biases[loc.z, i] = -100
		# cc_biases = cc_biases*exclusion_factor
		# print(cc_biases[0,:])
		# print(cc_biases[1,:])

		#pre-build edge lookup table:
		edge_lookup = {}
		for u, v, data in G.edges(data=True):
			edge_lookup[data.get("net")] = (u, v)

		def follow_carry_chain_step(cell_index):
			cell = self.Partitions2Cells[cell_index]
			nn = GetCellPortNetName(cell, 'COUT')
			if nn is None:
				return -1 #indicate that the chain is over
			cell_nodes = edge_lookup[nn]
			assert cell.name in cell_nodes
			assert len(cell_nodes) == 2
			if cell_nodes[0] != cell.name:
				next_cell = cell_nodes[0]
			else:
				next_cell = cell_nodes[1]
			return self.CellNames2Partitions[next_cell]


		#this is harder.  Find the start of carry chains and follow them biasing them to constrain them to increasing locations
		for i, cell in enumerate(self.Partitions2Cells):
			# print("cell %i with name %s"%(i, cell.name))
			if cell.type == "ICESTORM_LC" and GetCellPortNetName(cell, 'COUT') is not None and GetCellPortNetName(cell, 'CIN') is None:
				#then this is the start of a chain!

				# print("Carry chain:")
				lut_color_counter = 0
				cell_index = i
				chain = []
				while (follow_carry_chain_step(cell_index) >= 0):
					# print("cell %i, name %s"%(cell_index, self.Partitions2Cells[cell_index].name))
					chain.append(self.Partitions2Cells[cell_index].name) #make a list of all the cells in the chain
					self.biases[cell_index, :] = cc_biases[lut_color_counter, :]
					lut_color_counter = lut_color_counter + 1
					cell_index = follow_carry_chain_step(cell_index)
				
				#add false edges between all the nodes in the chain, so that the chains are coerced into staying together better:
				for name1 in chain:
					for name2 in chain:
						if name1 != name2:
							G.add_edge(name1, name2, net="not a real net")


		#first, fill with constraint kernels:
		self.InitKernelManager() #sets up for a host of BaseTask functions used for kernel management
		for i, cell1 in enumerate(self.Partitions2Cells):
			for j, cell2 in enumerate(self.Partitions2Cells):

				if i==j:
					#the assigned position of one cell should not feed back upon itself
					self.kmap[i,j] = self.KernelIndex(lambda n: self.NullKernel(n))

				elif cell1.type == "ICESTORM_LC" and cell2.type == "ICESTORM_LC":
					self.kmap[i,j] = self.LcKernelOptions(ctx, cell1, cell2)

				elif cell1.type == "SB_IO" and cell2.type == "SB_IO":
					self.kmap[i,j] = self.KernelIndex(lambda n: self.NullKernel(n))
					#assumes that all IOs are constrained, therefore constraints between them are not needed

				elif G.has_edge(cell1.name, cell2.name):
					#any connected types not captured in the above special cases.
					#Either same type or different type connected.
					self.kmap[i,j] = self.KernelIndex(lambda n: self.WirelenKernel(ctx, cell1.type, cell2.type, n))

				elif cell1.type == cell2.type:
					#unconnected, same type
					self.kmap[i,j] = self.KernelIndex(lambda n: self.IdentityKernel(n))

				else:
					#cells are of different type and unconnected, therefore they do not affect each other
					self.kmap[i,j] = self.KernelIndex(lambda n: self.NullKernel(n))

		self.CompileKernels() #a baseTask function
				# if G.has_edge(cell1.name, cell2.name) and cell1.type == cell2.type:
					# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "wirelen_exclusion"]
					# if (cell1.type == "ICESTORM_LC" and self.AreLcsIncompatible(cell1, cell2)):
						# print("Found incompatible LCs")
						# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "wirelen_block_exclusion"]
					# if (cell1.type == "SB_IO" and self.AreIoIncompatible(cell1, cell2)):
						# print("Found incompatible LCs")
						# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "wirelen_block_exclusion"]


				# elif G.has_edge(cell1.name, cell2.name):
					# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "wirelen"]

				# elif cell1.type == cell2.type:
					# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "exclusion"]
					# if (cell1.type == "ICESTORM_LC" and self.AreLcsIncompatible(cell1, cell2)):
						# print("Found incompatible LCs")
						# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "block_exclusion"]
					# if (cell1.type == "SB_IO" and self.AreIoIncompatible(cell1, cell2)):
						# print("Found incompatible LCs")
						# self.kmap[i,j] = self.Knames2Numbers[cell1.type + "-" + cell2.type + "-" + "block_exclusion"]



		# for i in range(nPartitions):
			# print(self.biases[i,:])
		#hmm. instead of this, could break bels into subtypes, where carry chain subtypes are constrained to their carry chain index.
		#i.e. the 3rd element in a carry chain would be constrained to LUT2 only, etc. reducing the option count of each carry chain cell by a factor of 8.
		#even better: could randomly assign all cells to only live in a particular LUT slot.  Over larger designs, this probably wouldn't make a very big difference
		#in performance, since each lut in a tile otherwise faces the same constraints.
		#oh, but: Although the placement of a particular LUT in a particular slot does not matter, if two related lut (oh I see where this is going... graph coloring bwahahaha)

		#overall, cut down options by factor of 8.  Not bad. And, the connections between logic cells becomes much more sparse (1 in 8, instead of 1 in 1).
			# for thing in dir(cell1):
			# 	print(thing)
			# for k, v in cell1.ports:
			# 	print(k, v)
			# for k, v in cell1.attrs:
			# 	print(k, v)


		



	

	def LcKernelOptions(self, ctx, cell1, cell2):
		#determines the relationship between two logic cells, 
		#and returns the index of the kernel that should be used

		#first check: if the dff configuration of cells is not identical, the cells cannot exist on the same logic tile
		dff_incompatible = False
		if (cell1.params['DFF_ENABLE']=='1') and (cell2.params['DFF_ENABLE']=='1'):
			if ((GetCellPortNetName(cell1, 'CEN') != GetCellPortNetName(cell2, 'CEN')) or
				(GetCellPortNetName(cell1, 'CLK') != GetCellPortNetName(cell2, 'CLK')) or
				(GetCellPortNetName(cell1, 'SR')  != GetCellPortNetName(cell2, 'SR'))  or
				(cell1.params['NEG_CLK'] != cell2.params['NEG_CLK'])):
					dff_incompatible = True

		#determine their wiring connectivity.  Can be unconnected, connected, or carry_connected.
		#carry_connected is directional, so there are two flavors.
		if GetCellPortNetName(cell1, 'COUT') is not None and GetCellPortNetName(cell1, 'COUT') == GetCellPortNetName(cell2, 'CIN'):
			return self.KernelIndex(lambda n: self.CarryKernelAtoB(ctx, n))

		elif GetCellPortNetName(cell1, 'CIN')  is not None and GetCellPortNetName(cell1, 'CIN')  == GetCellPortNetName(cell2, 'COUT'):
			return self.KernelIndex(lambda n: self.CarryKernelBtoA(ctx, n))

		elif self.G.has_edge(cell1.name, cell2.name) and dff_incompatible:
			return self.KernelIndex(lambda n: self.TileIncompatibleWirelenKernel(ctx, cell1.type, n))

		elif self.G.has_edge(cell1.name, cell2.name):
			return self.KernelIndex(lambda n: self.WirelenKernel(ctx, cell1.type, cell1.type, n))

		elif dff_incompatible:
			return self.KernelIndex(lambda n: self.TileIncompatibleKernel(ctx, cell1.type, n))

		else:
			return self.KernelIndex(lambda n: self.IdentityKernel(n))
		


	def AreIoIncompatible(self, cell1, cell2):
		return False
		# for thing in dir(cell1):
			# print(thing)
		# for k, v in cell1.ports:
			# print(k, v)
		# for k, v in cell1.attrs:
			# print(k, v)

		# for k, v in cell1.attrs["PACKAGE_PIN"]:
			# print(k, v)

		a = cell1.attrs
		b = cell2.attrs
		#individually, if output clock, input clock, and clock en are used in two io cells,
		#they must be the same in order to share a tile
		to_check = ["OUTPUT_CLK", "INPUT_CLK", "CLOCK_ENABLE"]
		for tc in to_check:
			if tc in a and tc in b:
				if a[tc] != None and b[tc] != None and a[tc] != b[tc]:
					return True

		return False


	def SetResultInContext(self, ctx, state, strength):
		#takes the final state of an annealing run and configures that result into the nextpnr context.

		for i, q in enumerate(state):
			# if i > 5:
				# return
			cell = self.Partitions2Cells[i]
			if "BEL" in cell.attrs:
				#this cell was previously constrained, so we don't set it:
				continue
			bel = self.BelTypes[cell.type][q]
			# print(cell.name, bel)
			# if (ctx.checkBelAvail(bel)):
			if bel not in self.used_bels:
				# ctx.bindBel(bel, cell, strength)
				cell.setAttr("BEL", bel)
				self.used_bels.append(bel)
				# cell.attrs["BEL"] = bel
			else:
				print("Error, bel %s assigned to multiple cells"%bel)
				# return


	# =============================================================================================== kernel generators
	# =================================================================================================================
	def WirelenKernel(self, ctx, type1, type2, n=False):
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
				kernel[i, j] = abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)

		if type1 == type2:
			return kernel + self.IdentityKernel()
		else:
			return kernel

	def TileIncompatibleKernel(self, ctx, bel_type, n=False):
		if n:
			return "Tile incompatible. Type=" + bel_type

		bels = self.BelTypes[bel_type]
		kernel = numpy.zeros([len(bels), len(bels)])
		for i, bel1 in enumerate(bels):
			for j, bel2 in enumerate(bels):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				kernel[i, j] = (loc1.x == loc2.x)*(loc1.y == loc2.y) #one, if they are on the same x/y site, 0 otherwise

		return kernel*self.exclusion_factor

	def TileIncompatibleWirelenKernel(self, ctx, beltype, n):
		if n:
			return "tile incompatible with wirelength. Type=" + beltype

		return self.TileIncompatibleKernel(ctx, beltype) + self.WirelenKernel(ctx, beltype, beltype)

	def IdentityKernel(self, n=False):
		if n:
			return "IdentityKernel"

		qMax = numpy.max(self.qSizes)
		return numpy.eye(qMax) * self.exclusion_factor


	def CarryKernelAtoB(self, ctx, n=False):
		if n:
			return "Carry chain legalizer flavor AtoB"
		
		#If cell A sends a carry signal to cell B, then cell B should only be in the next LUT over.
		#also, if A is in a LUT7 position, B can be in the LUT0 position of the next Tile up.
		kernel = numpy.zeros([len(self.BelTypes["ICESTORM_LC"]), len(self.BelTypes["ICESTORM_LC"])])
		for i, bel1 in enumerate(self.BelTypes["ICESTORM_LC"]):
			for j, bel2 in enumerate(self.BelTypes["ICESTORM_LC"]):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				if ((loc1.x == loc2.x and loc1.y == loc2.y and loc1.z+1 == loc2.z) or
					(loc1.x == loc2.x and loc1.y+1 == loc2.y and loc1.z == 7 and loc2.z == 0)):
						kernel[i,j] = -1 #negative, since we want to encourage this relationship

		return kernel*self.exclusion_factor


	def CarryKernelBtoA(self, ctx, n=False):
		if n:
			return "Carry chain legalizer flavor BtoA"
		
		#If cell A sends a carry signal to cell B, then cell B should only be in the next LUT over.
		#also, if A is in a LUT7 position, B can be in the LUT0 position of the next Tile up.
		kernel = numpy.zeros([len(self.BelTypes["ICESTORM_LC"]), len(self.BelTypes["ICESTORM_LC"])])
		for i, bel1 in enumerate(self.BelTypes["ICESTORM_LC"]):
			for j, bel2 in enumerate(self.BelTypes["ICESTORM_LC"]):
				loc1 = ctx.getBelLocation(bel1)
				loc2 = ctx.getBelLocation(bel2)
				if ((loc1.x == loc2.x and loc1.y == loc2.y and loc1.z == loc2.z+1) or
					(loc1.x == loc2.x and loc1.y == loc2.y+1 and loc1.z == 0 and loc2.z == 7)):
						kernel[i,j] = -1 #negative, since we want to encourage this relationship

		return kernel*self.exclusion_factor



# If a master 4-wire SPI interface could be bridged to one or more slave 3-wire SPI devices using only passive electronic components, what is the minimum number of passive components it would take?  Assume ideal circumstances, i.e. no need to worry about glitches, the master and slave devices all work at the same voltage, and the SPI master can be configured to turn the clock on or off when not communicating, and can set MOSI to high or low when not communicating.

# A SPI controller set to make data transmissions in mode 3 is connected to an I2C bus with only I2C targets (No I2C controller devices).  The SPI clock directly drives the I2C clock, while SPI chip select, SPI MOSI, and Vdd are all connected to the I2C data line through identical resistors.  Considering SPI CS and MOSI as inputs and I2C SDA as the output, what logical function are the resistors intended to create, and what state should MOSI idle in for the SPI master to successfully create an I2C start condition?