import numpy
import networkx as nx
from matplotlib import pyplot as plt
from PottsPlayground.Tasks import BaseTask
import blifparser.blifparser as blifparser
import itertools
import math
import copy

def BlifTruth2PottsTruth(blifTruth):
	ninputs = len(blifTruth[0])-1
	blifCases = [[int(v) for v in row[:-1]] for row in blifTruth] #input cases that result in a true output
	# print(blifCases)

	#start out representing the truth table as a high-dimension matrix:
	tt = numpy.zeros([2]*ninputs)
	for case in blifCases:
		tt[tuple(case)] = 1

	#also create array of unique ids for each entry:
	ids = numpy.linspace(1, 2**ninputs, 2**ninputs, dtype=int)
	ids = numpy.reshape(ids, [2]*ninputs)

	covers = {}

	#now, cover as much of the truth table with don't cares as possible: 
	for i, row in enumerate(itertools.product([0, 1, float('nan')], repeat=ninputs)):
		sl = tuple([v if not math.isnan(v) else numpy.s_[:] for v in row])
		# print(sl)
		# print(row)
		ttsl = tt[sl].flatten()

		if not numpy.all(ttsl == ttsl[0]):
			continue #this cover does not work

		covered_ids = set(ids[sl].flatten())
		#if the current set is a subset of a previous set, throw it out:
		old_ii = []
		for ii in covers:
			old_ii.append(ii)
			if covered_ids.issubset(covers[ii][0]):
				continue

		#likewise, if the new cover subsumes prior covers, throw out the old ones:
		for ii in old_ii:
			if covers[ii][0].issubset(covered_ids):
				del covers[ii]

		rowl = [thing for thing in row]

		rowl.append(ttsl[0])
		covers[i] = (covered_ids, numpy.array(rowl))

	#re-make truth table in 2-D format, with don't cares represetned as NaNs:
	tt = []
	for i in covers:
		# print(covers[i])
		tt.append(covers[i][1])
	return numpy.array(tt)


def Table2HighDim(Table, nets, outputs):
	#converts a table of allowed states into an n+1 dimensional format, where there are n inputs and one or more outputs
	ninputs = len(nets)-len(outputs)
	noutputs = len(outputs)
	input_slice = tuple([i for i, net in enumerate(nets) if net not in outputs])
	output_slice = tuple([i for i, net in enumerate(nets) if net in outputs])
	hypertable = numpy.zeros([2]*ninputs+[noutputs])
	for i in range(Table.shape[0]):
		inputs = Table[i,input_slice]
		output_values = Table[i,output_slice]
		sl = tuple([int(v) if not math.isnan(v) else numpy.s_[:] for v in inputs]+[numpy.s_[:]])
		# print(inputs)
		# print(input_slice)
		# print(sl)
		# print(ninputs)
		# print(noutputs)
		# print(nets)
		# print(outputs)
		# print(hypertable.shape)
		# print((numpy.s_[:], numpy.s_[:]))
		hypertable[sl] = output_values

	#also return re-ordered net list:
	nets = [nets[i] for i in input_slice] + [nets[i] for i in output_slice]
	return hypertable, nets

def SimplifyStates(hypertable):
	#also create array of unique ids for each entry:
	ninputs = len(hypertable.shape)-1
	noutputs = hypertable.shape[ninputs]
	ids = numpy.linspace(1, 2**ninputs, 2**ninputs, dtype=int)
	ids = numpy.reshape(ids, [2]*ninputs)

	covers = {}

	#now, cover as much of the truth table with don't cares as possible: 
	for i, row in enumerate(itertools.product([0, 1, float('nan')], repeat=ninputs)):
		sl = tuple([int(v) if not math.isnan(v) else numpy.s_[:] for v in row] + [numpy.s_[:]])
		# print(sl)
		# print(row)
		ttsl = hypertable[sl].reshape(-1, noutputs)

		
		# print(ttsl)
		if not numpy.all(ttsl == ttsl[0,:]):
			continue #this cover does not work
		# print(ttsl)
		# exit()
		sl_id = tuple([int(v) if not math.isnan(v) else numpy.s_[:] for v in row])
		covered_ids = set(ids[sl_id].flatten())
		#if the current set is a subset of a previous set, throw it out:
		old_ii = []
		for ii in covers:
			old_ii.append(ii)
			if covered_ids.issubset(covers[ii][0]):
				continue

		#likewise, if the new cover subsumes prior covers, throw out the old ones:
		for ii in old_ii:
			if covers[ii][0].issubset(covered_ids):
				del covers[ii]

		rowl = [thing for thing in row]

		for outval in ttsl[0,:]:
			rowl.append(outval)
		covers[i] = (covered_ids, numpy.array(rowl))

	#re-make truth table in 2-D format, with don't cares represetned as NaNs:
	tt = []
	for i in covers:
		# print(covers[i])
		tt.append(covers[i][1])
	# exit()
	return numpy.array(tt)


def MergeStateTables(Table1, Nets1, Table2, Nets2):

	cols2merge = []
	for net in Nets1:
		if net in Nets2:
			cols2merge.append(net)

	#generate all cross-combinations:
	Table1 = [Table1[i, :] for i in range(Table1.shape[0])]
	Table2 = [Table2[i, :] for i in range(Table2.shape[0])]
	MergedTable = []
	for r1 in Table1:
		for r2 in Table2:
			MergedTable.append((copy.deepcopy(r1), copy.deepcopy(r2)))

	for net in cols2merge:
		col1 = Nets1.index(net)
		col2 = Nets2.index(net)
		expunge_rows = []
		for i, (r1, r2) in enumerate(MergedTable):
			if not numpy.isnan(r1[col1]) and numpy.isnan(r2[col2]):
				r2[col2] = r1[col1]
			elif numpy.isnan(r1[col1]) and not numpy.isnan(r2[col2]):
				r1[col1] = r2[col2]
			elif numpy.isnan(r1[col1]) and numpy.isnan(r2[col2]):
				continue
			elif r1[col1] != r2[col2]:
				#if 
				expunge_rows.append(i)

		# print("Deleting %i of %i rows in merged table"%(len(expunge_rows), len(MergedTable)))
		for row in sorted(expunge_rows, reverse=True):
			del MergedTable[row]

	MergedTable = [numpy.append(r1, r2) for r1, r2 in MergedTable]
	MergedTable = numpy.array(MergedTable)

	#now, delete redundant columns:
	MergedNets = Nets1 + Nets2
	for net in cols2merge:
		col = MergedNets.index(net)
		del MergedNets[col]
		MergedTable = numpy.delete(MergedTable, col, axis=1)

	return MergedTable, MergedNets


def MergeLuts(G, max_states):
	#okay, new code: try merging luts, with multi-output support:
	NNC = G.number_of_nodes() + 1 #NewNodeCount

	while(1):
		#search through edges for a pair of nodes to merge:
		# max_outputs = 7
		for u, v in G.edges():
			#if u and v are both single output, merge them
			if G.nodes[u]['StateTable'].shape[0]*G.nodes[v]['StateTable'].shape[0] > max_states:
				continue
			else:
				break

		if G.nodes[u]['StateTable'].shape[0]*G.nodes[v]['StateTable'].shape[0] > max_states:
			break #no more edges match the merging condition, so we break out of this algorithm

		NewTable, NewNets = MergeStateTables(G.nodes[u]['StateTable'], G.nodes[u]['AllNets'], G.nodes[v]['StateTable'], G.nodes[v]['AllNets'])
		NewOutputs = G.nodes[u]['Outputs'] + G.nodes[v]['Outputs']
		#post-process new truth tables to reduce the state dimensionality:
		# print(NewTable)
		hypertable, NewNets = Table2HighDim(NewTable, NewNets, NewOutputs)
		NewTable = SimplifyStates(hypertable)
		
		# exit()
		#create the combined node:
		G.add_node(NNC)
		G.nodes[NNC]['StateTable'] = NewTable
		G.nodes[NNC]['AllNets'] = NewNets
		G.nodes[NNC]['Outputs'] = NewOutputs

		# tables = [(G.nodes[u]['StateTable'], G.nodes[u]['AllNets'], G.nodes[u]['Outputs']),
		# (G.nodes[v]['StateTable'], G.nodes[v]['AllNets'], G.nodes[v]['Outputs']),
		# (NewTable, NewNets, G.nodes[NNC]['Outputs'])
		# ]		
		# for table, nets, outputs in tables:
		# 	print(table)
		# 	print(nets)
		# 	print(outputs)
		# 	print()

		#copy edges to refer to the combined node:
		for node in (u, v):
			for x, y in G.in_edges(node):
				if x == NNC:
					continue
				if G.has_edge(x, NNC):
					G[x][NNC]['nets'] = G[x][NNC]['nets'] + G[x][y]['nets']
				else:
					G.add_edge(x, NNC)
					G[x][NNC]['nets'] = G[x][y]['nets']
			for x, y in G.out_edges(node):
				if y == NNC: #not sure why this would be true, but sometimes it is, so filtering it out here
					continue
				if G.has_edge(NNC, y):
					G[NNC][y]['nets'] = G[NNC][y]['nets'] + G[x][y]['nets']
				else:
					G.add_edge(NNC, y)
					G[NNC][y]['nets'] = G[x][y]['nets']


		#delete the original nodes:
		G.remove_node(u)
		G.remove_node(v)

		NNC = NNC + 1

	#reorder nodes
	# exit()
	G = nx.convert_node_labels_to_integers(G)
	print("Reduced to %i state variables"%G.number_of_nodes())
	return G
	# pl = nx.dag_longest_path_length(G)
	# print("After combining logic, depth of logic design is now %i states"%pl)


class InvertibleLogicTask(BaseTask.BaseTask):
	"""
	Represents combinatorial logic circuits in a Potts Model format.
	Since traditional mappings that use the bit values of wires as state pirimitives is fundamentally binary,
	this mapping uses the state of LUTs as state pirimitives.  This creates a way for the Potts model to reduce the number
	of Spins required to represent a particular problem.
	"""
	def __init__(self, fname, max_outputs=-1):
		"""
		Constructs a Potts model to represent the circuit described in a BLIF (Berkley Logic Interchange Format) file.
		BLIF files can be compiled from verilog using yosys and/or ABC.
		This constructor can further optimize a circuit by combining single-output LUTs into multi-output LUTs, which is
		not productive from the perspective of forward-evaluating logic circuits but can help reduce
		the state space of the Potts model invertible logic.


		:param fname: Name of blif file containing a logic circuit.
		:type fname: str
		:param max_outputs: If positive, LUTs with fewer than max_outputs ouputs are combined.
		:type max_outputs: int
		"""
		parser = blifparser.BlifParser(fname)
		blif = parser.blif
		self.e_th = 0.001 #energy of zero means no lut state conflicts exist, and a solution has been found


		#in the Potts model, nodes are LUTs, not nets.
		#So, first let's build a graph that reflects this:
		blifG = parser.get_graph()
		blifG = blifG.nx_graph
		self.blif = blif

		# blifparser is half-assed and doesn't seem to provide a way to connect boolean functions to the nodes in the graph.
		# Therefore, first step is to create a new directed graph that merges the connectivity and the boolean functions.
		# Node ID is an integer, indicating the partition number in the Potts model
		# Node data includes a Truth Table, and a vector of net names associated with each column of the truth table
		# Each edge data includes a list of nets that each edge represents, and the direction. At first there will be max one edge,
		# But if nodes are consolidated, there may be more than one edge
		G = nx.DiGraph()

		#create indexed lookup for truth table info:
		TruthTables = {}
		for bf in blif.booleanfunctions:
			if bf.output in ['$false', '$true', '$undef']:
				continue
			TruthTables[bf.output] = BlifTruth2PottsTruth(bf.truthtable)

		for u, v in blifG.edges:
			if len(v.outputs) == 0: #there's some garabage in the initial graph that needs to be filtered out
				continue
			# print(u.inputs, u.outputs, v.inputs, v.outputs)
			G.add_edge(u, v, nets=u.outputs)
			for node in (u, v):
				# print(node)
				if 'tt' not in G.nodes[node]:
					if node.outputs[0] in blif.inputs.inputs:
						#special case handling for nodes that represent input bits,
						#since these don't have any inputs or a truth table, so we fake it:
						G.nodes[node]['StateTable'] = numpy.array([[0, 0],[1, 1]])
						G.nodes[node]['AllNets'] = [node.outputs[0]+"_inp"] + node.outputs
						G.nodes[node]['Outputs'] = node.outputs
					else:
						G.nodes[node]['StateTable'] = TruthTables[node.outputs[0]]
						G.nodes[node]['AllNets'] = node.inputs + node.outputs
						G.nodes[node]['Outputs'] = node.outputs

		G = nx.convert_node_labels_to_integers(G)

		pl = nx.dag_longest_path_length(G)
		print("Depth of logic design is %i states"%pl)

		if max_outputs > 0:
			G = MergeLuts(G, max_outputs)

		self.Outputs2Partitions = {}
		# G serves as Partition2(anything else) storage
		qSizes = [1]*G.number_of_nodes()
		for i in range(G.number_of_nodes()):
			for output in G.nodes[i]['Outputs']:
				self.Outputs2Partitions[output] = i
			qSizes[i] = G.nodes[i]['StateTable'].shape[0]


			
		self.SetPartitions(numpy.array(qSizes))

		unique, counts = numpy.unique(qSizes, return_counts=True)
		print("Logic design has %i LUTs/logic elements"%len(qSizes))
		print(qSizes)


		self.InitKernelManager()


		for node1, node2 in G.edges():

			for edge_name in G[node1][node2]['nets']:
				

				#since each weight kernel represents a single bit of interaction between the state of two LUTs,
				#only the 'truth column' of each LUT table for that particular bit is needed to create the kernel
				truth_columns = []
				for node in (node1, node2):

					tt_index = G.nodes[node]['AllNets'].index(edge_name)
					truth_columns.append(G.nodes[node]['StateTable'][:,tt_index])
				
				# partitions = [self.Luts2Partitions[node.name] for node in (node1, node2)]

				self.AddKernel(lambda n: self.LCkernel(truth_columns[0], truth_columns[1], n), node1, node2, weight=1)
				self.AddKernel(lambda n: self.LCkernel(truth_columns[1], truth_columns[0], n), node2, node1, weight=1)

				k1 = self.LCkernel(truth_columns[1], truth_columns[0], False)
				k2 = self.LCkernel(truth_columns[0], truth_columns[1], False)
				# if numpy.sum(numpy.abs(k1.transpose()-k2)) != 0:
				# print(k1)
				# print(k2)
				# exit()
				# print(node1.outputs)
				# print(node2.inputs)


		self.CompileKernels()

		self.biases = numpy.zeros([len(qSizes), numpy.max(qSizes)], dtype="float32")

		self.G = G
		

	def AddBitConstraints(self, constraints):
		#uh. Let's say... constraints is dictionary of net names and their values to be fixed?
		for net in constraints:
			partition = self.Outputs2Partitions[net]
			fixed_value = constraints[net]
			# lut = self.Outputs2Names[net]
			# partition = self.Luts2Partitions[lut]
			# print(net)
			# print(self.TruthTables[net])
			col = self.G.nodes[partition]['AllNets'].index(net)
			truth_column = self.G.nodes[partition]['StateTable'][:,col]
			demoted_states = (truth_column != fixed_value)*100
			# print(demoted_states)
			# print(partition)
			# print()

			self.biases[partition,:demoted_states.shape[0]] = self.biases[partition,:demoted_states.shape[0]] + demoted_states

	def AddWordConstraint(self, word, value, nbits):
		value_bits = numpy.binary_repr(value, width=nbits)
		print(value, value_bits)
		bit_constraints = {}
		for i, bit in enumerate(value_bits):
			# print(i, bit)
			bitname = word + "[%i]"%(nbits-i-1)
			bit_constraints[bitname] = int(bit)
		self.AddBitConstraints(bit_constraints)

	def DecodeEntireState(self, state):
		"""
		Returns a dictionary of {net: value} pairs for all of the nets in the logic circuit.
		"""
		#takes the Potts state and returns the bit values of all the nets
		bitvals = {}
		for netname in self.Outputs2Partitions:
			partition = self.Outputs2Partitions[netname]
			col = self.G.nodes[partition]['AllNets'].index(netname)
			tc = self.G.nodes[partition]['StateTable'][:,col]
			lut_output = tc[state[partition]]
			bitvals[netname] = lut_output
		return bitvals

	def DecodeWordValue(self, word, nbits, state):
		"""
		Extracts integer values of multi-bit words from the given state.

		:param word: name of the value to be extracted.  Must match a name used when compiling the BLIF logic specification.
		:type word: str
		:param nbits: The number of bits in the word.  Nets named "word[0]"" through "word[nbits-1]" are assumed to comprise the word.
		:type nbits: int
		:param state: spin values, i.e. a particular model state.
		:type state: 1-D numpy int
		:return: The value of the word.
		:rtype: int
		"""
		bitvals = self.DecodeEntireState(state)
		value = 0
		for i in range(nbits):
			bitname = word + "[%i]"%i
			value = value + (2**i)*bitvals[bitname]
		return value



	def TcString(self, TcList):
		s = ""
		for v in TcList:
			if math.isnan(v):
				s = s + '-'
			else:
				s = s + "%i"%v
		return s

	def LCkernel(self, tc1, tc2, n):
		name = "Kernel between LUT truth columns %s and %s"%(self.TcString(tc1), self.TcString(tc2))
		if n:
			return name

		tc1 = numpy.expand_dims(tc1, 1)
		tc2 = numpy.expand_dims(tc2, 0)
		matches = tc1-tc2
		matches = numpy.nan_to_num(matches)
		kernel = 1*(matches!=0)
		# kernel = numpy.nan_to_num(kernel)
		return kernel