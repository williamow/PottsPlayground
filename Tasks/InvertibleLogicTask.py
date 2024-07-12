import numpy
import networkx as nx
from matplotlib import pyplot as plt
from Tasks import BaseTask
from verilog_parser.parser import parse_verilog


class InvertibleLogicTask(BaseTask.BaseTask):

	def __init__(self, fname):
		ast = parse_verilog(open(fname).read())
			


		