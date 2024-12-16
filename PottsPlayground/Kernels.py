"""
Commonly used kernel definitions for constructing Potts models.
Each 'kernel' is a weight matrix between two Potts spins *i* and *j*.
Given a model state, each kernel :math:`W_{i,j}` contributes energy :math:`W_{i,j}[m_i,m_j]`
to the overall model, where :math:`m_i` and :math:`m_j` are the spin values of Potts nodes :math:`i` and :math:`j`.

Each kernel 'definition' is actually a function that returns either a string identifier for the kernel
or the kernel itself.  This is designed so that complex kernels that take a long time to construct are
only constructed once for each problem (A large problem might have thousands of copies of a big kernel that 
is created element by element).  Kernels in this module should work seemlessly with the baseTask AddKernel method,
however many nontrivial applications will require custom-designed kernels.
"""
import numpy

def BinaryQuadratic(n=False):
	"""
	Kernel definition for Binary Quadratic Ising models.
	Is simply

	K = [[0	0]
	  [0	1]]

	Since the processing component of PottsPlayground additionally weighs each kernel,
	the BinaryQuadratic kernel is a fixed value, allowing a simplidied Ising computation mode.
	"""
	if n:
		return "BQ"
	k = numpy.zeros([2,2])
	k[1,1] = 1
	return k