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
	Kernel definition for Binary Quadratic Ising models
	where each spin is interpreted to have values of 0 or 1.
	Is simply

	K = [[0	0]
	     [0	1]]

	Since the processing component of PottsPlayground additionally weighs each kernel,
	the BinaryQuadratic kernel is a fixed value, allowing a simplified Ising computation mode.
	"""
	if n:
		return "BQ"
	k = numpy.zeros([2,2])
	k[1,1] = 1
	return k

def IsingSpin(n=False):
	"""
	Kernel definition for physical Ising models,
	where each spin is interpreted to have values of -1 or +1.
	Is simply

	K = [[-1  1]
	     [ 1 -1]]

	The kernel is designed such that when it is scaled positively,
	It creates correlation between the connected i and j spins;
	and when scaled by a negative value, creates anticorrelation.
	This is to match typical sign conventions in +1/-1 spin systems.
	
	"""
	if n:
		return "IS"
	k = numpy.array([[-1, 1], [1, -1]])
	return k

def Identity(dim=2, n=False):
	"""
	Identity kernel, i.e. all entries are zero except for the diagonal, which have values 1.
	Often used to prevent Potts nodes from being in the same state as each other.

	To use this kernel, the dimension needs to be pre-specified before
	the kernel is passed to AddWeight, such as 

	Identity10 = lambda n: Identity(dim=10, n=n)
	AddWeight(Identity10, spin_i, spin_j)
	
	:param dim: Size of the identity kernel.
	:type dim: int
	"""
	if n:
		return "Identity"
	return numpy.eye(dim)