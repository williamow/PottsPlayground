"""
A collection of definitions for annealing schedules.  Each function returns a 2-D numpy array
that can be interpreted as a piecewise linear function inside of PottsPlayground.Anneal.

The piecewise linear function defines multiple aspects of an annealing run: the number of iterations/spin updates,
a changing temperature as annealing progresses, and at what points annealing should be temporarily halted
so that intermediate data can be recorded.  The intermediate points are the discontinuities in the Piecewise linear function.
Note that progress can also be recorded more frequently, on a set periodic frequency, using the 
nReports option of the PottsPlayground.Anneal function.
"""
import numpy

def constTemp(niters, temp):
	"""
	Anneal a model at a constant temperature for the given number of iterations.

	:param niters: The total number of updates/spin flips over the course of the annealing run.
	:type niters: int
	:param temp: Computational temperature at which the model operates.
	:type temp: float
	"""
	PwlTemp = numpy.zeros([2, 2], dtype="float32")
	PwlTemp[0,0] = temp
	PwlTemp[0,1] = temp
	PwlTemp[1,0] = 0
	PwlTemp[1,1] = niters
	return PwlTemp

def LinearTemp(niters, temp, temp2=0.0001):
	"""
	Anneal a model at a linearly changing temperature, starting at temp and ending at temp2.

	:param niters: The total number of updates/spin flips over the course of the annealing run.
	:type niters: int
	:param temp: Initial temperature.
	:type temp: float
	:param temp2: Final temperature.
	:type temp2: float
	"""
	PwlTemp = numpy.zeros([2, 2], dtype="float32")
	PwlTemp[0,0] = temp
	PwlTemp[0,1] = temp2
	PwlTemp[1,0] = 0
	PwlTemp[1,1] = niters
	return PwlTemp

def constTempLog2Space(niters, temp, nReports):
	"""
	Anneal a model at a constant temperature for the given number of iterations.
	Defines nReports breakpoints spaced logarithmically, 
	so that both fast and slow dynamics of model evolution can be recorded.

	:param niters: The total number of updates/spin flips over the course of the annealing run.
	:type niters: int
	:param temp: Constant temperature used during sampling.
	:type temp: float
	:param nReports: The number of times over the course of annealng that the model is stopped and measured.  The duration ratio between samples is calculated to meet this spec.
	:type nReports: int
	"""
	PwlTemp = numpy.zeros([2, nReports+1], dtype="float32")
	PwlTemp[0,:] = temp
	ratio = niters**(1./(nReports-1))
	for i in range(nReports):
		PwlTemp[1,i+1] = int((ratio**i))
		if (PwlTemp[1,i+1] <= PwlTemp[1,i]):
			PwlTemp[1,i+1] = PwlTemp[1,i] + 1 #just for very begining, so that each step is at least 1 iter more than the last
	PwlTemp[1,0] = 0
	return PwlTemp

def SawtoothTempLog2Space(niters, temp, MinTemp, nTeeth, FirstTooth=10):
	"""
	Anneal a model according to a sawtooth temperature waveform.
	The duration of each sawtooth increases exponentially.
	Unless nReports is defined in the annealing run, a sample will be recorded at the end of each
	sawtooth when the model has been frozen into at least a local minima.
	Allows the "problem solving" capacity of an annealing setup to be observed more directly,
	especially for models that only mix at high temperatures but only settle into valid states at low temperatures.

	:param niters: The total number of updates/spin flips over the course of the annealing run.
	:type niters: int
	:param temp: Initial, high temperature of the sawtooth temperature profile.
	:type temp: float
	:param MinTemp: The temperature at the end of each linearly changing sawtooth cycle.
	:type MinTemp: float
	:param nTeeth: The number of sawtooth cycles over the course of the annealing run.  The duration ratio between teeth is calculated to meet this spec.
	:type nTeeth: int
	:param FirstTooth: How many iterations the first tooth should be, or equivalently.
	:type FirstTooth: int
	"""
	PwlTemp = numpy.zeros([2, nTeeth*2+1], dtype="float32")
	ratio = (niters/FirstTooth)**(1./(nTeeth-1))
	for i in range(nTeeth):
		PwlTemp[1,i*2+1] = int(FirstTooth*(ratio**i))   #bottom of tooth
		PwlTemp[1,i*2+2] = int(FirstTooth*(ratio**i))+1 #top of next tooth
		PwlTemp[0,i*2+1] = MinTemp   #bottom of tooth
		PwlTemp[0,i*2+2] = temp #top of next tooth
		# if (PwlTemp[1,i*2+1] <= PwlTemp[1,i]):
			# PwlTemp[1,i*2+1] = PwlTemp[1,i*] + 1 #just for very begining, so that each step is at least 1 iter more than the last
	PwlTemp[1,0] = 0
	PwlTemp[0,0] = temp
	return PwlTemp