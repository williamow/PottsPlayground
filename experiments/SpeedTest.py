import PottsPlayground
import numpy
import time

def test_setting(task, **kwargs):
	
	times = [1,1]
	iters=1000
	while (times[-1]-times[-2]) < 0.1:
		iters = iters*2
		temp1 = PottsPlayground.Schedules.constTemp(iters, temp=1)
		start = time.time()
		time_results = PottsPlayground.Anneal(task, temp1, **kwargs)
		times.append(time.time()-start)

	fps = iters/2/(times[-1]-times[-2])
	return fps

# PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=5, ConstraintFactor=1.5)
PottsTask = PottsPlayground.Tasks.GraphColoring(20, 0.2, 200)
IsingTask = PottsPlayground.Tasks.Binarized(PottsTask)



print("Birds Eye, IsingPrecompute:", test_setting(IsingTask, model="IsingPrecompute", device="CPU", algo="BirdsEye"))
print("Birds Eye, Precompute:", test_setting(PottsTask, model="PottsPrecompute", device="CPU", algo="BirdsEye"))
print("Birds Eye, Jit:", test_setting(PottsTask, model="Potts", device="CPU", algo="BirdsEye"))
print("Simple OA:", test_setting(PottsTask, model="Potts", device="CPU", algo="OptionsActions"))
print("Simple:", test_setting(PottsTask, model="Potts", device="CPU", algo="Simple"))
print("Birds Eye OA, Jit:", test_setting(PottsTask, model="Potts", device="CPU", algo="OptionsActions", nActions=1, nOptions=1000000))
print("Birds Eye OA, Precompute:", test_setting(PottsTask, model="PottsPrecompute", device="CPU", algo="OptionsActions", nActions=1, nOptions=1000000))