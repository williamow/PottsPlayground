import PottsPlayground
import numpy
from matplotlib import pyplot as plt

plt.figure()

task = PottsPlayground.Tasks.TravelingSalesman(ncities=8, ConstraintFactor=1.5, seed=0)
task = PottsPlayground.Tasks.Binarized(task, ConstraintFactor=1.5)

for VirtualTimestep in [10000, 20000, 50000, 100000, 1e6]:
	tempsteps = [i/10. + 0.05 for i in range(10)]
	realFPS = []
	for temp in tempsteps:
		t = PottsPlayground.Schedules.constTemp(1e5, temp)
		results = PottsPlayground.Anneal(task, t, nActions=int(VirtualTimestep), model="Ising-NHPP")
		realFPS.append(results["RealFlips"][-1]/1e5)
	plt.plot(tempsteps, realFPS, label="%i"%int(VirtualTimestep))
	print(numpy.mean(realFPS))

plt.legend()
plt.show()