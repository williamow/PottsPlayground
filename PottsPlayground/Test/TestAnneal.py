import PottsPlayground
import random
import numpy

print("Testing: PottsPlayground.Anneal")

PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=5, ConstraintFactor=1.5)
valid_energies, invalid_energies = PottsTask.EnergyBands()
optE = numpy.min(valid_energies)

temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=3, niters=1e4)

results = PottsPlayground.Anneal(PottsTask, temp)
if results["MinEnergies"][-1] > optE*1.001 or results["MinEnergies"][-1] < optE*0.999:
	print("FAILURE: did not return expected optimal result")
	print("Test: default options")
	exit()

for i in range(25):
	model = random.choice(["PottsJit", "PottsPrecompute", "Tsp"])
	device = random.choice(["CPU", "GPU"])
	nReplicates = random.choice([1,3,41])
	nWorkers = random.choice([1,3,41])
	nActions = random.choice([1,2,3])
	nOptions = random.choice([1,5,10])

	
	results = PottsPlayground.Anneal(PottsTask, temp, 
		model=model, 
		device=device, 
		nReplicates=nReplicates, 
		nWorkers=nWorkers,
		nActions=nActions,
		nOptions=nOptions)

	if results["MinEnergies"][-1] > optE*1.001 or results["MinEnergies"][-1] < optE*0.999:
		print("FAILURE: did not return expected optimal result")
		print("Test: model=%s, device=%s, nReplicates=%i, nWorkers=%i, nActions=%i, nOptions=%i"%(model, device, nReplicates, nWorkers, nActions, nOptions))
		exit()


#test that initial conditions set correctly:
temp = PottsPlayground.Schedules.LinearTemp(temp=1, niters=0)
for device in ['CPU', 'GPU']:
	#test single broadcast initial condition
	IC = numpy.random.randint(0, high=5, size=[5], dtype=numpy.int32)
	result = PottsPlayground.Anneal(PottsTask, temp, device=device, nReplicates=4, IC=IC)
	for i in range(4):
		assert (result['AllStates'][-1,i,:] == IC).all()
	#test unique initial conditions
	IC = numpy.random.randint(0, high=5, size=[4,5], dtype=numpy.int32)
	result = PottsPlayground.Anneal(PottsTask, temp, device=device, nReplicates=4, IC=IC)
	assert (result['AllStates'][-1,:,:] == IC).all()
