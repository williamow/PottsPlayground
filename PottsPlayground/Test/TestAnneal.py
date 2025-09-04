import PottsPlayground
import random
import numpy
import scipy.stats

print("Testing: PottsPlayground.Anneal")

#====================================================================================================
#test set 1: check that minimum TSP energy can be found:
PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=5, ConstraintFactor=1.5)

valid_energies, invalid_energies = PottsTask.EnergyBands()
optE = numpy.min(valid_energies)

solver_temp = PottsPlayground.Schedules.SawtoothTempLog2Space(temp=1., MinTemp=0.05, nTeeth=3, niters=1e4)

#randomly check a number of operating combinations, 
for i in range(25):
	args = {}
	args['model'] = random.choice(["Potts", "PottsPrecompute", "Tsp"])
	args['device'] = random.choice(["CPU", "GPU"])
	args['algo'] = random.choice(["Simple", "BirdsEye", "OptionsActions"])
	args['nReplicates'] = random.choice([1,3,41])
	args['nWorkers'] = random.choice([1,3,41])
	args['nActions'] = random.choice([1,2,3]) #only applies if algo=OptionsActions
	args['nOptions'] = random.choice([1,5,1000]) #only applies if algo=OptionsActions

	results = PottsPlayground.Anneal(PottsTask, solver_temp, **args)

	if results["MinEnergies"][-1] > optE*1.001 or results["MinEnergies"][-1] < optE*0.999:
		print("FAILURE: did not return expected optimal result, settings =")
		print(args)
		exit()


#=======================================================================================================
#test 2: that initial conditions set correctly:
print("testing setting initial conditions, expect two warnings:")
PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=5, ConstraintFactor=1.5)
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
	#test that the initial conditions can be left empty using the None keyword:
	PottsPlayground.Anneal(PottsTask, temp, device=device, nReplicates=4, IC=None)
	#test that the program does not segfault when the initial conditions are wrong dimensions:
	PottsPlayground.Anneal(PottsTask, temp, device=device, nReplicates=3, IC=IC)

#=========================================================================================================
#test set 3: make sure that average sampled energy is right (verify boltzmann distribution)
# PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=3, ConstraintFactor=1.5)
# IsingTask = PottsPlayground.Tasks.Binarized(PottsTask, ConstraintFactor=1.5)

# nSamples = 100
# ref_temp = 2.
# temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=ref_temp)
# results = PottsPlayground.Anneal(IsingTask, temp, nReports=nSamples)
# ref_samples = results["AllEnergies"]


# #randomly check a number of operating combinations, 
# for i in range(25):
# 	args = {}
# 	args['model'] = random.choice(["Ising", "Potts", "PottsPrecompute", "PottsPrecomputePE"])
# 	args['device'] = random.choice(["CPU", "GPU"])
# 	if args['device'] == "CPU":
# 		args['algo'] = random.choice(["Simple", "BirdsEye", "OptionsActions"])
# 	else:
# 		args['algo'] = random.choice(["OptionsActions"])
# 		args['nWorkers'] = random.choice([1,4,16])
# 	if args['algo'] == "OptionsActions":
# 		args['nOptions'] = random.choice([1,1000])

# 	#have to set the right temperature, depending on the algorithm
# 	if (args['algo'] == "BirdsEye" and args["model"] == "PottsPrecomputePE"):
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=ref_temp)
# 	elif (args['algo'] == "BirdsEye"):
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=2*ref_temp)
# 	elif (args['algo'] == "OptionsActions" and args['nOptions'] == 1000 and args["model"] == "PottsPrecomputePE"):
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=2*ref_temp)
# 	elif (args['algo'] == "OptionsActions" and args['nOptions'] == 1000):
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=4*ref_temp)
# 	elif (args['algo'] == "OptionsActions"):
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=ref_temp)
# 	else:
# 		temp = PottsPlayground.Schedules.constTemp(niters=1e4, temp=ref_temp)

# 	results = PottsPlayground.Anneal(IsingTask, temp, nReports=nSamples, **args)

# 	if (args['algo'] == "BirdsEye" or ("nOptions" in args and args["nOptions"] == 1000)):
# 		#these have a continuous time interpretation, 
# 		#and thus must be scaled by the dwell time:
# 		dwell = results["DwellTimes"]
# 		norm_dwell = dwell/numpy.mean(dwell)
# 		samples = results["AllEnergies"]*norm_dwell
# 	else:
# 		samples = results["AllEnergies"]

	
# 	#two-sample t-test that this average energy and reference avg energy come from the same distribution:
# 	p = scipy.stats.ttest_ind(a=samples, b=ref_samples, equal_var=True)
# 	# print(p)
# 	# e_avg = numpy.mean(samples)
# 	# e_var = numpy.var(samples)
# 	# T = (e_avg - ising_e_avg)/(e_var/nSamples+ising_e_var/nSamples)**0.5

# 	# sp = ((nSamples*ising_e_var+nSamples*e_var)/(2*nSamples))**0.5
# 	# p = (ising_e_avg-e_avg)*(nSamples/2.)**0.5 / sp
# 	if (p.pvalue < 0.05):
# 		print("The average sampled energy is wrong! p=%.3f"%p.pvalue)
# 		print("Expected Eavg: %.3f Actual Eavg: %.3f"%(numpy.mean(ref_samples), numpy.mean(samples)))
# 		print(args)
# 		# exit()