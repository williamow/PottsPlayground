from Tasks import InvertibleLogicTask
import Annealing

# for i in range(2,6):
# 	print(i)
# 	task = InvertibleLogicTask.InvertibleLogicTask("DemoFiles/Multipliers/mult%i.blif"%i, max_outputs=10)
# 	task.PrintCombinatorics()
# 	print()
# exit()

i = 16
print("Using multiplier with %i-bit inputs"%i)
task = InvertibleLogicTask.InvertibleLogicTask("DemoFiles/Multipliers/mult%i.blif"%i, max_outputs=0)

# task.MakeWeights()
# task.DisplayWeights()

task.AddWordConstraint('result', value=41*47, nbits=i)
# task.AddWordConstraint('result', value=4277546633, nbits=2*i)
# task.AddBitConstraints({'OUT': 1})
# task.AddWordConstraint('IN1', value=47, nbits=i)
# task.AddWordConstraint('IN2', value=111, nbits=i)
# print(task.biases)
# print(task.qSizes)

results = Annealing.Anneal(task, task.LinearTemp(niters=1e6, temp=2, t2=0.), OptsPerThrd=1, 
		TakeAllOptions=False, backend="PottsJit", substrate="GPU", nReplicates=2048, nWorkers=1, nReports=10)

state = results['MinStates'][-1,:]
for word in ['IN1', 'IN2']:
	print(word, task.DecodeWordValue(word, i, state))

for word in ['result']:
	print(word, task.DecodeWordValue(word, i, state))
print(state)
print(results['MinEnergies'])
# print(task.DecodeEntireState(state))