PottsPlayground is for constructing and simulating combinatorial optimization problems represented as Potts Models.  It includes a system for constructing weight matrices in Python and a C++/CUDA extension for minimizing the Potts model energy on a CPU or GPU using various flavors of simulated annealing. It is intended as a demonstration of what the Potts model is and is not capable of, and as a tool for further research with the Potts model of computation.

Read the full docs `here <https://williamow.github.io/PottsPlayground/index.html>`_.

Built-in combinatorial problems can be generated and solved easily:

.. code-block:: python

	import PottsPlayground
	PottsTask = PottsPlayground.Tasks.TravelingSalesman(ncities=10, ConstraintFactor=1.5)
	temp = PottsPlayground.Schedules.SawtoothTempLog2Space(MaxTemp=1., MinTemp=0.05, nTeeth=3, nIters=1e4)
	results = PottsPlayground.Anneal(PottsTask, temp)
	final_best_soln = results['MinStates'][-1,:]
	PottsTask.DisplayState(final_best_soln)

There is also a BaseTask class that can be used to try out any Potts Model problem representations you can think of, and anneal it using the high-performance backend.

Features:

* Structured construction of Potts Models
* Automatic conversion from Potts to Ising models
* Minor Embedding of Ising models
* GPU accelerated annealing and sampling
* Parallel replicates
* Parallel updating
* Comprehensive statistical reporting
