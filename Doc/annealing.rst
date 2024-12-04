=========
Annealing
=========

Annealing API
=============
There is a single C++/CUDA extension that performs annealing.  However there are a large number of options, which are described in greater detail below.

.. py:function:: PottsPlayground.Anneal(task, PwlTemp, IC=None, nOptions=1, nActions=1, model="PottsJit", device="CPU", IncludedReports"ABCDEFGHIJKLMNOPQRSTUVWXYZ", nReplicates=1, nWorkers=1, nReports=-1)
    

   Anneals a Potts model and returns a dictionary of results.

   :param task: A PottsPlayground task object containing a complete Potts Model description.
   :param PwlTemp: An array describing the annealing schedule, i.e. the annealing temperature as a piece-wise linear function of iterations.
   :type PwlTemp: 2-D numpy float.  See elsewhere for format.
   :param IC: optional Initial Condition for the annealing run.  Either a 1-D Numpy int array containing a single IC for all replicates, or a 2-D Numpy array of ints (#Replicates X #Spins) indicating a distinct initial condition for each replicate.
   :param nOptions: The number of parallel action hypotheses to consider in each update cycle.
   :type nOptions: int
   :param nActions: The number of parallel actions to actually take in each update cycle.
   :type nActions: int
   :param model: What model to anneal.  See "Models" section.
   :type model: str
   :param device: Which physical device to run the annealing on.  Either "CPU" or "GPU"
   :type device: str
   :param IncludedReports: Str containing letter flags for which types of information are returned by the function. See "Reports" section.
   :param nReplicates: Number of Potts Model instances to compute in parallel.
   :type nReplicates: int
   :param nWorkers: On a CPU, the number of CPU threads to use; on a GPU, the number of GPU threads that work together collectively on a single replicate.
   :type nWorkers: int
   :param nReports: How many times over the course of the annealing run annealing should be stopped and vitals checked and reported. If -1, Reports are taken at the points defined in PwlTemp.
   :type nReports: int
   :return: A collection of values, states, and more that describe the outcome of the annealing.
   :rtype: dict

Models
========
Annealing can be done with one of several computational "models", which is usually the Potts Model.  However Ising model and a 
natural Traveling Salesman Model are also included for comparison purposes.  Furthermore two Potts model variants are available;
they function identically, but depending on other factors one or the other will execute faster.

PottsJit
--------

The Potts Just-in-time (JIT) backend computes the DE of changing a spin only when the spin change is actually proposed in the annealing algorithm.
This is the normal mode of computation found across simuated annealing implimentations in the software world.

PottsPrecompute
---------------

The Potts Precompute backend always knows the DE of every single possible spin change.  The DE values are kept current by an update algorithm that 
calculates DDE values after every spin change that occurs; this is more efficient when many possible DE values are queried between actual spin changes.

Ising
-----
When a Potts model has q=2 for all spins, and the weight kernels are specified in a binary-quadratic format, the Ising backend provides a more streamlined but functionally equivalent computation.

TSP
---
The TSP background is for use with the traveling salesman task only, and defines a completely different model of computation where
the order of cities is strictly enforced and only city swaps are allowed.  This is actually the standard annealing format for the traveling salesman problem.

Parallelism
===========

Several types of parallelism exist when a model is annealed.  Two are algorithmic and somewhat subtlely affect the the annealing result; the others are mechanical and concern the acceleration of an annealing run using multithreading.

Parallel Hypotheses
-------------------

Parallel Updates
----------------

Replicates
----------

Multithreading
--------------

Reports
=======
The Annealing function returns a dictionary of Reports.  Each report concerns a particular aspect or measure of the annealing run, formatted as a Numpy array.  The first dimension is always equal to the number or reports; some reports are only a scalar value at each sampling point, while others which may be a vector or matrix at each sampling point return Numpy matrices with two or three dimensions respectively.

In this list, the letter corresponds to the flag that includes the report, and the name indentifier can be used to find the information in the returned dictionary.

A - "MinEnergies" - (#Reports) - The Annealing core keeps track of the minimum-energy state each replicate has ever found.  MinEnergies is the minimum from all the replicates.
B - "AvgEnergies" - (#Reports) - The average energy of all the replicates at the time each report is gathered.
C - "AvgMinEnergies" - (#Reports) - The average of all the the all-time minimums from each replicate.
D - "AllEnergies" - (#Reports, #Replicates) - At each sample, the energy of each replicate is reported.
E - "AllStates" - (#Reports, #Replicates, #Spins) - Every spin value from every replicate, at the time of measurement.
F - "AllMinStates"
G - "MinStates"
H - "DwellTimes"
I - "Iter"
J - "Temp"
K - "Time"
L - "RealFlips"