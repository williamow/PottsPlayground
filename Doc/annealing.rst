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
   :type PwlTemp: 2-D numpy float.  See Schedules page for format.
   :param IC: optional Initial Condition for the annealing run.  Either a 1-D Numpy int array containing a single IC for all replicates, or a 2-D Numpy array of ints (#Replicates X #Spins) indicating a distinct initial condition for each replicate.
   :param nOptions: The number of parallel action hypotheses to consider in each update cycle.
   :type nOptions: int
   :param nActions: The number of parallel actions to actually take in each update cycle.
   :type nActions: int
   :param model: What model to anneal.  See "Models" section.
   :type model: str
   :param device: Which physical device to run the annealing on.  Either "CPU" or "GPU".  Defaults to CPU if the project was not compiled with GPU support, or if no suitable GPU was found on import.
   :type device: str
   :param IncludedReports: Str containing letter flags for which types of information are returned by the function. See "Reports" section.
   :param nReplicates: Number of Potts Model instances to compute in parallel.
   :type nReplicates: int
   :param nWorkers: On a CPU, the number of CPU threads to use; on a GPU, the number of GPU threads that work together collectively on a single replicate.
   :type nWorkers: int
   :param nReports: How many times over the course of the annealing run annealing should be stopped and vitals checked and reported. If -1, Reports are taken at the points defined in PwlTemp.
   :type nReports: int
   :return: A collection of energies, states, and more that describe the outcome of the annealing.
   :rtype: dict

Models
========
Annealing can be done with one of several computational "models", which is usually the Potts Model.  However Ising model and a natural Traveling Salesman Model are also included for comparison purposes.  Furthermore two Potts model variants are available; they function identically, but depending on other factors one or the other will execute faster.

PottsJit
--------

The Potts Just-in-time (JIT) backend computes the DE of changing a spin only when the spin change is actually proposed in the annealing algorithm. This is the normal mode of computation found across simuated annealing implimentations in the software world.

PottsPrecompute
---------------

The Potts Precompute backend always knows the DE of every single possible spin change.  The DE values are kept current by an update algorithm that calculates DDE values after every spin change that occurs; this is more efficient when many possible DE values are queried between actual spin changes.

Ising
-----
When a Potts model has q=2 for all spins, and the weight kernels are specified in a binary-quadratic format, the Ising backend provides a more streamlined but functionally equivalent computation.

TSP
---
The TSP background is for use with the traveling salesman task only, and defines a completely different model of computation where the order of cities is strictly enforced and only city swaps are allowed.  This is the standard annealing format for the traveling salesman problem outside of the realm of physics-inspired computing.

Parallelism
===========

Several types of parallelism exist when a model is annealed.  Two are algorithmic and somewhat subtlely affect the the annealing result; the others are mechanical and concern the acceleration of an annealing run using multithreading.

Parallel Hypotheses (**nOptions**)
----------------------------------
In typical software annealing, a single state change is proposed, evaluated, and then either actualized or not.  However it is also possible to propose multiple possible state changes, evaluate the cost of each one, and then stochastically select the best.  This can actually increase flip efficiency, but at a computational cost.

Parallel Updates (**nActions**)
-------------------------------
The more often discussed form of parallelism, multiple state changes can be made in parallel without considering their mutual effects, breaking Gibbs sampling.  State changes occur at a faster rate, but each change may be less efficient.

Replicates (**nReplicates**)
----------------------------
Multiple copies of a Potts or other model can be executed in parallel.  This is particularly helpful for sampling, since the different copies will yield independent states suitable for taking averages, etc.  For many problem-solving annealing runs however, performance is often limited by the mixing/settling time of a single replicate, and a large population of replicates provides rapidly diminishing returns.

Multithreading (**nWorkers**)
-----------------------------
This parameter has different effects for GPU and CPU operation.  On a CPU, each model replicate is contained within a single thread, and nWorkers defines how many threads can run at once - usually, the number of cores in the system.

For a GPU, threads can more easily work together to speed up computation on a single replicate.  In this case nWorkers specifies how many GPU threads should work cooperatively on each replicate; this value should be chosen based on the level of parallelism specified by nOptions and nActions.  If there is only one option and one action, multiple cooperating GPU threads won't provide much of a speed-up; however if there are many, 32 workers is a good choice (the number of threads in an Nvidia GPU warp; something to read about elsewhere).  A GPU call will automatically use as many of the available GPU cores as possible, i.e. 1024 cores if there are 32 replicates each with 32 workers.

Reports
=======
The Annealing function returns a dictionary of Reports.  Each report concerns a particular aspect or measure of the annealing run, formatted as a Numpy array.  The first dimension is always equal to the number or reports; some reports are only a scalar value at each sampling point, while others which may be a vector or matrix at each sampling point return Numpy matrices with two or three dimensions respectively.

There are two ways the reporting schedule can be defined.  If **nReports** is given, the iterations of the annealing run will be split into **nReports** segments, and the performance/behavior of the annealing run will be recorded at the end of each segment.  If nReports is not defined, the reports are made at the end of each segment defined in the piece-wise linear annealing schedule definition.

In this list, the letter corresponds to the flag that includes the report, and the name indentifier can be used to find the information in the returned dictionary.

**A - "MinEnergies" - [#Reports]** - The minimum energy acheived by any replicate at any iteration over the course of the annealing run.

**B - "AvgEnergies" - [#Reports]** - The average energy of all the replicates at the time each report is gathered.

**C - "AvgMinEnergies" - [#Reports]** - The minimum energies of each replicate from any iteration, averaged.

**D - "AllEnergies" - [#Reports, #Replicates]** - Individual energy of each replicate at the time each report is gathered.

**E - "AllStates" - [#Reports, #Replicates, #Spins]** - Every spin value from every replicate, at the time each report is gathered.

**F - "AllMinStates" - [#Reports, #Replicates, #Spins]** - Best (lowest energy) configurations found by each replicate by the time each report is gathered.

**G - "MinStates" - [#Reports, #Spins]** - Overall best (lowest energy) configuration, across all replicates, found by the time each report is gathered.  Corresponds to the **MinEnergies** report.

**H - "DwellTimes" - [#Reports, #Replicates]** - Niche use, for correcting statistical biases when running fully-parallel hypothesis annealing.  Since the annealing core selects decisions on a first-to-fire principle, samples must be weighted by their dwell time to correctly match a Boltzmann distribution.

**I - "Iter" - [#Reports]** - For convienience, the number of iterations at each report.

**J - "Temp" - [#Reports]** - The annealing temperature at each report.

**L - "RealFlips" - [#Reports]** - Execution iterations defines oppurtunities for spin flips, but often none of the proposed spin updates are actualized.  This counts the actualized spin flips taken up until the time of each report.