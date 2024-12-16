.. PottsPlayground documentation master file, created by
   sphinx-quickstart on Mon Dec  2 13:57:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PottsPlayground's documentation!
===========================================

PottsPlayground is a Python package for constructing and simulating combinatorial optimization problems represented as Potts Models.  It includes a system for constructing weight matrices in Python and a C++/CUDA extension for minimizing the Potts model energy on a CPU or GPU using various flavors of simulated annealing.

Several example combinatorial problems are built-in, ranging from simple graph coloring to logic element placement for Ice40 FPGAs.  There are also methods for converting any Potts model into an Ising model, and for creating minor-embedded representations of Ising models.

It is intended as a demonstration of what the Potts model is and is not capable of, and as a tool for further research with the Potts model of computation.

.. toctree::
   :name: "Potts Playground"

   gettingstarted
   basetask
   tasks
   taskmodifiers
   annealing
   Schedules
   kernels