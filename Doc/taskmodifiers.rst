================
Model Converters
================

These tasks manipulate any input Potts task into a new format.
Currently there are two Modifier Tasks, one that can convert any Potts model into an (effective) Ising model,
and one that can minor-embed an Ising model task into a given graph, to meet requirements of any particular hardware accelerator.

Binarized
=========
.. autoclass:: PottsPlayground.Tasks.Binarized
   :members:


MinorEmbedded
=============
.. autoclass:: PottsPlayground.Tasks.MinorEmbedded
   :members: