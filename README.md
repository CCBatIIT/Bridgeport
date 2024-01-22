# Minh Sim Prep
## Overview
This repo is a gneeral tool for the preparation of molecular complexes for simulation.

## Environment
Construct the environment by downloading the following packages from conda-forge:
openff-toolkit, pdbfixer, openbabel

$ conda create -n openff -c conda-forge openff-toolkit, pdbfixer, openbabel
$ pip install pdb2pqr

## Modules
### Simulation_Preparer usage:
from prepare import Simulation_Preparer \
prepper = Simulation_Preparer('3mxf.pdb', 'resname JQ1') \
prepper.generate_topologies(save_as_jsons=True) \
prepper.generate_interchanges(["openff-2.1.0.offxml", "opc-1.0.1.offxml", "ff14sb_off_impropers_0.0.3.offxml"]) \
prepper.openmm_writeout() \
