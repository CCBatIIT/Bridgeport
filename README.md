# YankBindingPose
Code for the preparation and running of YANK Simulations with the Open Force Field (Requires Python 3.9)

## Modules
### Simulation_Preparer usage:
from prepare import Simulation_Preparer \
prepper = Simulation_Preparer('3mxf.pdb', 'resname JQ1') \
prepper.generate_topologies(save_as_jsons=True) \
prepper.generate_interchanges(["openff-2.1.0.offxml", "opc-1.0.1.offxml", "ff14sb_off_impropers_0.0.3.offxml"]) \
prepper.openmm_writeout() \
