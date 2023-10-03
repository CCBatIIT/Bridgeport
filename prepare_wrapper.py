from prepare import Simulation_Preparer
from ybp_utils import *

################
## USER INPUT ##
################
prepper = Simulation_Preparer('3mxf.pdb', 'resname JQ1')
yank_output_dir = 'JQ1_run'
yaml_file_fn = 'JQ1_script.yaml'

###############
## No Edits ###
###############
prepper.generate_topologies(save_as_jsons=True)
prepper.generate_interchanges(["openff-2.1.0.offxml", "opc-1.0.1.offxml", "ff14sb_off_impropers_0.0.3.offxml"])
prepper.openmm_writeout()
complex_fns = ('complex_final.pdb', 'complex_final.xml')
solvent_fns = ('solvent_final.pdb', 'solvent_final.xml')
restraint_string = determine_restrained_residues(complex_fns[0], 4, prepper.ligand_resname)

with open(yaml_file_fn, 'w') as f:
    f.write(write_the_yaml(complex_fns, solvent_fns, prepper.ligand_resname, yank_output_dir, restraint_string))
