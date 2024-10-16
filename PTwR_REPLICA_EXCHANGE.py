"""
USAGE: python REPLICA_EXCHANGE.py $INPUT_DIR $NAME $OUTPUT_DIR $REPLICATE $SIM_TIME $NUM_OF_REPLICA 

PARAMETERS:
-----------
    INPUT_DIR: absolute path to the directory with input xml and pdb
    NAME: pdb file before the extension
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored
    REPLICATE: Replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting
    SIM_TIME: Total simulation aggregate time. Default is 500 ns. 
    NUM_OF_REPLICA: number of replica to start with between T_min (300 K) and T_max (360 K)
"""


import os, sys
sys.path.append('FultonMarket')
from FultonMarket import FultonMarket

# Inputs
input_dir = sys.argv[1]
name = sys.argv[2]
input_sys = os.path.join(input_dir, name+'_sys.xml')
input_state = os.path.join(input_dir, name+'_state.xml')
input_pdb = os.path.join(input_dir, name+'.pdb')
print(input_pdb)

# Outputs
rep = int(sys.argv[4])
output_dir = os.path.join(sys.argv[3], name + '_' + str(rep))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.exists(output_dir)

# Simulation parameters
try:
    total_sim_time = int(sys.argv[5])
except:
    total_sim_time = 500

sub_sim_length = 50

try:
    n_replica = int(sys.argv[6])
except:
    n_replica = 90

# Restraints
selection_string = 'protein and ('
intracellular_inds = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]
intracellular_inds = [ind - 67 for ind in intracellular_inds]
print(len(intracellular_inds))
for ind in intracellular_inds:
    selection_string += f'(resid {ind}) or '
selection_string = selection_string[:-4] + ')'

# Run rep exchange
market = FultonMarket(input_pdb=input_pdb, input_system=input_sys, input_state=None)

market.run(total_sim_time=total_sim_time, iteration_length=0.001,
	   n_replicates=n_replica, sim_length=sub_sim_length,
           output_dir=output_dir, restrained_atoms_dsl=selection_string)

