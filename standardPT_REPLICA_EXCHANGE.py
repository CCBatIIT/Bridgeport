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

sub_sim_length = 5

try:
    n_replica = int(sys.argv[6])
except:
    n_replica = 90

# Run rep exchange
market = FultonMarket(input_pdb=input_pdb, input_system=input_sys, input_state=None)

market.run(total_sim_time=total_sim_time, iteration_length=0.001, n_replicates=n_replica, sim_length=sub_sim_length, output_dir=output_dir)

