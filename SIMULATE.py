"""
USAGE: python SIMULATE.py INPUT_DIR NAME OUTPUT_DIR REPLICATE NSTEPS
    INPUT_DIR: absolute path to the directory with input xml and pdb
    NAME: name of both xml and pdb file before the extension
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored
    REPLICATE: Replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting
    NSTEPS: Number of simulation steps to take. Default is 167000000 which equils 501 ns w/ a 3 fs timestep

DEFAULT SIMULATION PARAMETERS:





"""

import os, sys
sys.path.append('MotorRow')
from MotorRow import MotorRow

# Inputs
input_dir = sys.argv[1]
name = sys.argv[2]
input_xml = os.path.join(input_dir, name+'.xml')
input_pdb = os.path.join(input_dir, name+'.pdb')

# Outputs
output_dir = os.path.join(sys.argv[3], name)
rep = sys.argv[4]
assert isinstance(rep, int)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.exists(output_dir)
output_pdb = os.path.join(output_dir, name+'_'+rep+'.pdb')
output_dcd = os.path.join(output_dir, name+'_'+rep+'.dcd')
output_dat = os.path.join(output_dir, name+'_'+rep+'.dat')
output_xml = os.path.join(output_dir, name+'_'+rep+'.xml')

# Simulation parameters
try:
    n_steps = sys.argv[5]
except:
    n_steps = 167000000

# Simulate
row = MotorRow(input_xml, input_pdb, output_dir)
state, pdb = row._minimize(input_pdb)
_, _ = row._run_step(state, stepnum=5, dt=3.0, n_steps=n_steps, nstdout=30000, fn_stdout=output_dat, ndcd=300000, append_dcd=True, fn_dcd=output_dcd, pdb_out=output_pdb, state_xml_out=output_xml)
