"""
USAGE: python SIMULATE.py INPUT_DIR NAME OUTPUT_DIR REPLICATE NSTEPS
--------------------------------------------------------------------
    INPUT_DIR: absolute path to the directory with input xml and pdb
    NAME: name of both xml and pdb file before the extension
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored
    REPLICATE: Replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting
    NSTEPS: Number of simulation steps to take. Default is 167000000 which equils 501 ns w/ a 3 fs timestep

DEFAULT SIMULATION PARAMETERS:
------------------------------
    timestep -> 3 fs
    stdout -> 10 ps
    dcdout -> 100 ps
    append_dcd -> True
    temp -> 300 K
    pressure -> 1 bar
"""

import os, sys
sys.path.append('MotorRow')
from MotorRow import MotorRow

# Inputs
input_dir = sys.argv[1]
name = sys.argv[2]
input_sys = os.path.join(input_dir, name+'_sys.xml')
input_state = os.path.join(input_dir, name+'_state.xml')
input_pdb = os.path.join(input_dir, name+'.pdb')
print(input_pdb)

# Outputs
output_dir = os.path.join(sys.argv[3], name)
rep = sys.argv[4]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.exists(output_dir)
output_pdb = os.path.join(output_dir, name+'_'+rep+'.pdb')
output_dcd = os.path.join(output_dir, name+'_'+rep+'.dcd')
output_dat = os.path.join(output_dir, name+'_'+rep+'.dat')
output_xml = os.path.join(output_dir, name+'_'+rep+'.xml')

# Simulation parameters
try:
    n_steps = int(sys.argv[5])
except:
    n_steps = 167000000

# Append?
if os.path.exists(output_dcd):
    append_dcd = True
else:
    append_dcd = False

# Simulate
row = MotorRow(input_sys, input_pdb, output_dir)
_, _ = row._run_step(input_state, stepnum=5, dt=2.0, nsteps=n_steps, nstdout=5, fn_stdout=output_dat, ndcd=50000, append_dcd=append_dcd, fn_dcd=output_dcd, pdb_out=output_pdb, state_xml_out=output_xml)

