"""
USAGE: python restart_simulation.py INPUT_DIR EQUIL_DIR NAME REP
"""

import os, sys
sys.path.append('MotorRow')
from MotorRow import MotorRow
import mdtraj as md
import math

# Arguments 
input_dir = sys.argv[1]
equil_dir = sys.argv[2]
name = sys.argv[3]
rep = sys.argv[4]

# Inputs
input_dcd = os.path.join(input_dir, name + '_' + rep + '.dcd')
input_pdb = os.path.join(equil_dir, name + '.pdb')
input_sys = os.path.join(equil_dir, name + '_sys.xml')
input_state = os.path.join(equil_dir, name + '_state.xml')

# Outputs
output_pdb = os.path.join(input_dir, name + '_' + rep + '.pdb')
output_dat = os.path.join(input_dir, name + '_' + rep + '.dat')
output_dcd = os.path.join(input_dir, name + '_' + rep + '.dcd')
output_xml = os.path.join(input_dir, name + '_' + rep + '.xml')


# Create .pdb file from last trajectory frame
traj = md.load(input_dcd, top=input_pdb)
n_frames = len(traj)
last_frame = traj.slice(-1)
last_frame.save_pdb(output_pdb)
print('wrote .pdb to', output_pdb)

# Determine number of steps left
nsteps = 250000000 - 50000 * n_frames
ncycles = math.ceil(nsteps / 5000000)
print('n_frames', n_frames, 'nsteps', nsteps, 'ncycles', ncycles)

row = MotorRow(input_sys, output_pdb, input_dir)
_, _ = row._run_step(input_state, stepnum=5, dt=2.0, ncycles=ncycles, nsteps=nsteps, nstdout=5000, fn_stdout=output_dat, ndcd=50000, append_dcd=True, fn_dcd=output_dcd, pdb_out=output_pdb, state_xml_out=output_xml, positions_from_pdb=output_pdb)










