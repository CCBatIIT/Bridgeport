"""
USAGE: python SAVE_SYSTEMS.py PDB_INPUT_PATH --lig-resname


Positional Arguments:
---------------------
    PDB_INPUT_PATH: Absolute path to .pdb file to overwrite. This MUST be the final .pdb in directory called 'systems'

Optional Arguments:
-------------------
    --lig-resname: Resname of ligand to properly parse from PDB_INPUT_PATH. Typically 'UNK' in systems created by Bridgeport
"""

# Imports
import os, sys
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter
import numpy as np
sys.path.append('utils')
from ProteinPreparer import ProteinPreparer
from bp_utils import trim_env
sys.path.append('ForceFields')
from ForceFieldHandler import ForceFieldHandler
sys.path.append('MotorRow')
from MotorRow import MotorRow
from OpenMMJoiner import Joiner
from openmm import *
from openmm.app import *
from openmm.unit import *
from datetime import datetime
import argparse

# Arguments
parser - argparse.ArgumentParser()
parser.add_argument(sys_pdb, help='Positional argument must be absolute path to system.pdb file to save!')
parser.add_argument('--lig-resname', help='Resname of ligand to parse, typically "UNK"')
args = parser.parse_args()


# Organize
sys_pdb = args.sys_pdb
name = sys_pdb.split('.')[-1].split('/')[-1]

sys_xml = sys_pdb.split('.')[0] + '.xml'
assert os.path.exists(sys_xml), f"Could not find {sys_xml}"  

working_dir = '/'.join(sys_pdb.split('/')[:-1])

lig_sdf = os.path.join(working_dir, 'ligands/' + name + '.sdf')
assert os.path.exists(lig_sdf), f"Could not find {lig_sdf}"

env_pdb = os.path.join(working_dir, 'proteins/' + name + '_env.pdb')
assert os.path.exists(env_pdb), f"Could not find {env_pdb}"

trimmed_pdb = os.path.join(os.getcwd(), name + '_trimmed.pdb')
trimmed_xml = os.path.join(os.getcwd(), name + '_trimmed.xml')


# Trim _env.pdb
trim_env(env_pdb)


# Create OpenMM System
prot_sys, prot_top, prot_pos = ForceFieldHandler(env_pdb).main()
lig_sys, lig_top, lig_pos = ForceFieldHandler(lig_sdf).main()
sys, top, pos = Joiner((lig_sys, lig_top, lig_pos), (prot_sys, prot_top, prot_pos)).main()


# Write out trimmed_system.pdb
box_vectors = sys.getDefaultPeriodicBoxVectors()
translate = Quantity(np.array((box_vectors[0].x,
                               box_vectors[1].y,
                               box_vectors[2].z))/2,
                               unit=nanometer)
int = LangevinIntegrator(300 * kelvin, 1/picosecond, 0.001 * picosecond)
sim = Simulation(top, sys, int)
sim.context.setPositions(pos + translate)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Initial structure potential energy:', np.round(sim.context.getState(getEnergy=True).getPotentialEnergy()._value, 2), flush=True)

with open(trimmed_pdb, 'w') as f:
    PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Wrote:', trimmed_pdb, flush=True)

with open(trimmed_xml, 'w') as f:
    f.write(XmlSerializer.serialize(sim.system))
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Wrote:', trimmed_xml, flush=True)


# Adjust protein/ligand coordinates
lines = [line for line in open(sys_pdb, 'r').readlines() if not line.startswith('CONECT')]
open(sys_pdb, 'w').writelines(lines)

old_u = mda.Universe(sys_pdb)
u = mda.Universe(trimmed_pdb)

# Align
from MDAnalysis.analysis.align import alignto
_, _ = alignto(old_u, u, select='protein and backbone')

# Adjust protein coordinates
old_prot = old_u.select_atoms('protein')
prot = u.select_atoms('protein')
prot.positions = old_prot.positions.copy()
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Matched protein coordinates from original:', sys_pdb, flush=True)

# Adjust ligand coordinates 
if args.lig_resname != None:
    old_lig = old_u.select_atoms(f'resname {args.lig_resname}')
    lig = u.select_atoms(f'resname {args.lig_resname}')
lig.positions = old_lig.positions.copy()
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Matched ligand coordinates from original:', sys_pdb, flush=True)

u.select_atoms('all').write(sys_pdb)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Overwrote final trimmed file to:', sys_pdb, flush=True)






