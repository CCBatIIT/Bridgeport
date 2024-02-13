import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
import mdtraj as md
from pdbfixer import PDBFixer
from openbabel import openbabel
from datetime import datetime
#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *


def change_resname(pdb_file_in, pdb_file_out, resname_in, resname_out):
    """
    Changes a resname in a pdb file by changing all occurences of resname_in to resname_out
    
    """

    with open(pdb_file_in, 'r') as f:
        lines = f.readlines()
    print('Effected Lines:')
    eff_lines = [line for line in lines if resname_in in line]
    for line in eff_lines:
        print(line, "-->", line.replace(resname_in, resname_out))
    user_input = input("Confirm to make these changes [y/n] :")
    if user_input == 'y':
        lines = [line.replace(resname_in, resname_out) for line in lines]
        with open(pdb_file_out, 'w') as f:
            f.writelines(lines)
        return pdb_file_out
    else:
        print('Aborting....')
        return None

def describe_system(sys: System):
    box_vecs = sys.getDefaultPeriodicBoxVectors()
    print('Box Vectors')
    [print(box_vec) for box_vec in box_vecs]
    forces = sys.getForces()
    print('Forces')
    [print(force) for force in forces]
    num_particles = sys.getNumParticles()
    print(num_particles, 'Particles')

def describe_state(state: State, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")
