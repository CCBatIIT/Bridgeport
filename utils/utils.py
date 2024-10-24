import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.lib.distances import calc_dihedrals
from MDAnalysis.coordinates.PDB import PDBWriter
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
#rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20
from IPython.display import display
from typing import List


def write_FASTA(sequence, name, fasta_path):

    # Write FASTA
    FASTA = f""">P1;{name}
                 sequence; {name}:::::::::
                 {sequence}*"""
    
    with open(fasta_path, 'w') as f:
        f.write(FASTA)
        f.close()



def trim_env(pdb, padding: float=15):
    """
    Remove the excess membrane and solvent added by calling PDBFixer.addMembrane()

    Protocol:
    ---------
        1. Get dimensions of protein and new periodic box
        2. Write corresponding CRYST1 line
        3. Identify atoms outside of box
        4. Identify corresponding resnames and resids outside of box
        5. Remove residues outside of box 
        6. Overwrite original file ('pdb' parameter)

    Parameters:
    -----------
        pdb (str):
            String path to pdb file to trim.

        padding (float):
            Amount of padding (Angstrom) to trim to. Default is 15 Angstrom to accomodate the default 10 Angstrom NonBondededForce cutoff.     
    """

    # Get protein dimensions
    u = mda.Universe(pdb)
    prot_sele = u.select_atoms('protein')
    max_coords = np.array([prot_sele.positions[:,i].max() for i in range(3)]) + padding
    min_coords = np.array([prot_sele.positions[:,i].min() for i in range(3)]) - padding
    deltas = np.subtract(max_coords, min_coords)
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified new box size:', deltas, flush=True)

    
    # Write CRYST1 line
    temp_crys_pdb = 'temp_crys.pdb'
    writer = PDBWriter(temp_crys_pdb)
    writer.CRYST1(list(deltas) + [90, 90, 90])
    writer.close()
    
    cryst1_line = open(temp_crys_pdb, 'r').readlines()[0]
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Writing new CRYST1 line:', cryst1_line[:-2], flush=True)
    os.remove(temp_crys_pdb)

    lines = open(pdb, 'r').readlines()
    for i, line in enumerate(lines):
        if line.startswith('CRYST1'):
            cryst_line_ind = i
            break

    lines = lines[:cryst_line_ind] + [cryst1_line] + lines[cryst_line_ind+1:]
    open(pdb, 'w').writelines(lines)


    # Identify atoms outside of box
    u = mda.Universe(pdb)
    all_atoms = u.select_atoms('all')
    remove_inds = []
    for i, atom_xyz in enumerate(all_atoms.positions):
        if (atom_xyz[0] > max_coords[0]) or (atom_xyz[1] > max_coords[1]) or (atom_xyz[2] > max_coords[2]):
            remove_inds.append(i)
        elif (atom_xyz[0] < min_coords[0]) or (atom_xyz[1] < min_coords[1]) or (atom_xyz[2] < min_coords[2]):
            remove_inds.append(i)
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified ', len(remove_inds), 'atoms to remove.' , flush=True)



    # Identify resnames and resids outside of box
    remove_resnames = all_atoms.resnames[remove_inds]
    remove_resids = all_atoms.resids[remove_inds]
    
    remove = np.unique([[remove_resnames[i], remove_resids[i]] for i in range(len(remove_inds))], axis=0)
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified ', len(remove), 'resids to remove.' , flush=True)

    # Remove residues outside of box
    trimmed_sele = u.select_atoms('all')
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Untrimmed no. of atoms:', trimmed_sele.n_atoms , flush=True)
    
    for rem in remove:
        trimmed_sele = trimmed_sele - u.select_atoms(f'resname {rem[0]} and resid {rem[1]}')
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Trimmed no. of atoms:', trimmed_sele.n_atoms , flush=True)

    # Write over original file
    trimmed_sele.write(pdb)
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Trimmed environment saved to:', pdb , flush=True)

    # # Remove MDAnalysis headers
    # lines = [line for line in open(pdb, 'r').readlines() if line.startswith('ATOM') or line.startswith('HETATM') or line.startswith('CONECT') or line.startswith('CRYST1')]
    # open(pdb, 'w').writelines(lines)



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
