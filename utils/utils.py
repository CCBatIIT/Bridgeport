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

    # Check for atoms outside of box
    remove_residues = []
    for i, line in enumerate(lines):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            key = line[6]
            resname = line[17:20]
            chain = line[21]
            resid = line[22:26]
            x = float(line[31:38])
            y = float(line[39:46])
            z = float(line[47:54])
            
            if x > max_coords[0] or y > max_coords[1] or z > max_coords[2]:
                remove_residues.append([resname, chain, resid, key])
                 
            elif x < min_coords[0] or y < min_coords[1] or z < min_coords[2]:
                remove_residues.append([resname, chain, resid, key])



    remove_resnames = np.array(remove_residues)[:,0]
    remove_chains = np.array(remove_residues)[:,1]
    remove_resids = np.array(remove_residues)[:,2]
    keys = np.array(remove_residues)[:,3]
        
    # Remove lines
    write_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            key, resname, chain, resid = line[6], line[17:20], line[21], line[22:26]
            x = float(line[31:38])
            y = float(line[39:46])
            z = float(line[47:54])

            if resname in remove_resnames and resid in remove_resids and chain in remove_chains:
                inds1 = np.where(remove_resnames == resname)[0]
                inds2 = np.where(remove_resids == resid)[0]
                inds3 = np.where(remove_chains == chain)[0]
            
                cross = np.intersect1d(inds1, inds2)
                cross = np.intersect1d(inds3, cross)
                
                if len(cross) == 0 or key not in keys[cross]:
                    write_lines.append(line)
                elif resname == 'HOH' and (x-1 < max_coords[0] and y-1 < max_coords[1] and z-1 < max_coords[2]) and (x+1 > min_coords[0] and y+1 > min_coords[1] and z+1 > min_coords[2]):
                    write_lines.append(line)

                
            else:
                write_lines.append(line)
        else:
            write_lines.append(line)


    # Write lines
    with open(pdb, 'w') as f:
        f.writelines(write_lines)
        f.close()





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
