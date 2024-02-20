import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
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

def analogue_alignment(smiles: str, known_pdb: str, known_resname: str, analogue_out_path: str):
    """
    Creates an aligned analogue of a known ligand structure.

    Parameters:
    -----------
        smiles (str):
            String of smiles that represent the analogue to generate.

        known_pdb (str):
            Path to pdb file that contains the known ligand to align analogue to.

        known_resname (str):
            Resname of ligand to parse in known_pdb.

        analogue_out_path (str):
            Path to pdb file to write of aligned analogue. 
    """
    # Open known ligand in rdkit and MDAnalysis
    ref_mol = Chem.MolFromPDBFile(known_pdb)
    ref_sele = mda.Universe(known_pdb).select_atoms('all')

    # Create analogue with smiles
    new_mol = Chem.MolFromSmiles(smiles)
    new_mol_pdb_block = Chem.MolToPDBBlock(new_mol)
    new_mol = Chem.MolFromPDBBlock(new_mol_pdb_block)
    AllChem.EmbedMolecule(new_mol)

    # Get indices of max. common substructure 
    ref_match_inds, new_match_inds = return_max_common_substructure(ref_mol, new_mol)

    # Save atom names to align
    ref_atom_sele_str = [ref_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in ref_match_inds]
    new_atom_sele_str = [new_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in new_match_inds]
    
    # Write out analogue to .pdb file
    Chem.MolToPDBFile(new_mol, analogue_out_path, flavor=2)

    # Align analogue to reference 
    new_u = mda.Universe(analogue_out_path)
    new_sele = new_u.select_atoms('all')
    ref_align_sele = ref_sele.select_atoms('')
    for ref_atom in ref_atom_sele_str:
        ref_align_sele = ref_align_sele + ref_sele.select_atoms('name '+ ref_atom)
    new_align_sele = new_sele.select_atoms('')
    for new_atom in new_atom_sele_str:
        new_align_sele = new_align_sele + new_sele.select_atoms('name ' + new_atom)
    alignto(mobile=new_align_sele,
            reference=ref_align_sele)
    new_sele.write(analogue_out_path)

    # Remove CONECT records
    write_lines = []
    lines = open(analogue_out_path, 'r').readlines()
    for line in write_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            write_lines.append(line)

    with open(analogue_out_path, 'w') as f:
        for line in lines:
            f.write(line)
    f.close()

def return_max_common_substructure(mol1, mol2):
    """
    Return indices of maximum common substructure between two rdkit molecules
    """
    mcs = rdFMCS.FindMCS([mol1,mol2], matchValences=True, completeRingsOnly=True)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol)
    target_atm1 = []
    for i in match1:
        atom = mol1.GetAtoms()[i]
        target_atm1.append(atom.GetIdx())
    match2 = mol2.GetSubstructMatch(mcs_mol)
    target_atm2 = []
    for i in match2:
        atom = mol2.GetAtoms()[i]
        target_atm2.append(atom.GetIdx())
        
    Draw.MolsToGridImage([mol1, mol2],highlightAtomLists=[target_atm1, target_atm2])
    
    return target_atm1, target_atm2

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
