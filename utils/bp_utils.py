import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.bat import BAT
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

def analogue_alignment(smiles: str, known_pdb: str, analogue_out_path: str, analogue_atoms: List[str]=[], known_atoms: List[str]=[], known_resids: List[int]=[], rmsd_thres: float=2.0):
    """
    Creates an aligned analogue of a known ligand structure. 

    Parameters:
    -----------
        smiles (str):
            String of smiles that represent the analogue to generate.

        known_pdb (str):
            Path to pdb file that contains the known ligand to align analogue to.

        analogue_out_path (str):
            Path to pdb file to write of aligned analogue. 

        analogue_atoms(List[str]):
            List of atoms in analogue that have matching atoms in known_pdb. EX: ['C12', 'N1', 'O2']

        known_atoms (List[str]):
            List of known atom names in reference structure to add to the alignment RMSD evaluation. Must match order of analogue_atoms. EX: ['CA', 'N', 'O']

        known_resids (List[int]):
            List of known resids of atoms in known_atoms. EX:['UNK', 'UNK']   

    Returns:
    --------
        RMSD (float):
            RMSD of similar atoms after alignment protocol. 
    """
    # Open known ligand in rdkit and MDAnalysis
    ref_mol = Chem.MolFromPDBFile(known_pdb)
    Chem.MolToPDBFile(ref_mol, known_pdb)
    ref_sele = mda.Universe(known_pdb).select_atoms('all')
    ref_sele.write(known_pdb)

    # Create analogue with smiles
    new_mol = Chem.MolFromSmiles(smiles)
    new_mol_pdb_block = Chem.MolToPDBBlock(new_mol)
    new_mol = Chem.MolFromPDBBlock(new_mol_pdb_block)
    AllChem.EmbedMolecule(new_mol)

    # Get indices of max. common substructure 
    ref_match_inds, new_match_inds = return_max_common_substructure(ref_mol, new_mol)

    RMSD = rmsd_thres + 1
    counter = 0
    while RMSD > rmsd_thres:
        #Generate conformer
        AllChem.EmbedMolecule(new_mol, randomSeed=counter)
        
        # Write out analogue to .pdb file
        Chem.MolToPDBFile(new_mol, analogue_out_path)
    
        #Get reference atoms to align
        ref_align_atoms = [ref_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in ref_match_inds] 
        ref_align_resids = [ref_mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in ref_match_inds] 
        ref_align_sele = ref_sele.select_atoms('')
        for ref_atom, ref_resid in zip(ref_align_atoms, ref_align_resids):
            ref_align_sele = ref_align_sele + ref_sele.select_atoms('resid '+ str(ref_resid) + ' and name '+ ref_atom)

        # Get matching reference atoms 
        ref_match_atoms = [ref_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in ref_match_inds] + [atom for atom in known_atoms]
        ref_match_resids = [ref_mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in ref_match_inds] + [resid for resid in known_resids]
        ref_match_sele = ref_sele.select_atoms('')
        for ref_atom, ref_resid in zip(ref_match_atoms, ref_match_resids):
            ref_match_sele = ref_match_sele + ref_sele.select_atoms('resid '+ str(ref_resid) + ' and name '+ ref_atom)

        # Get analogue atoms to align
        new_u = mda.Universe(analogue_out_path)
        new_sele = new_u.select_atoms('all')
        new_align_sele = new_sele.select_atoms('')
        new_align_atoms = [new_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in new_match_inds] 
        for new_atom in new_align_atoms:
            new_align_sele = new_align_sele + new_sele.select_atoms('name ' + new_atom)

        # Get analogue matching atoms
        new_match_sele = new_sele.select_atoms('')
        new_match_atoms = [new_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in new_match_inds] + [atom for atom in analogue_atoms]
        for new_atom in new_match_atoms:
            new_match_sele = new_match_sele + new_sele.select_atoms('name ' + new_atom)
            
        # Match internal coordinates   
        ref_match_sele.write('test.pdb')
        ref_match_sele = mda.Universe('test.pdb').select_atoms('all')
        new_sele = match_internal_coordinates(ref_match_sele, ref_match_atoms, ref_match_resids, new_sele, new_match_atoms)
        if os.path.exists('test.pdb'):
            os.remove('test.pdb')  
            
        # Align analogue to reference
        alignto(mobile=new_align_sele,
                reference=ref_align_sele)

        # Evaluate RMSD
        RMSD = rmsd(new_match_sele.positions.copy(), ref_match_sele.positions.copy())
        print(RMSD)
        counter += 1

    new_sele.write(analogue_out_path, bonds=None)

    return RMSD

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
        
    display(Draw.MolsToGridImage([mol1, mol2],highlightAtomLists=[target_atm1, target_atm2]))
    
    return target_atm1, target_atm2

def match_internal_coordinates(ref_match: mda.AtomGroup, ref_match_atoms: List, ref_match_resids: List, mobile: mda.AtomGroup, mobile_match_atoms: List):
    """
    Return an MDAnalysis.AtomGroup with internal coordinates that match a reference. 

    Parameters:
    -----------
        ref_match (mda.AtomGroup)
            Selection of chemically equivalent atoms to calculate internal angles to copy to new molecule (mobile).

        ref_match_atoms (List[str])
            List of atom names that correspond to the atoms that have an equivalent atom in the mobile group. EX: ['CA', 'CB']

        ref_match_resids (List[int])
            List of resids of atoms in ref_match_atoms. EX:['UNK', 'UNK']

        mobile (mda.AtomGroup)
            Selection of atoms to change internal angles to match those of ref_match. 

        mobile_match_atoms (List[str]):
            List of atom names that correspond to the matching atom in mobile compared to ref_match_atoms. EX: ['C12', 'C13']

    Returns:
    --------
        mobile (mda.AtomGroup)
            Selection of atoms with torsions that reflect the internal coordinates present in ref_match. 
    """

    def return_BAT(atomGroup: mda.AtomGroup):
        R = BAT(atomGroup)
        R.run()
        bat = R.results.bat.copy()
        tors = bat[0, -len(R._torsion_XYZ_inds):]
        
        return R, bat, tors
    
    def torsion_inds_to_names(atomGroup: mda.AtomGroup, tors: np.array):
        atom_names = atomGroup.atoms.names
        atom_resids = atomGroup.atoms.resids
        tors_atom_names = np.empty(tors.shape, dtype='<U6')
        tors_atom_resids = np.empty(tors.shape, dtype=int)
        for i, atom_inds in enumerate(tors):
            for j, ind in enumerate(atom_inds):
                tors_atom_names[i,j] = atom_names[ind]
                tors_atom_resids[i,j] = atom_resids[ind]
    
        return tors_atom_names, tors_atom_resids
                
    def convert_ref_to_mobile_torsion_names(ref_tors_names, ref_tors_resids, ref_match_names, ref_match_resids, mobile_match_names):
        ref_to_mobile = np.empty(ref_tors_names.shape, dtype='<U6')
        for i, (atoms, resids) in enumerate(zip(ref_tors_names, ref_tors_resids)):
            for j, (atom, resid) in enumerate(zip(atoms, resids)):
                for k, (ref_atom, ref_resid) in enumerate(zip(ref_match_names, ref_match_resids)):
                    if ref_atom == atom and ref_resid == resid:
                        ref_to_mobile[i,j] = mobile_match_names[k]
    
        return ref_to_mobile
    
    def change_torsions(mobile_tors, mobile_tors_names, ref_converted, ref_tors):
        new_tors = mobile_tors.copy()
        for i, mobile_names in enumerate(mobile_tors_names):
            for j, ref_names in enumerate(ref_converted):
                if list(mobile_names) == list(ref_names):
                    new_tors[i] = ref_tors[j]
                elif list(mobile_names) == list(np.flip(ref_names)):
                    new_tors[i] = ref_tors[j] - np.pi
    
        return new_tors


    mobile_R, mobile_bat, mobile_tors = return_BAT(mobile)
    for fragment in ref_match.fragments:    
        ref_R, ref_bat, ref_tors = return_BAT(fragment)
        
        ref_tors_inds = np.array(ref_R._torsion_XYZ_inds)
        mobile_tors_inds = np.array(mobile_R._torsion_XYZ_inds)
        
        ref_tors_names, ref_tors_resids = torsion_inds_to_names(fragment, ref_tors_inds)
        mobile_tors_names, _ = torsion_inds_to_names(mobile, mobile_tors_inds)
        
        ref_converted = convert_ref_to_mobile_torsion_names(ref_tors_names, ref_tors_resids, ref_match_atoms, ref_match_resids, mobile_match_atoms)
        mobile_tors = change_torsions(mobile_tors, mobile_tors_names, ref_converted, ref_tors)
    
        mobile_bat[0, -len(mobile_tors):] = mobile_tors
        mobile.positions = mobile_R.Cartesian(mobile_bat[0])

    return mobile

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
