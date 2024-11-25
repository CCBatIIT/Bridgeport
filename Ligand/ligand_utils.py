import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.lib.distances import calc_dihedrals
from MDAnalysis.coordinates.PDB import PDBWriter
from typing import List
from datetime import datetime
import rdkit
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers




def match_internal_coordinates(ref_match: mda.AtomGroup, ref_match_atoms: List, ref_match_resids: List, mobile: mda.AtomGroup, mobile_match_atoms: List, verbose: bool=False):
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
    
    def torsion_inds_to_names(atomGroup: mda.AtomGroup, tors_inds: np.array):
        atom_names = atomGroup.atoms.names
        print(atom_names)
        atom_resids = atomGroup.atoms.resids
        tors_atom_names = np.empty(tors_inds.shape, dtype='<U6')
        tors_atom_resids = np.empty(tors_inds.shape, dtype=int)
        for i, atom_inds in enumerate(tors_inds):
            for j, ind in enumerate(atom_inds):
                tors_atom_names[i,j] = atom_names[ind]
                tors_atom_resids[i,j] = atom_resids[ind]
    
        return tors_atom_names, tors_atom_resids

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
                
    def convert_atoms(mobile_atom_names:List, mobile_match_names: List, ref_match_names: List, ref_match_resids: List):
        
        ref_converted_names = []
        ref_converted_resids = []
        
        for mobile_atom in mobile_atom_names:
            if mobile_atom in mobile_match_names:
                atom_match_ind = mobile_match_names.index(mobile_atom)
                ref_eq_name = ref_match_names[atom_match_ind]
                ref_eq_resid = ref_match_resids[atom_match_ind]

                ref_converted_names.append(ref_eq_name)
                ref_converted_resids.append(ref_eq_resid)
            else:
                ref_converted_names.append('X')
                ref_converted_resids.append('X')
    
        return ref_converted_names, ref_converted_resids

    # Get analogue torsion information
    mobile_R, mobile_bat, mobile_tors = return_BAT(mobile)
    mobile_tors_inds = np.array(mobile_R._torsion_XYZ_inds)
    mobile_tors_names, _ = torsion_inds_to_names(mobile, mobile_tors_inds)

    # Iterate through torsions
    changed = [False for i in range(len(mobile_tors_names))]
    for i, atom_names in enumerate(mobile_tors_names):
        prev_tors = mobile_tors[i]
        # Convert mobile atom names to reference atoms and resids
        ref_eq_atoms, ref_eq_resids = convert_atoms(mobile_atom_names=atom_names,
                                                    mobile_match_names=mobile_match_atoms,
                                                    ref_match_names=ref_match_atoms,
                                                    ref_match_resids=ref_match_resids)

        #TEST
        if 'X' not in ref_eq_atoms:
            
            # Select reference atoms
            ref_tors_sele = ref_match.select_atoms('')
            # print(ref_match.select_atoms('all').atoms.names, ref_match.select_atoms('all').atoms.resids)
            for (r, a) in zip (ref_eq_resids, ref_eq_atoms):
                # print(ref_tors_sele.atoms.names, a, ref_tors_sele.atoms.resids, r, ref_match.select_atoms(f"resid {r} and name {a}").n_atoms)
                ref_tors_sele = ref_tors_sele + ref_match.select_atoms(f"(resid {r} and name {a})")

            # Calculated dihedral angle and assign to analogue
            try:
                c1, c2, c3, c4 = ref_tors_sele.positions
            except:
                # print(ref_eq_resids, ref_eq_atoms)
                print('reference atoms names attempted to match:', ref_tors_sele.atoms.names, 'reference resids attempted to match', ref_tors_sele.atoms.resids, flush=True)
                raise Exception("Could not match torsion")
            dihedral = calc_dihedrals(c1, c2, c3, c4)
            mobile_tors[i] = dihedral
            changed[i] = True
            if verbose:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Changed', atom_names, f'({prev_tors}) to match', ref_eq_atoms, f'({mobile_tors[i]})', flush=True)
        elif verbose:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not change', atom_names, 'to match', ref_eq_atoms, flush=True)

        
    # Convert BAT to cartesian
    mobile_bat[0, -len(mobile_tors):] = mobile_tors
    changed_inds = [i for i in range(len(changed)) if changed[i] == True]
    mobile_R._unique_primary_torsion_indices = list(np.unique(mobile_R._unique_primary_torsion_indices + changed_inds)) # Cancel handling of improper torsions from BAT class
    mobile.positions = mobile_R.Cartesian(mobile_bat[0])

    return mobile



def translate_rdkit_inds(mol, rdkit_inds):
    """
    """
    atoms = [mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in rdkit_inds]
    resids = [mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in rdkit_inds]

    return atoms, resids



def select(sele: mda.AtomGroup, atoms: List[str], resids: List[int]=None):
    """
    """
    # Make new sele
    new_sele = sele.select_atoms('')

    # If resids are provided
    if resids is not None:
        assert len(resids) == len(atoms)
        for (atom, resid) in zip(atoms, resids):
            new_sele = new_sele + sele.select_atoms(f'resid {resid} and name {atom}')

    # If resids not provided
    else:
        for atom in atoms:
            new_sele = new_sele + sele.select_atoms(f'name {atom}')


    return new_sele 

def embed_rdkit_mol(mol, template_mol=None):

    # Embed
    Chem.AllChem.EmbedMolecule(mol)

    # Minimize
    mmffps = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol,mmffps)
    maxIters = 10000
    while ff.Minimize(maxIts=1000) and maxIters>0:
        maxIters -= 1

    # Get PDB naming with correct bond orders if template is provided
    if template_mol is not None:
        pdb_block = Chem.MolToPDBBlock(mol)
        mol = Chem.MolFromPDBBlock(pdb_block, proximityBonding=False)
        mol = Chem.AllChem.AssignBondOrdersFromTemplate(template_mol, mol)

    return mol
    



    