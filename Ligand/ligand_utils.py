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
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate




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
    try:
        atoms = [mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in rdkit_inds]
        resids = [mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in rdkit_inds]
    except:
        for i in rdkit_inds:
            try:
                print(i, mol.GetAtoms()[i].GetMonomerInfo().GetName(), mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber())
            except:
                from rdkit.Chem import Draw
                mol_copy = Chem.Mol(mol)
                Chem.rdDepictor.Compute2DCoords(mol_copy)
                dopts = Chem.Draw.rdMolDraw2D.MolDrawOptions()
                dopts.addAtomIndices = True
                print('Could not find atom with ind:', i)
                display(Draw.MolsToGridImage([mol_copy], drawOptions=dopts))
                raise Exception()
                

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

    # Add Hs and Embed
    template_mol = AllChem.AddHs(template_mol)
    mol = AllChem.AddHs(mol)
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
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=False)
        mol = Chem.AllChem.AssignBondOrdersFromTemplate(template_mol, mol)
        Chem.AssignStereochemistryFrom3D(mol)

    mol = AllChem.RemoveHs(mol)

    return mol


    
def mol_with_atom_idx(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol



def compute_C_positions(A, B, angle_deg=120.0, length=1.03):
    A = np.array(A)
    B = np.array(B)
    
    # Step 1: BA vector
    BA = A - B
    BA_unit = BA / np.linalg.norm(BA)

    # Step 2: Define a perpendicular vector in the same plane
    # Choose a normal to define the plane — default is Z
    ref = np.array([0, 0, 1])
    plane_normal = np.cross(BA, ref)
    if np.linalg.norm(plane_normal) < 1e-6:
        # BA is parallel to Z, pick a different reference
        ref = np.array([0, 1, 0])
        plane_normal = np.cross(BA, ref)
    plane_normal /= np.linalg.norm(plane_normal)

    # Vector perpendicular to BA, in the plane
    perp = np.cross(plane_normal, BA_unit)
    perp /= np.linalg.norm(perp)

    # Step 3: Rotate by ±θ
    theta = np.deg2rad(angle_deg)
    
    # First C position (rotation in one direction)
    BC1 = (np.cos(theta) * BA_unit + np.sin(theta) * perp) * length
    C1 = B + BC1

    # Second C position (mirror across BA, i.e., -theta)
    BC2 = (np.cos(theta) * BA_unit - np.sin(theta) * perp) * length
    C2 = B + BC2

    return C1, C2


def remove_xml_atoms(xml_fn, resname, remove_atoms, change_atoms, external_bonds=None):

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    
    # Remove atom types
    # remove_types = []
    # atom_types = root.find('AtomTypes')
    # for at in list(atom_types):
    #     name = at.attrib.get('name').split('-')[1]
    #     if name in remove_atoms:
    #         remove_types.append(at.attrib.get('name'))
    #         atom_types.remove(at)
    
    # Remove residues
    for r in list(root.find('Residues')):
        r.attrib['name'] = resname
        
        # Remove atoms
        change_types = {}
        for atom in list(r.findall('Atom')):
            if atom.attrib.get('name') in remove_atoms:
                r.remove(atom)
            if atom.attrib.get('name') in change_atoms.keys():
                change_types[atom.attrib.get('type')] = change_atoms[atom.attrib.get('name')]
        
        # Remove bonds
        for bond in list(r.findall('Bond')):
            if bond.attrib.get('atomName1') in remove_atoms or bond.attrib.get('atomName2') in remove_atoms:
                r.remove(bond)
    
        # Add external bond
        if external_bonds != None:
            for atom_name in external_bonds:
                r.append(ET.Element('ExternalBond', {'atomName': atom_name}))
    
    # # Remove forces
    def change_force_entries(section_name, *attrib_keys):
        section = root.find(section_name)
        if section is not None:
            for entry in list(section):
                new_entry_needed = False
                new_entry = {}
                for k in entry.keys():
                    if entry.attrib.get(k) in change_types.keys():
                        new_entry_needed = True
                        new_entry[k] = change_types[entry.attrib.get(k)]
                    else:
                        new_entry[k] = entry.attrib.get(k)
                        
                if new_entry_needed:
                    section.append(ET.Element(entry.tag, new_entry))
    
    change_force_entries("HarmonicBondForce", "type1", "type2")
    change_force_entries("HarmonicAngleForce", "type1", "type2", "type3")
    change_force_entries("PeriodicTorsionForce", "type1", "type2", "type3", "type4")
    # change_force_entries("NonbondedForce", "type")
    
    tree.write(xml_fn, encoding="utf-8", xml_declaration=True)




    
