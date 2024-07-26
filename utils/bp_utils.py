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


def analogue_alignment(smiles: str, known_pdb: str, known_smiles: str, analogue_out_path: str, analogue_atoms: List[str]=[], remove_analogue_atoms: List[str]= [], known_atoms: List[str]=[], known_resids: List[int]=[], rmsd_thres: float=None, n_conformers: int=100, align_all: bool=False):
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
    # Get name
    analogue_name = analogue_out_path.split('/')[-1].split('.pdb')[0]
    analogue_out_dir = os.path.join(analogue_out_path.split(f'{analogue_name}.pdb')[0], analogue_name + '_conformers')
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created analogue conformer directory ', analogue_out_dir, flush=True)

    # Open known ligand in rdkit and MDAnalysis
    template = Chem.MolFromSmiles(known_smiles)
    ref_mol = Chem.MolFromPDBFile(known_pdb)
    ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
    Chem.MolToPDBFile(ref_mol, known_pdb)
    ref_sele = mda.Universe(known_pdb).select_atoms('all')
    ref_sele.write(known_pdb)
    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created known ligand', known_pdb, 'from smiles:', known_smiles , flush=True)

    # Create analogue with smiles
    new_mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(new_mol) # Embed immediately to retain stereochem specified in isomeric smiles
    new_mol_pdb_block = Chem.MolToPDBBlock(new_mol)
    new_mol = Chem.MolFromPDBBlock(new_mol_pdb_block)

    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created analogue', analogue_name, 'from smiles:', smiles , flush=True)

    # Get indices of max. common substructure 
    ref_match_inds, new_match_inds = return_max_common_substructure(ref_mol, new_mol)

    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Generating', n_conformers, 'conformers of analogue.', flush=True)
    # Create counter
    n = 0 
    while n <= n_conformers:
        #Generate conformer
        AllChem.EmbedMolecule(new_mol)

        # Write out analogue to .pdb file
        Chem.MolToPDBFile(new_mol, analogue_out_path)
        
        #Get reference atoms to align
        ref_align_atoms = [ref_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in ref_match_inds] 
        ref_align_resids = [ref_mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in ref_match_inds]
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to align in reference', ref_align_atoms, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified resids to align in reference', ref_align_resids, flush=True)


        # Get analogue atoms to align
        new_align_atoms = [new_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in new_match_inds] 
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to align in analogue', new_align_atoms, flush=True)
        
        # Remove duplicates from user specifications
        for atom in analogue_atoms + remove_analogue_atoms:

            # Check if atom already identified
            if atom in new_align_atoms:

                # Find atom
                atom_ind = new_align_atoms.index(atom)
                
                # Remove atom                
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Removing', atom, 'from atoms to align in analogue and', ref_align_atoms[atom_ind], ref_align_resids[atom_ind], 'from atoms to align in reference.', flush=True)
                ref_align_atoms.pop(atom_ind)
                ref_align_resids.pop(atom_ind)
                new_align_atoms.pop(atom_ind)


        # Make selection for reference atoms to align
        ref_align_sele = ref_sele.select_atoms('')
        for ref_atom, ref_resid in zip(ref_align_atoms, ref_align_resids):
            ref_align_sele = ref_align_sele + ref_sele.select_atoms('resid '+ str(ref_resid) + ' and name '+ ref_atom)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to align in reference', ref_align_sele.atoms.names, flush=True)

        # Make selection for matching reference atoms
        ref_match_atoms = ref_align_atoms + known_atoms
        ref_match_resids = ref_align_resids + known_resids
        ref_match_sele = ref_sele.select_atoms('')
        for ref_atom, ref_resid in zip(ref_match_atoms, ref_match_resids):
            ref_match_sele = ref_match_sele + ref_sele.select_atoms('resid '+ str(ref_resid) + ' and name '+ ref_atom)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to adjust torsions in reference ', ref_match_sele.atoms.names, flush=True)

        # Make selection for analogue atoms to align
        new_u = mda.Universe(analogue_out_path)
        new_sele = new_u.select_atoms('all')
        new_align_sele = new_sele.select_atoms('')
        for new_atom in new_align_atoms:
            new_align_sele = new_align_sele + new_sele.select_atoms('name ' + new_atom)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to align in analogue', new_align_sele.atoms.names, flush=True)

        # Make selection for matching analogue atoms
        new_match_atoms = new_align_atoms + analogue_atoms
        new_match_sele = new_sele.select_atoms('')
        for new_atom in new_match_atoms:
            new_match_sele = new_match_sele + new_sele.select_atoms('name ' + new_atom)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Identified atoms to adjust torsions in analogue ', new_match_sele.atoms.names, flush=True)

        # Match internal coordinates   
        ref_bat_pdb = f'{"/".join(analogue_out_path.split("/")[:-1])}/{analogue_name}_mcs.pdb'
        ref_match_sele.write(ref_bat_pdb)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved MCS atoms of reference to', ref_bat_pdb, flush=True)
        ref_match_sele = mda.Universe(ref_bat_pdb).select_atoms('all')

        # Match internal coordinates        
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Matching torsions...', flush=True)
        new_sele = match_internal_coordinates(ref_match_sele, ref_match_atoms, ref_match_resids, new_sele, new_match_atoms)
            
        # Align analogue to reference
        if align_all:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Aligning the following atoms from reference', ref_match_sele.atoms.names, flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Aligning the following atoms from analogue', new_match_sele.atoms.names, flush=True)
            alignto(mobile=new_match_sele,
                    reference=ref_match_sele)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Aligning the following atoms from reference', ref_align_sele.atoms.names, flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Aligning the following atoms from analogue', new_align_sele.atoms.names, flush=True)
            alignto(mobile=new_align_sele,
                    reference=ref_align_sele)

        # Evaluate RMSD
        RMSD = rmsd(new_match_sele.positions.copy(), ref_match_sele.positions.copy())
        if RMSD <= rmsd_thres:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Final RMSD between MCS of analogue and reference:', RMSD, flush=True)
    
            # Write out conformer            
            if not os.path.exists(analogue_out_dir):
                os.mkdir(analogue_out_dir)
            conformer_out_path = os.path.join(analogue_out_dir, analogue_name + '_' + str(n) + '.pdb')
            new_sele.write(conformer_out_path)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved conformer to', conformer_out_path, flush=True)

            # Increase accepted conformation counter
            n += 1
            
    return analogue_out_dir, new_match_atoms

def return_max_common_substructure(mol1, mol2):
    """
    Return indices of maximum common substructure between two rdkit molecules
    """
    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.CompleteRingsOnly = True
    params.AtomCompareParameters.MatchValences = True
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.MatchFusedRingsStrict = True
    mcs = rdFMCS.FindMCS([mol1,mol2], params)
    
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

        if 'X' not in ref_eq_atoms:
            
            # Select reference atoms
            ref_tors_sele = ref_match.select_atoms('')
            for (r, a) in zip (ref_eq_resids, ref_eq_atoms):
                ref_tors_sele = ref_tors_sele + ref_match.select_atoms(f"(resid {r} and name {a})")

            # Calculated dihedral angle and assign to analogue
            try:
                c1, c2, c3, c4 = ref_tors_sele.positions
            except:
                print('reference atoms names attempted to match:', ref_tors_sele.atoms.names, 'reference resids attempted to match', ref_tors_sele.atoms.resids, flush=True)
                raise Exception("Could not match torsion")
            dihedral = calc_dihedrals(c1, c2, c3, c4)
            mobile_tors[i] = dihedral
            #changed[mobile_R._primary_torsion_indices[i]] = True
            changed[i] = True
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Changed', atom_names, f'({prev_tors}) to match', ref_eq_atoms, f'({mobile_tors[i]})', flush=True)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not change', atom_names, 'to match', ref_eq_atoms, flush=True)



        
    # Convert BAT to cartesian
    mobile_bat[0, -len(mobile_tors):] = mobile_tors
    changed_inds = [i for i in range(len(changed)) if changed[i] == True]
    mobile_R._unique_primary_torsion_indices = list(np.unique(mobile_R._unique_primary_torsion_indices + changed_inds)) # Cancel handling of improper torsions from BAT class
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
