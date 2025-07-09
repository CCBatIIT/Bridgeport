import textwrap, sys, os, glob, shutil
import numpy as np
from copy import deepcopy
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

# rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20
from IPython.display import display
from typing import List
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
from ligand_utils import *


class Ligand():
    """
    """
    def __init__(self, working_dir: str, name: str, 
                 resname: str=False, smiles: str=False,
                 chainid: str=False, sequence: str=False,
                 verbose: bool=False):
        """
        
        """

        # Initialize attributes
        self.working_dir = working_dir
        self.name = name
        self.pdb = os.path.join(working_dir, name + '.pdb')
        self.sdf = os.path.join(working_dir, name + '.sdf')
        self.verbose = verbose

        # Small molecule?
        if resname is not False:
            self.resname = resname
        if smiles is not False:
            self.smiles = smiles

        # Peptide?
        if chainid is not False:
            self.chainid = chainid
            if sequence is not None:
                self.sequence = sequence
            else:
                self.sequence = False


        if chainid is not False and resname is not False:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Must set either resname or chainid', flush=True)
                  

    def prepare_ligand(self, 
                       small_molecule_params: bool=True,
                       sanitize: bool=True,
                       removeHs: bool=True,
                       proximityBonding: bool=False, 
                       pH: float=7.0,
                       nstd_resids: List[int]=[],
                       neutral_Cterm: bool=False,
                       loops: bool=False,
                       chain: str=False,
                       visualize: bool=False):
        """
        Prepare a ligand

        Parameters:
        -----------
            small_molecule_params (bool):
                If true, treat ligand like a small molecule. Default is True.

            sanitize (bool):
                If true, sanitize molecule with rdkit. Default is True. Only applicable if small_molecule_params is True. 

            removeHs (bool):
                If true, remove any hydrogens that may be present. Default is True. Only applicable if small_molecule_params is True. 

            pH (float):
                pH to protonate a peptide ligand. Default is 7.0.

            nstd_resids (List[int]):
                List of nonstandard resids to conserve from input structure. 

            neutral_C-term (bool):
                If true, neutralize the C-terminus of a peptide ligand. Only applicable is small_molecule_params is False
        """
        # Set attributes
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.proximityBonding = proximityBonding
        self.pH = pH
        self.nstd_resids = nstd_resids
        self.neutral_Cterm = neutral_Cterm
        self.visualize = visualize
        self.chain = chain
        self.loops = loops
        
        # If treating ligand like a small molecule
        if small_molecule_params:
            self._prepare_small_molecule()

        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found peptide ligand with resname:', self.chainid, flush=True)
            self._prepare_peptide()

        # Change chain, if specified
        if self.chain != False:
            u = mda.Universe(self.pdb)
            u.atoms.chainIDs = self.chain
            u.atoms.write(self.pdb)

    
    def return_rdkit_mol(self, from_pdb: bool=True, 
                               from_smiles: bool=True,
                               sanitize: bool=True,
                               removeHs: bool=False,
                               proximityBonding: bool=True):
        """
        """        
        # Load molecules
        if from_smiles:
            template = Chem.MolFromSmiles(self.smiles, sanitize=True)
        if from_pdb:
            mol = Chem.MolFromPDBFile(self.pdb, sanitize=sanitize, removeHs=removeHs, proximityBonding=proximityBonding)

            if from_smiles:
                
                # Assign bond order from smiles
                try:
                    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
                except:
                    mol_copy = deepcopy(mol)
                    Chem.rdDepictor.Compute2DCoords(mol_copy)
                    display(Draw.MolsToGridImage([mol_copy], subImgSize=(600,600)))
                    display(Draw.MolsToGridImage([template], subImgSize=(600,600)))
                    raise Exception(f'Could not find match between molecule smiles: {Chem.MolToSmiles(mol)} and template: {Chem.MolToSmiles(template)}')

            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=False)
    
        # Visualize, if specified
        if self.visualize:
            if from_pdb:
                mol_copy = deepcopy(mol)
            else:
                mol_copy = deepcopy(template)
            Chem.rdDepictor.Compute2DCoords(mol_copy)
            display(Draw.MolsToGridImage([mol_copy], subImgSize=(600,600)))

        
        if from_pdb and from_smiles:
            return template, mol
        elif from_pdb:
            return mol
        elif from_smiles:
            return template



    def return_MDA_sele(self):
        """
        """
        # Assertions
        assert os.path.exists(self.pdb)

        # MDA 
        u = mda.Universe(self.pdb)
        return u.select_atoms('all')



    def _prepare_small_molecule(self):
        """
        """
        # Load input w/ rdkit
        template, self.mol = self.return_rdkit_mol(sanitize=self.sanitize, removeHs=self.removeHs, proximityBonding=self.proximityBonding)
        
        # Add Hs
        self.mol = AllChem.AddHs(self.mol, addCoords=True, addResidueInfo=False)

        # Save
        Chem.MolToPDBFile(self.mol, self.pdb)
        writer = Chem.SDWriter(self.sdf)
        writer.write(self.mol)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved prepared ligand to', self.pdb, self.sdf, flush=True)



    def _prepare_peptide(self):
        """
        """
        from utils.utils import write_FASTA
        from utils.ProteinPreparer import ProteinPreparer
        from RepairProtein.RepairProtein import RepairProtein
        # Repair with RepairProtein
        if self.sequence is not False:
            # Write fasta
            fasta_fn = os.path.join(os.getcwd(), 'lig.fasta')
            write_FASTA(self.sequence, 'lig', fasta_fn)

            # RepairProtein
            temp_working_dir = os.path.join(os.getcwd(), 'modeller')
            repairer = RepairProtein(pdb_fn=self.pdb,
                                     fasta_fn=fasta_fn,
                                     working_dir=temp_working_dir)

            repairer.run(pdb_out_fn=self.pdb,
                         tails=True,
                         nstd_resids=self.nstd_resids,
                         loops=self.loops)

        # Protonate with pdb2pqr30
        pp = ProteinPreparer(pdb_path=self.pdb,
                 working_dir=self.working_dir,
                 pH=self.pH,
                 env='SOL',
                 ion_strength=0) 
        prot_mol_path = pp._protonate_with_pdb2pqr()
        prot_mol_path = pp._protonate_with_PDBFixer()        
        os.rename(prot_mol_path, self.pdb)

        # Neutralize C terminus
        if self.neutral_Cterm:
            # Open pdb
            pdb_lines = open(self.pdb, 'r').readlines()
            oxt_line = ''
            for line in pdb_lines:
                if line.find('OXT') != -1:
                    oxt_line = line

            # Change OXT -> NXT
            nxt_line = [c for c in oxt_line]
            nxt_line[13] = 'N'
            nxt_line[-4] = 'N'
            nxt_line = ''.join(nxt_line)

            # Calculate H coordinates
            pdb = md.load_pdb(self.pdb)
            nxt_sele = pdb.topology.select('name OXT')[0]
            nxt_xyz = pdb.xyz[0, nxt_sele]*10
            print(nxt_xyz)
            c_sele = pdb.topology.select('name C and resSeq ' + str(pdb.topology.atom(nxt_sele).residue.resSeq))[0]
            c_xyz = pdb.xyz[0, c_sele]*10
            h1_xyz, h2_xyz = compute_C_positions(c_xyz, nxt_xyz)
            
            # Add hydrogens
            h1_line = deepcopy(nxt_line)
            h2_line = deepcopy(nxt_line)
            h_lines = []
            atom_no = pdb.n_atoms
            for i, (h_line, name, h_xyz) in enumerate(zip([h1_line, h2_line], ['H1', 'H2'], [h1_xyz, h2_xyz])):
                h_line = [a for a in h_line]
                h_line[13:17] = name + '  '
                h_line[-4] = 'H'
                h_xyz_str = ''
                for crd in h_xyz:
                    if crd < 0 or crd >= 10:
                        h_xyz_str += f'{crd:.3f}'
                    else:
                        h_xyz_str += f' {crd:.3f}'
                    h_xyz_str += '  '
                h_line[32:56] = h_xyz_str
    
                if atom_no + i + 1 < 10:
                    h_line[8:11] = '  ' + str(atom_no_i+1)
                elif atom_no + i + 1 >= 10 and atom_no + i + 1 < 100:
                    h_line[8:11] = ' ' + str(atom_no + i + 1)            
                elif atom_no + i + 1 >= 100 and atom_no + i + 1 < 1000:
                    h_line[8:11] = str(atom_no + i + 1)
                else:
                    raise NotImplementedError(atom_no + i + 1)
                
                h_lines.append(''.join(h_line))
    
            with open(self.pdb, 'w') as f:
                for line in pdb_lines:
                    if line.startswith('ATOM'):
                        if line.find('OXT') != -1:
                            f.write(nxt_line)
                        else:
                            f.write(line)
                        
                for h_line in h_lines:
                    f.write(h_line)
    
                f.write('END')
                
                f.close()





        
        