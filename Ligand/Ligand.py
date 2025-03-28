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
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
from ligand_utils import *
from utils.utils import write_FASTA
from utils.ProteinPreparer import ProteinPreparer
from RepairProtein.RepairProtein import RepairProtein

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


        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Must set either resname or chainid', flush=True)

    

    def prepare_ligand(self, 
                       small_molecule_params: bool=True,
                       sanitize: bool=True,
                       removeHs: bool=True,
                       proximityBonding: bool=False, 
                       pH: float=7.0,
                       nstd_resids: List[int]=[],
                       neutral_Cterm: bool=False,
                       visualize: bool=True):
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
        
        # If treating ligand like a small molecule
        print('!!!!!', small_molecule_params)
        if small_molecule_params:
            self._prepare_small_molecule()

        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found peptide ligand with resname:', self.chainid, flush=True)
            self._prepare_peptide()


    
    def return_rdkit_mol(self, from_pdb: bool=True, 
                               from_smiles: bool=True,
                               sanitize: bool=None,
                               removeHs: bool=None,
                               proximityBonding: bool=None):
        """
        """        
        # Load molecules
        if from_smiles:
            template = Chem.MolFromSmiles(self.smiles, sanitize=True)
        if from_pdb:
            mol = Chem.MolFromPDBFile(self.pdb, sanitize=sanitize, removeHs=removeHs, proximityBonding=proximityBonding)

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
        template, mol = self.return_rdkit_mol(sanitize=self.sanitize, removeHs=self.removeHs, proximityBonding=self.proximityBonding)
        if self.visualize:
            display(Draw.MolsToGridImage([template], subImgSize=(600,600)))

        # Assign bond order from smiles
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)

        # Add Hs
        mol = AllChem.AddHs(mol, addCoords=True, addResidueInfo=True)

        # Save
        Chem.MolToPDBFile(mol, self.pdb)
        writer = Chem.SDWriter(self.sdf)
        writer.write(mol)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved prepared ligand to', self.pdb, self.sdf, flush=True)



    def _prepare_peptide(self):
        """
        """
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
                         loops=False)

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
            pdb_lines = open(self.pdb, 'r').readlines()
            oxt_line = ''
            for line in pdb_lines:
                if line.find('OXT') != -1:
                    oxt_line = line

            nxt_line = [c for c in oxt_line]
            nxt_line[13] = 'N'
            nxt_line[-4] = 'N'
            nxt_line = ''.join(nxt_line)

            with open(self.pdb, 'w') as f:
                for line in pdb_lines:
                    if line.find('OXT') == -1:
                        f.write(line)
                    else:
                        f.write(nxt_line) 
                f.close()

            # Protonate again
            prot_mol_path = pp._protonate_with_PDBFixer()        # THIS DOES NOT ADD NECESSARY HYDROGENS OF NXT :(
            os.rename(prot_mol_path, self.pdb)

            # Clean up extra Hs
            lines = open(self.pdb, 'r').readlines()
            prev_resid = 0
            write_lines = []
            H_counter = 0
            max_resid = max([int(line[24:26].strip()) for line in lines if line.startswith('ATOM')])
            for i, line in enumerate(lines):
                if line.startswith('ATOM'):
                    atom, resid = line[12:16].strip(), int(line[24:26].strip())
                    if resid >= prev_resid:
                        prev_resid = resid
                        if resid == max_resid:
                            if atom == 'H':
                                if H_counter > 0:
                                    write_lines.append(line[:12] + f' HT{H_counter} NCT' + line[20:])
                                else:
                                    write_lines.append(line[:17] + 'NCT' + line[20:])
                                H_counter += 1
                            else:
                                write_lines.append(line[:17] + 'NCT' + line[20:])
                        else:
                            write_lines.append(line)
                else:
                    write_lines.append(line)

            with open(self.pdb, 'w') as f:
                for line in write_lines:
                    f.write(line)       




        
        