import textwrap, sys, os, glob, shutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.lib.distances import calc_dihedrals
from MDAnalysis.coordinates.PDB import PDBWriter
import mdtraj as md
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20
from IPython.display import display
from typing import List
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
try:
    from Ligand.Ligand import Ligand
except:
    from Ligand import Ligand
from ligand_utils import *
import py3Dmol



class Analogue(Ligand):
    """
    """

    def __init__(self, template: Ligand, working_dir: str, name: str, 
                     resname: str=False, smiles: str=False,
                     chainid: str=False, sequence: str=False,
                     verbose: bool=False, visualize: bool=True):
        """
        """

        # Initialize inheritated attributes
        super().__init__(working_dir, name, resname, smiles, chainid, sequence, verbose)
        self.visualize = visualize

        # Assign new attributes
        self.template = template
        self.conformer_dir = os.path.join(self.working_dir, f'{name}_conformers')
        if not os.path.exists(self.conformer_dir):
            os.mkdir(self.conformer_dir)

    

    def get_MCS(self, 
                subImgSize: tuple=(600,600),
                add_atoms: List[List[int]]=None,
                remove_atoms: List[int]=None,
                strict: bool=False,
                removeHs: bool=True,
                from_pdb: bool=False,
                from_smiles: bool=True,
                sanitize: bool=True,
                proximityBonding: bool=True):
        """
        """

        # Set attributes
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.proximityBonding = proximityBonding

        # Get rdkit molecules
        mol = self.return_rdkit_mol(from_pdb=from_pdb,
                                  from_smiles=from_smiles,
                                  sanitize=self.sanitize,
                                  removeHs=self.removeHs,
                                  proximityBonding=self.proximityBonding)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created analogue', self.name, 'from smiles:', self.smiles , flush=True)
        
        template_mol = self.template.return_rdkit_mol(from_pdb=True,
                          from_smiles=False,
                          sanitize=True,
                          removeHs=False,
                          proximityBonding=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created known ligand', self.template.name, 'from smiles:', self.template.smiles , flush=True)
        
        # Detect MCS
        self.strict = strict
        self._detect_MCS(mol, template_mol)
        self.matching_inds = deepcopy(self.align_inds)
        self.template_matching_inds = deepcopy(self.template_align_inds)

        # Remove user specified atoms
        if remove_atoms is not None:
            self._remove_atoms_from_MCS(remove_atoms)

        # Add user specified atoms
        if add_atoms is not None:
            self._add_atoms_to_MCS(add_atoms)
            
        # Print matching atoms
        if self.verbose:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found matching inds:', flush=True)
            for (atom, ref_atom) in zip(self.matching_inds, self.template_matching_inds):
                print(f'atom={atom}, ref_atom={ref_atom}')

        # Draw molecules
        if self.visualize:
            template_mol_copy = Chem.Mol(self.template_mol)
            Chem.rdDepictor.Compute2DCoords(template_mol_copy)
            dopts = Chem.Draw.rdMolDraw2D.MolDrawOptions()
            dopts.addAtomIndices = True
            print('Analogue, Template')
            display(Draw.MolsToGridImage([self.mol, template_mol_copy],
                                         subImgSize=subImgSize,
                                         highlightAtomLists=[self.matching_inds, self.template_matching_inds],
                                         drawOptions=dopts))

    

    def generate_conformers(self, n_conformers: int=1, align_all: bool=False, rmsd_thresh: float=3.0):
        """
        """

        # Run setup methods
        self._load_molecules(load_template=True)

        # Get MDA atoms
        if align_all:
            self.align_inds = deepcopy(self.matching_inds)
            self.template_align_inds = deepcopy(self.template_matching_inds)
            
        self._get_MDA_atoms()
        

        # Iterate for the n_conformers
        n=0
        conformer = 0
        while n < n_conformers:

            # Generate conformer
            self._load_molecules()
            
            # Make selections
            self._make_selections()

            # Save MCS
            self.bat_pdb = os.path.join(self.working_dir, f'{self.name}_mcs.pdb')
            self.template_matching_sele.write(self.bat_pdb)
            self.template_matching_sele = mda.Universe(self.bat_pdb).select_atoms('all')

            # Match internal coordinates
            self.sele = match_internal_coordinates(ref_match=self.template_matching_sele,
                                       ref_match_atoms=self.template_matching_atoms,
                                       ref_match_resids=self.template_matching_resids,
                                       mobile=self.sele,
                                       mobile_match_atoms=self.matching_atoms,
                                       verbose=self.verbose)

            # Align
            alignto(mobile=self.align_sele, reference=self.template_align_sele, tol_mass=1000)

            # Save if RMSD is below threshold
            RMSD = rmsd(self.align_sele.positions.copy(), self.template_align_sele.positions.copy())
            if RMSD <= rmsd_thresh:
                conformer_out_pdb = os.path.join(self.conformer_dir, f'{self.name}_{n}.pdb')
                self.sele.write(conformer_out_pdb)
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved conformer to', conformer_out_pdb, flush=True)

                # Up that ticker :)
                n += 1
            conformer += 1

        # Save final structure
        self.sele.write(self.pdb)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved first conformer to', self.pdb, flush=True)


    def _detect_MCS(self, mol1, mol2):
        """
        Return indices of maximum common substructure between two rdkit molecules

        mol1 should be analogue, mol2 should be template
        
        """
        
        # Set parameters
        params = rdFMCS.MCSParameters()
        if self.strict:
            params.AtomCompareParameters.CompleteRingsOnly = True
            params.AtomCompareParameters.MatchValences = True
            params.AtomCompareParameters.RingMatchesRingOnly = True
            params.BondCompareParameters.MatchFusedRingsStrict = True

        # Compute MCS
        mcs = rdFMCS.FindMCS([mol1,mol2], params)

        # Get atom lists
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

        # Set final attributes
        self.mol = mol1
        self.template_mol = mol2
        self.align_inds = target_atm1
        self.template_align_inds = target_atm2
    

    
    def _load_molecules(self, load_template: bool=False):
        """
        """
        # Store bond orders from SMILES and save to .pdb for MDAnalysis
        self.mol = embed_rdkit_mol(self.mol, self.mol)
        Chem.MolToPDBFile(self.mol, self.pdb)

        # Load template from pdb
        if load_template:
            self.template_mol = self.template.return_rdkit_mol(from_pdb=True,
                                                               from_smiles=False,
                                                               sanitize=True,
                                                               removeHs=True,
                                                               proximityBonding=True)
        
            
            # Load with MDAnalysis
            self.template_sele = self.template.return_MDA_sele()

    

    def _get_MDA_atoms(self):
        """
        """
        # Translate atoms for alignment
        self.align_atoms, _ = translate_rdkit_inds(self.mol, self.align_inds)
        self.template_align_atoms, self.template_align_resids = translate_rdkit_inds(self.template_mol, self.template_align_inds)

        # Translate atoms for torsion matching
        self.matching_atoms, _ = translate_rdkit_inds(self.mol, self.matching_inds)
        self.template_matching_atoms, self.template_matching_resids = translate_rdkit_inds(self.template_mol, self.template_matching_inds)


    
    def _add_atoms_to_MCS(self, add_atoms):
        """
        """
        
        # Add atoms
        for atom, temp_atom in add_atoms:

                # Add
                self.matching_inds.append(atom)
                self.template_matching_inds.append(temp_atom)


    
    def _remove_atoms_from_MCS(self, remove_atoms):
        """
        """
        # Remove user specified atoms
        for atom in remove_atoms:

            # See if already in there
            if atom in self.matching_inds:
            
                # Find atoms
                atom_ind = self.matching_inds.index(atom)
    
                # Remove
                self.matching_inds.pop(atom_ind)
                self.template_matching_inds.pop(atom_ind)


    
    def _make_selections(self):
        """
        """
        # Make selections
        self.sele = self.return_MDA_sele()
        self.align_sele = select(self.sele, self.align_atoms)
        self.template_align_sele = select(self.template_sele, self.template_align_atoms, self.template_align_resids)
        self.matching_sele = select(self.sele, self.matching_atoms)
        self.template_matching_sele = select(self.template_sele, self.template_matching_atoms, self.template_matching_resids)

        # print(self.matching_sele.atoms.names)



    def visualize_alignment(self):
        view = py3Dmol.view()
        print(f'BLUE: {self.name}')
        print(f'YELLOW: {self.template.name}')
        view.setBackgroundColor('white')
        view.addModel(open(self.pdb, 'r').read(),'pdb')
        view.addModel(open(self.template.pdb, 'r').read(),'pdb')
        view.setStyle({'model':0}, {'stick': {'colorscheme':'blueCarbon'}})
        view.setStyle({'model':1}, {'stick': {'colorscheme':'yellowCarbon'}})
        view.zoomTo()
        view.show()
        
            