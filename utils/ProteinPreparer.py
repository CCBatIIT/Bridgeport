import textwrap, sys, os
import numpy as np
from pdbfixer import PDBFixer
from openbabel import openbabel
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *
from datetime import datetime



class ProteinPreparer():
    """
    The purpose of this class is to prepare a crystal structure for molecular simulation. Steps include
        Addition of Missing atoms and side chains
        Protonation at user specified pH
        Creation of environment (Membrane and Solvent, Solvet) at user specified ion conc
    
    Attributes:
    ------------
        receptor_path (str): 
            The path to the PDB file that contains the protein structure intended for preparation.
        
        prot_name (str): 
            The name of the protein, extracted from the input file path, which is used in naming the output files.
        
        working_dir (str): 
            The directory designated for storing all the intermediate and output files.
        
        pH (float): 
            Specifies the pH level at which the protein is to be protonated; the default value is 7.
        
        env (str): 
            Defines the type of environment to be created around the protein; 'MEM' indicates a combination of a membrane and solvent, while 'SOL' specifies a solvent-only environment.
            
        ion (float): 
            The ionic strength of the solvent, measured in molar units, with a default value of 0.15 M NaCl.

    Methods:
    ---------
        init(pdb_path, working_dir: str, pH=7, env='MEM', ion_strength=0.15): 
            Initializes the ProteinPreparer class with specified parameters for protein preparation, including the path to the PDB file, working directory, pH level for protonation, environmental setup (membrane and solvent or solvent only), and ion strength.
        
        main(): 
            Coordinates the main workflow for preparing the protein structure for molecular simulation. This includes steps for protonation, adding missing atoms and side chains, and setting up the simulation environment as specified by the user.
        
        _protonate_with_pdb2pqr(at_pH=7): 
            Protonates the protein structure at the specified pH using pdb2pqr30, which also adds missing atoms but not missing residues. Outputs file paths to the protonated pdb and pqr files.
        
        _protonate_with_PDBFixer(at_pH=7): 
            An alternative method to protonate the protein using PDBFixer, which can add missing hydrogens at the specified pH. This method is used if pdb2pqr30 is not employed or for additional protonation adjustments.
        
        _run_PDBFixer(mode: str = "MEM", out_file_fn: str = None, padding = 1.5, ionicStrength = 0.15): 
            Generates a solvated and possibly membrane-added system based on the specified mode. It uses PDBFixer to create an environment around the protein, either with just solvent or with both membrane and solvent, according to the user's choice.
    """
    
    def __init__(self, pdb_path, working_dir: str, pH=7, env='MEM', ion_strength=0.15):
        """
        Parameters:
            pdb_path: string path to protein structure file
            pH: default 7: pH to protonate at
            env: default 'MEM': 'MEM' for explicit membrane and solvent
                                'SOL' for explicit solvent only
                                no other modes supported
        """
        self.receptor_path = pdb_path
        self.prot_name = self.receptor_path.split('.')[0]
        try:
            self.prot_name = self.prot_name.split('/')[-1]
        except:
            pass
        self.working_dir = working_dir
        self.pH = pH
        self.env = env
        self.ion = ion_strength
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Welcome to ProteinPreparer', flush=True)

    def main(self):
        # Protonate (with pdb2pqr30 which also adds missing atoms, but not missing residues)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protonating protein with pdb2pqr30', flush=True)
        protein_protonated = self._protonate_with_pdb2pqr(at_pH=self.pH)
        if os.path.exists(protein_protonated):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', protein_protonated, flush=True)
        
        #Create Environment
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Creating environment with pdbfixer', flush=True)
        protein_solvated = self._run_PDBFixer(mode=self.env,
                                              padding=1.5,
                                              ionicStrength=self.ion)
        if os.path.exists(protein_solvated):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', protein_solvated, flush=True)

        return protein_solvated
        

    def _protonate_with_pdb2pqr(self, at_pH=7):
        """
        Protonates the given structure using pdb2pqr30 on the linux command line
        
        Parameters:
        -----------
            at_pH: pH for protonation (default 7)
            
        Returns:
        --------
            2-tuple = (protein_H_fn, protein_pqr_fn); file paths (as strings) to the protonated pdb and pqr file
        """
        self.protein_H_path = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
        protein_pqr_path = os.path.join(self.working_dir, self.prot_name + '.pqr')
        # my_cmd = f'pdb2pqr30 --ff AMBER --log-level CRITICAL --nodebump --keep-chain --ffout AMBER --pdb-output {self.protein_H_path} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {self.protein_H_path} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        print('Protanting using command line')
        print(f'Running {my_cmd}')
        exit_status = os.system(my_cmd)
        # print(f'Done with exit status {exit_status}')
        return self.protein_H_path

    def _protonate_with_PDBFixer(self, at_pH=7):
        if not hasattr(self, "protein_H_path"):
            H_pdb_path = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
            if os.path.exists(H_pdb_path):
                self.protein_H_path = H_pdb_path
            else:
                self.protein_H_path = self.receptor_path
        fixer = PDBFixer(self.protein_H_path)
        # fixer.findMissingResidues()
        # fixer.findMissingAtoms()
        # print('!!!missingTerminals', fixer.missingAtoms)
        # fixer.addMissingAtoms()
        fixer.addMissingHydrogens(at_pH)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.protein_H_path, 'w'), keepIds=True)

        return self.protein_H_path
    
    def _run_PDBFixer(self,
                      mode: str = "MEM",
                      out_file_fn: str = None,
                      padding = 1.5,
                      ionicStrength = 0.15):
        """
        Generates a solvated and membrane-added system (depending on MODE)
        MODE = 'MEM' for membrane and solvent
        MODE = 'SOL' for solvent only
        Parameters:
            mode: string: must be in ['MEM', 'SOL']
            solvated_file_fn: Filename to save solvated system; default is to add '_solvated' between the body and extension of the input file name
            padding: float or int: minimum nanometer distance between the boundary and any atoms in the input.  Default 1.5 nm = 15 A
            ionicStrength: float (not int) : molar strength of ionic solution. Default 0.15 M = 150 mmol
        Returns:
            solvated_file_fn: the filename of the solvated output
        """
        assert mode in ['MEM', 'SOL']
        fixer = PDBFixer(self.protein_H_path)

        if mode == 'MEM':
            fixer.addMembrane('POPC', minimumPadding=padding * nanometer, ionicStrength=ionicStrength * molar)
        elif mode == 'SOL':
            fixer.addSolvent(padding=padding * nanometer, ionicStrength=ionicStrength * molar)

        fixer.addMissingHydrogens()
        
        # ADD PDBFixer hydrogens and parsing crystal structures (Hydrogens with pdb2pqr30 at the moment)
        if out_file_fn is None:
            out_file_fn = os.path.join(self.working_dir, self.prot_name + f'_env.pdb')
        
        with open(out_file_fn, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        return out_file_fn
        
        