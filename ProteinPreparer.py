import textwrap, sys, os
import numpy as np
from pdbfixer import PDBFixer
from openbabel import openbabel
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *


class ProteinPreparer():
    """
    The purpose of this class is to prepare a crystal structure for molecular simulation. Steps include
        Addition of Missing atoms and side chains
        Protonation at user specified pH
        Creation of environment (Membrane and Solvent, Solvet) at user specified ion conc
    """
    def __init__(self, protein_pdb_fn, pH=7, env='MEM', ion_strength=0.15):
        """
        Parameters:
            protein_pdb_fn: protein structure file
            pH: default 7: pH to protonate at
            env: default 'MEM': 'MEM' for explicit membrane and solvent
                                'SOL' for explicit solvent only
                                no other modes supported
        """
        self.receptor_path = protein_pdb_fn
        self.pH = pH
        self.env = env
        self.ion = ion_strength

    def main(self):
        # Protonate (with pdb2pqr30 which also adds missing atoms, but not missing residues)
        protein_protonated, protein_pqr = self._protonate_with_pdb2pqr(self.receptor_path,
                                                                       at_pH=self.pH)
        print(protein_protonated, os.path.isfile(protein_protonated))
        print(protein_pqr, os.path.isfile(protein_pqr))
        #Create Environment
        protein_solvated = self._run_PDBFixer(protein_protonated,
                                              mode=self.env,
                                              padding=1.5,
                                              ionicStrength=self.ion)
        print(protein_solvated, os.path.isfile(protein_solvated))
        return protein_solvated
        

    def _protonate_with_pdb2pqr(self,
                                protein_fn: str,
                                protein_H_fn: str = None,
                                at_pH=7):
        """
        Protonates the given structure using pdb2pqr30 on the linux command line
        Parameters:
            protein_fn: structure to be protonated
            protein_H_fn: filepath for protonated receptor (as pdb)
                          pqr output of pdb2pqr is inferred as protein_H_fn with a pqr extension instead of pdb
            at_pH: pH for protonation (default 7)
        Returns:
            2-tuple = (protein_H_fn, protein_pqr_fn); file paths (as strings) to the protonated pdb and pqr file
        """
        if protein_H_fn is None:
            protein_H_fn = os.path.splitext(protein_fn)[0] + '_H' + os.path.splitext(protein_fn)[-1]
        protein_pqr_fn = os.path.splitext(protein_H_fn)[0] + '.pqr'
        my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {protein_H_fn} --with-ph {at_pH} {protein_fn} {protein_pqr_fn}'
        print('Protanting using command line')
        print(f'Running {my_cmd}')
        exit_status = os.system(my_cmd)
        print(f'Done with exit status {exit_status}')
        return protein_H_fn, protein_pqr_fn

    def _run_PDBFixer(self,
                      in_pdbfile: str,
                      mode: str = "MEM",
                      out_file_fn: str = None,
                      padding = 1.5,
                      ionicStrength = 0.15):
        """
        Generates a solvated and membrane-added system (depending on MODE)
        MODE = 'MEM' for membrane and solvent
        MODE = 'SOL' for solvent only
        Parameters:
            in_pdbfile: the structure to be solvated
            mode: string: must be in ['MEM', 'SOL']
            solvated_file_fn: Filename to save solvated system; default is to add '_solvated' between the body and extension of the input file name
            padding: float or int: minimum nanometer distance between the boundary and any atoms in the input.  Default 1.5 nm = 15 A
            ionicStrength: float (not int) : molar strength of ionic solution. Default 0.15 M = 150 mmol
        Returns:
            solvated_file_fn: the filename of the solvated output
        """
        assert mode in ['MEM', 'SOL']
        fixer = PDBFixer(in_pdbfile)

        if mode == 'MEM':
            fixer.addMembrane('POPC', minimumPadding=padding * nanometer, ionicStrength=ionicStrength * molar)
        elif mode == 'SOL':
            fixer.addSolvent(padding=padding * nanometer, ionicStrength=ionicStrength * molar)

        fixer.addMissingHydrogens()
        
        # ADD PDBFixer hydrogens and parsing crystal structures (Hydrogens with pdb2pqr30 at the moment)
        if out_file_fn is None:
            out_file_fn = os.path.splitext(in_pdbfile)[0] + f'_{mode}' + os.path.splitext(in_pdbfile)[-1]
        
        with open(out_file_fn, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return out_file_fn
        
        