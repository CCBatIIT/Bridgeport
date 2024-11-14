import textwrap, sys, os
import numpy as np
from pdbfixer import PDBFixer
from openbabel import openbabel
#OpenMM
from openmm.app import *
from openmm import *
from openmm import unit
from datetime import datetime
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
from openmm.app.pdbfile import PDBFile
import MDAnalysis as mda
import mdtraj as md




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
    
    def __init__(self, pdb_path, working_dir: str, pH=7, env='MEM', ion_strength=0.15, verbose: bool=False):
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
        self.verbose = verbose
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Welcome to ProteinPreparer', flush=True)

    def main(self):
        
        # Protonate (with pdb2pqr30 which also adds missing atoms, but not missing residues)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protonating protein with pdb2pqr30', flush=True)
        self.H_pdb = self._protonate_with_pdb2pqr(at_pH=self.pH)

        # Check completion
        if os.path.exists(self.H_pdb):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', self.H_pdb, flush=True)
        
        #Create Environment
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Creating environment with pdbfixer', flush=True)
        self._run_PDBFixer(mode=self.env, padding=1.5, ionicStrength=self.ion)

        # Check completion
        if os.path.exists(self.env_pdb):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', self.env_pdb, flush=True)


        return self.env_pdb
        

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
        self.H_pdb = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
        protein_pqr_path = os.path.join(self.working_dir, self.prot_name + '.pqr')
        # my_cmd = f'pdb2pqr30 --ff AMBER --log-level CRITICAL --nodebump --keep-chain --ffout AMBER --pdb-output {self.H_pdb} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {self.H_pdb} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        print('Protanting using command line')
        print(f'Running {my_cmd}')
        exit_status = os.system(my_cmd)
        # print(f'Done with exit status {exit_status}')
        return self.H_pdb

    def _protonate_with_PDBFixer(self, at_pH=7):
        if not hasattr(self, "protein_H_path"):
            H_pdb_path = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
            if os.path.exists(H_pdb_path):
                self.H_pdb = H_pdb_path
            else:
                self.H_pdb = self.receptor_path
        fixer = PDBFixer(self.H_pdb)
        # fixer.findMissingResidues()
        # fixer.findMissingAtoms()
        # print('!!!missingTerminals', fixer.missingAtoms)
        # fixer.addMissingAtoms()
        fixer.addMissingHydrogens(at_pH)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.H_pdb, 'w'), keepIds=True)

        return self.H_pdb
    
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
        self.fixer = PDBFixer(self.H_pdb)

        if mode == 'MEM':
            self.fixer.addMembrane('POPC', minimumPadding=padding * unit.nanometer, ionicStrength=ionicStrength * unit.molar)
        elif mode == 'SOL':
            self.fixer.addSolvent(padding=padding * unit.nanometer, ionicStrength=ionicStrength * unit.molar)

        # ADD hydrogens
        self.fixer.addMissingHydrogens()

        # Trim
        self._trim_env()
        


    def _trim_env(self, padding: float=15):
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
    
        # Copy the fixer topology to a new topology,
        # excluding residues with atoms outside of the box
        def copyTopology(topology_o, residx_to_exclude = None):
          from openmm.app.topology import Topology
          topology_n = Topology()
        
          atom_indices = []
        
          chains = {}
          residues = {}
          atoms = {}
          # Add chains
          for c_o in topology_o.chains():
            c_n = topology_n.addChain()
            chains[c_o.id] = c_n
            # Add residues that are not excluded
            for r_o in c_o.residues():
              if not r_o.index in residx_to_exclude:
                r_n = topology_n.addResidue(r_o.name, c_n)
                residues[r_o.id] = r_n
                for a_o in r_o.atoms():
                  a_n = topology_n.addAtom(a_o.name, a_o.element, r_n)
                  atoms[a_o.id] = a_n
                  atom_indices.append(a_o.index)
          # Add bonds
          for b_o in topology_o.bonds():
            if (b_o.atom1.index in atom_indices) and (b_o.atom2.index in atom_indices):
              b_n = topology_n.addBond(atoms[b_o.atom1.id], atoms[b_o.atom2.id], \
                                       b_o.type, b_o.order)
          return (topology_n, atom_indices)
    
    
        
        u = mda.Universe(self.H_pdb)
        prot_sele = u.select_atoms('protein')
        max_coords = (np.array([prot_sele.positions[:,i].max() for i in range(3)]) + padding) * unit.angstroms
        min_coords = (np.array([prot_sele.positions[:,i].min() for i in range(3)]) - padding) * unit.angstroms
        
        # Identify residues with atoms outside of the box
        residx_out_of_box = set([a.residue.index for (a, p) in zip(self.fixer.topology.atoms(), self.fixer.positions.in_units_of(unit.angstroms)) \
         if (p<min_coords).any() or (p>max_coords).any()])
        
        (topology, atom_indices) = copyTopology(self.fixer.topology, \
                                                residx_to_exclude = residx_out_of_box)

        # Remove excess Na or Cl
        top = md.Topology.from_openmm(topology)
        positions = np.array(self.fixer.positions.value_in_unit(unit.angstroms))[atom_indices]
        if self.verbose:
            print('topology n_atoms', top.n_atoms, 'pos n_atoms', positions.shape[0])
        Na_atoms = top.select('name Na')
        Cl_atoms = top.select('name Cl')
        min_count = min(len(Na_atoms), len(Cl_atoms))
        remove_Na = Na_atoms[min_count:]
        remove_Cl = Cl_atoms[min_count:]
        remove_atoms = np.concatenate((remove_Na, remove_Cl))
        if self.verbose:
            print('remove_atoms', remove_atoms)
        for i, atom in enumerate(sorted(remove_atoms)):
            top.delete_atom_by_index(atom - i)
        positions = np.delete(positions, remove_atoms, axis=0)
        
        if self.verbose:
            print('topology n_atoms', top.n_atoms, 'pos n_atoms', positions.shape[0])

        
        topology = top.to_openmm()
        topology.setUnitCellDimensions(max_coords - min_coords)

        # Save file
        if not hasattr(self, "env_pdb"):
            self.env_pdb = os.path.join(self.working_dir, self.prot_name + f'_env.pdb')
            
        with open(self.env_pdb, "w") as F:
          PDBFile.writeFile(topology, positions, F, keepIds=True)

    
            

    
            
            