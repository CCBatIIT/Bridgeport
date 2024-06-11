import shutil, os
from modeller import *
from modeller.automodel import *
from file_handling import parse_sequence, fasta_to_pir, pdb_to_pir
from SequenceWrapper import SequenceWrapper as SeqWrap
import mdtraj as md
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import mdtraj as md
from datetime import datetime
from typing import List

class RepairProtein():
    """
    The RepairProtein class is designed to repair incomplete or damaged protein structures, providing tools for the addition, removal, or modification of atomic details to produce a corrected structure suitable for simulation. Leveraging the capabilities of UCSF Modeller, this class facilitates homology modeling, loop optimization, and the maintenance of non-standard residues within the protein model. Additionally, it incorporates secondary structure templates to enhance model accuracy.
    
    Features:
    ---------
        - Automated repair of missing and mutated residues in protein structures.
        - Utilizes template sequences from FASTA files for accurate remodeling.
        - Supports optimization of loop regions for improved structure prediction.
        - Capable of preserving non-standard residues during the repair process.
        - Integrates with Modeller and OpenMM for a comprehensive structure repair workflow.
    
    Attributes:
    ------------
        pdb_fn (str): 
            Path to the input .pdb file to be repaired.
       
        fasta_fn (str): 
            Path to the .fasta file containing the template sequence.
       
        working_dir (str): 
            Directory for storing intermediate files created during the repair process. Defaults to the current directory.
      
        name (str): 
            Identifier derived from the input .pdb file, excluding the file extension.
      
        pdb_out_fn (str): 
            Path for saving the repaired .pdb file.
    
    
    Methods
    init(self, pdb_fn: str, fasta_fn: str, working_dir: str='./'): 
        Initializes the repair process by setting up file paths and directories.
  
    run(self, pdb_out_fn: str, tails: List=False, nstd_resids: List=None, loops: List=False, verbose: bool=False): 
        Executes the repair, including homology modeling and optional loop optimization. Allows for verbose output detailing missing and mutated residues.
   
    run_with_secondary(self, secondary_template_pdb: str, pdb_out_fn: str, tails: bool=False, loops: List=None): 
        Executes the repair using a secondary structure template to guide the modeling of missing secondary structures.
  
    _get_temp_seq(self): 
        Parses the template sequence from the provided .fasta file.
  
    _get_tar_seq(self): 
        Extracts the target sequence from the provided .pdb file, considering non-standard residues.
  
    _align_sequences(self): 
        Aligns the template and target sequences to identify missing or mutated residues.
  
    _build_homology_model(self, nstd_resids): 
        Constructs a homology model using UCSF Modeller, incorporating non-standard residues if specified.
  
    _optimize_loops(self, loops): 
        Optimizes specified loop regions within the protein model.
    """

    def __init__(self, pdb_fn: str, fasta_fn: str, working_dir: str='./'):
        """
        Initialize RepairProtein object.

        Parameters:
        -----------
            pdb_fn (str):
                String path to .pdb file to repair.
            
            fasta_fn (str):
                String path to .fasta file that contains sequence to use as a template to repair the protein .pdb.     

            working_dir (str):
                String path to working directory where all intermediate files made by UCSF modeller will be stored. Default is current working directory. 
        """

        # Initialize variables
        self.pdb_fn = pdb_fn
        self.fasta_fn = fasta_fn
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        self.name = self.pdb_fn.split('.pdb')[0]
        try:
            self.name = self.name.split('/')[-1]
        except:
            pass
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Welcome to RepairProtein', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protein to repair:', self.pdb_fn, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Template sequence:', self.fasta_fn, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Modeller intermediates will be written to:', self.working_dir, flush=True)

    def run(self, pdb_out_fn: str, tails: List=False, nstd_resids: List=None, loops: List=False, verbose: bool=False):
        """
        Run the remodelling.

        Parameters:
        -----------
            pdb_out_fn (str):
                String path to write repaired .pdb file. 

            tails (List):
                List of indices to parse the extra tails. EX: [30, 479].

            nstd_resids (List):
                If list is provided then nonstandard residues at these indices (0-indexed) will be conserved from input model to output structure.

            loops (2D-list):
                If list is provided then loops will be optimized. Should be in format [[resid_1, resid_2], ...] to represent the loops.

            verbose (bool):
                If true, show missing and mutated residues after each iteration of sequence alignment. Default is False. 

        """
        # Attributes
        self.pdb_out_fn = pdb_out_fn
        self.verbose = verbose
        self.nstd_resids = nstd_resids

        # Parse template sequence from .fasta
        self._get_temp_seq()

        # Parse target sequence from .pdb
        self._get_tar_seq()
        
        # Find mutated/missing residues
        self._align_sequences()

        # Model 
        cwd = os.getcwd()
        os.chdir(self.working_dir)
        self.env = Environ()
        self.env.io.atom_files_directory = ['.', self.working_dir]
        if nstd_resids != None:
            self.env.io.hetatm=True
        self._build_homology_model(nstd_resids=self.nstd_resids)

        # Fix loops
        if loops != False and loops != None:
            self._optimize_loops(loops)

        os.chdir(cwd)

        # Delete tails if necessary
        if tails != False:
            if tails == True:
                pass
            else:
                traj = md.load_pdb(self.pdb_out_fn)
                top = traj.topology
                resid_range = ' '.join(str(i) for i in range(tails[0], tails[1]))
                sele = top.select(f'resid {resid_range}')
                traj = traj.atom_slice(sele)
                traj.save_pdb(self.pdb_out_fn)
        else:
            traj = md.load_pdb(self.pdb_out_fn)
            top = traj.topology
            resid_range = ' '.join(str(i) for i in range(self.term_residues[0], self.term_residues[1]))
            sele = top.select(f'resid {resid_range}')
            traj = traj.atom_slice(sele)
            traj.save_pdb(self.pdb_out_fn)
        
        # Fix missing residues if cutting tails created improper terminals
        fixer = PDBFixer(self.pdb_out_fn)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.pdb_out_fn, 'w'), keepIds=True)

        # Reinsert CRYS entry
        crys_line = ''
        with open(self.pdb_fn, 'r') as f:
            for line in f:
                if line.find('CRYST1') != -1:
                    crys_line = f'{line}'
        f.close()

        with open(self.pdb_out_fn, 'r') as f:
            pdb_lines = f.readlines()
        f.close()

        pdb_lines[0] = crys_line
        with open(self.pdb_out_fn, 'w') as f:
            for line in pdb_lines:
                f.write(line)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protein Repaired. Output written to:', self.pdb_out_fn, flush=True)

    def run_with_secondary(self, secondary_template_pdb: str, pdb_out_fn: str, tails: bool=False, loops: List=None, nstd_resids: List=None, verbose: bool=False):
        """
        Run the remodelling using a secondary structure to appropriately model secondary structure that is missing in input file. 

        Parameters:
        -----------
            secondary_template_pdb (str):
                String path to secondary template to use during modelling. 
        
            pdb_out_fn (str):
                String path to write repaired .pdb file. 

            tails (List):
                List of indices to parse the extra tails. EX: [30, 479].

            nstd_resids (List):
                If list is provided then nonstandard residues at these indices (0-indexed) will be conserved from input model to output structure.

            loops (2D-list):
                If list is provided then loops will be optimized. Should be in format [[resid_1, resid_2], ...] to represent the loops.

            verbose (bool):
                If true, show missing and mutated residues after each iteration of sequence alignment. Default is False.
                
            nstd_resids (List):
                If list is provided then nonstandard residues at these indices (0-indexed) will be conserved from input model to output structure.
        """
        # Attributes
        self.pdb_out_fn = pdb_out_fn
        self.secondary_template_pdb = secondary_template_pdb
        self.secondary_name = self.secondary_template_pdb.split('/')[-1].split('.')[0]
        self.nstd_resids = nstd_resids

        # Parse template sequence from .fasta
        self._get_temp_seq()

        # Parse target sequence from .pdb
        self._get_tar_seq()
        
        # Find mutated/missing residues
        self._align_sequences()

        # Model 
        cwd = os.getcwd()
        os.chdir(self.working_dir)
        self.env = Environ()
        self.env.io.atom_files_directory = ['.', self.working_dir]
        self._build_homology_model(self.nstd_resids)
        
        # Fix loops
        if loops != None:
            self._optimize_loops(loops)

        os.chdir(cwd)

        # Delete tails if necessary
        if tails != False:
            if tails == True:
                pass
            else:
                traj = md.load_pdb(self.pdb_out_fn)
                top = traj.topology
                resid_range = ' '.join(str(i) for i in range(tails[0], tails[1]))
                sele = top.select(f'resid {resid_range}')
                traj = traj.atom_slice(sele)
                traj.save_pdb(self.pdb_out_fn)
        else:
            traj = md.load_pdb(self.pdb_out_fn)
            top = traj.topology
            resid_range = ' '.join(str(i) for i in range(self.term_residues[0], self.term_residues[1]))
            sele = top.select(f'resid {resid_range}')
            traj = traj.atom_slice(sele)
            traj.save_pdb(self.pdb_out_fn)
        
        # Fix missing residues if cutting tails created improper terminals
        fixer = PDBFixer(self.pdb_out_fn)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.pdb_out_fn, 'w'), keepIds=True)

        # Reinsert CRYS entry
        crys_line = ''
        with open(self.pdb_fn, 'r') as f:
            for line in f:
                if line.find('CRYST1') != -1:
                    crys_line = f'{line}'
        f.close()

        with open(self.pdb_out_fn, 'r') as f:
            pdb_lines = f.readlines()
        f.close()

        pdb_lines[0] = crys_line
        with open(self.pdb_out_fn, 'w') as f:
            for line in pdb_lines:
                f.write(line)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protein Repaired. Output written to:', self.pdb_out_fn, flush=True)
            
    def _get_temp_seq(self):
        """
        Parse the template sequence from provided .fasta file
        """
        self.temp_pir_fn = self.working_dir + '/' + self.name + '.pir'
        fasta_to_pir(self.fasta_fn, self.temp_pir_fn)
        self.temp_seq = parse_sequence(self.temp_pir_fn)

    def _get_tar_seq(self):
        """
        Parse the target sequence from provided .pdb file
        """
        self.tar_pir_fn = self.working_dir + '/' + self.name + '.pir'
        shutil.copy(self.pdb_fn, self.working_dir + '/' + self.pdb_fn.split('/')[-1])
        pdb_to_pir(self.name, self.working_dir)
        self.tar_seq = parse_sequence(self.tar_pir_fn)
        if self.nstd_resids != None:
            tar_seq_splt = self.tar_seq.split()
            for nstd_resid in self.nstd_resids:
                tar_seq_splt.insert(nstd_resid, '.')
            self.tar_seq = ''.join(tar_seq_splt)
        # print('!!!nstd_resids')
        # print('!!!tar_seq', self.tar_seq)
        if hasattr(self, "secondary_template_pdb"):
            shutil.copy(self.secondary_template_pdb, self.working_dir + '/' + self.secondary_template_pdb.split('/')[-1])
            pdb_to_pir(self.secondary_name, self.working_dir)
            self.secondary_pir_fn = self.working_dir + '/' + self.secondary_name + '.pir'
            self.secondary_seq = parse_sequence(self.secondary_pir_fn)
        else:
            self.secondary_pir_fn = None
            
    def _align_sequences(self):
        """
        Write the necessary alignment file for Modeller to build the appropriate residues. 
        """
        if hasattr(self, "secondary_seq"):
            sw = SeqWrap(self.temp_seq, self.tar_seq, self.secondary_seq)
        else:
            sw = SeqWrap(self.temp_seq, self.tar_seq)

        sw.find_missing_residues(verbose=False)
        traj = md.load_pdb(self.pdb_fn)
        self.mutated_residues = sw.mutated_residues
        counter = 0
        while len(self.mutated_residues) > 0:
            # Remove mutation from input.pdb
            traj = md.load_pdb(self.pdb_fn)
            mutated_atoms = traj.topology.select('resid '+ str(" ".join(self.mutated_residues[:,0])))
            sele = [i for i in range(traj.topology.n_atoms) if i not in mutated_atoms]
            traj = traj.atom_slice(sele)
            traj.save_pdb(self.pdb_fn)

            # Reparse target sequence from new .pdb
            shutil.copy(self.pdb_fn, self.working_dir + '/' + self.pdb_fn.split('/')[-1])
            pdb_to_pir(self.name, self.working_dir)
            self.tar_seq = parse_sequence(self.tar_pir_fn)

            # Find missing
            if hasattr(self, "secondary_seq"):
                sw = SeqWrap(self.temp_seq, self.tar_seq, self.secondary_seq)
            else:
                sw = SeqWrap(self.temp_seq, self.tar_seq)  
                sw.find_missing_residues(verbose=self.verbose)

            self.mutated_residues = sw.mutated_residues

            counter += 1
            if counter == 3:
                raise RuntimeError

        self.missing_residues = sw.missing_residues
        self.term_residues = sw.term_residues
        self.ali_fn = self.working_dir + '/' + self.name + '.ali'
        sw.write_alignment_file(self.ali_fn, self.temp_pir_fn, self.secondary_pir_fn)


    def _build_homology_model(self, nstd_resids):
        """
        Build a homology model with Modeller.AutoModel
        """
        class MyModel(AutoModel):
            def select_atoms(self):
                return Selection(self.residue_range('1:A', '1:A'))
                    
        if hasattr(self, "secondary_name"):
            self.model = MyModel(self.env, 
                        alnfile = self.ali_fn, 
                        knowns=[self.name, self.secondary_name], 
                        sequence=self.name + '_fill')
        else:
            self.model = MyModel(self.env, 
                            alnfile = self.ali_fn, 
                            knowns=self.name, 
                            sequence=self.name + '_fill')
            
        self.model.starting_model = 1
        self.model.ending_model = 1
        self.model.make()
        self.model.write(self.pdb_out_fn, no_ter=True)

    def _optimize_loops(self, loops):
        """
        Optimize loops of homology model with Modeller.LoopModel
        """
        class MyLoop(LoopModel):
            def select_loop_atoms(self):
                sel = Selection()
                for loop in loops:
                    sel.add(self.residue_range(f'{loop[0]}:A', f'{loop[1]}:A'))
                return sel

        self.loopmodel = MyLoop(self.env, 
                        inimodel=self.model.outputs[0]['name'],
                        sequence=self.name+'_fill',
                        loop_assess_methods=assess.DOPE)
        
        self.loopmodel.loop.starting_model = 1
        self.loopmodel.loop.ending_model = 1
        self.loopmodel.md_level = refine.fast
        self.loopmodel.make()

        # Move UCSF modeller output to desired location
        self.loopmodel.write(self.pdb_out_fn, no_ter=True)




