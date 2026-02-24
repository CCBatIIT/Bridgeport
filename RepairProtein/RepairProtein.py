import shutil, os, sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
import modeller
from modeller import *
from modeller.automodel import *
import mdtraj as md
import numpy as np
import MDAnalysis as mda
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from datetime import datetime
from typing import List
modeller.log.none()

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
  
    _align_sequences(self): 
        Aligns the template and target sequences to identify missing or mutated residues.
  
    _build_homology_model(self, nstd_resids): 
        Constructs a homology model using UCSF Modeller, incorporating non-standard residues if specified.
  
    _optimize_loops(self, loops): 
        Optimizes specified loop regions within the protein model.
    """


    
    def __init__(self, pdb_fn: str, fasta_fn: str, mutated_resids: List[int]=None, working_dir: str='./'):
        """
        Initialize RepairProtein object.

        Parameters:
        -----------
            pdb_fn (str):
                String path to .pdb file to repair.
            
            fasta_fn (str):
                String path to .fasta file that contains sequence to use as a template to repair the protein .pdb.     

            mutated_resids (List[int]):
                List of resids that are engineered mutations in the input .pdb. RepairProtein will automatically discard those residues and rewrite .pdb file to ease the identification of missing residues. Default is None. 

            working_dir (str):
                String path to working directory where all intermediate files made by UCSF modeller will be stored. Default is current working directory. 
        """

        # Initialize variables
        self.pdb_fn = pdb_fn
        self.fasta_fn = fasta_fn
        self.fasta_name = fasta_fn.split('/')[-1].split('.')[0]
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

        
        if mutated_resids != None:
            traj = md.load_pdb(self.pdb_fn)
            top = traj.topology
            self.mdtraj_resids = [top.residue(i).resSeq for i in range(top.n_residues)]
            mutated_resids = [self.mdtraj_resids.index(resid) for resid in mutated_resids]
            sele = top.select(f'not resid {" ".join([str(i) for i in mutated_resids])}')
            traj.atom_slice(sele).save_pdb(self.pdb_fn)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Removed mutated residues with resids:', mutated_resids, 'from', self.pdb_fn, flush=True)

        
        shutil.copy(self.pdb_fn, os.path.join(self.working_dir, self.name + '.pdb'))



    
    def run(self, pdb_out_fn: str, secondary_template_pdb: str=None, tails: List=False, nstd_resids: List=None, loops: List=False, verbose: bool=False, align_after: bool=True, cyclic: bool=False):
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
        self.cyclic = cyclic

        print('\n\n\n', 'CYCLIC =', self.cyclic, '\n\n\n')
        
        if secondary_template_pdb is not None:
            self.secondary_template_pdb = secondary_template_pdb
            self.secondary_name = self.secondary_template_pdb.split('/')[-1].split('.')[0]
            shutil.copy(self.secondary_template_pdb, os.path.join(self.working_dir, self.secondary_name + '.pdb'))

        # Make a copy for alignment purposes
        temp_pdb = os.path.join(os.path.dirname(self.pdb_fn), os.path.basename(self.pdb_fn).split('.')[0] + '_temp.pdb')
        shutil.copy(self.pdb_fn, temp_pdb)
        
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
        if loops != False:
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

        
        # Fix missing residues if cutting tails created improper terminals
        if not self.cyclic:
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

        # Alignment correction
        if align_after:
            u = mda.Universe(self.pdb_out_fn)
            resids = u.atoms.resids
            ref_u = mda.Universe(temp_pdb)
            ref_resids = ref_u.atoms.resids
            matching_resids = np.intersect1d(resids, ref_resids)
            b, a = mda.analysis.align.alignto(u, ref_u, select=f'name CA and resid {" ".join(str(r) for r in matching_resids)}')
            u.atoms.write(self.pdb_out_fn)
            os.remove(temp_pdb)
            
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Moved protein from', b, 'to', a, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protein Repaired. Output written to:', self.pdb_out_fn, flush=True)

    
            
    def _align_sequences(self):
        """
        Write the necessary alignment file for Modeller to build the appropriate residues. 
        """

        # Create objs
        env = modeller.Environ()
        aln = modeller.Alignment(env)

        # Add target sequence
        try:
            aln.append(file=self.fasta_fn, align_codes=(self.fasta_name))
        except:
            print(open(self.fasta_fn, 'r').readlines())
            raise Exception(f'Could not find code {self.fasta_name} in {self.fasta_fn}. Contents printed above')

        # Add pdb
        m = modeller.Model(env, file=self.pdb_fn)
        aln.append_model(m, align_codes=(self.name))

        # Add secondary_template
        if hasattr(self, 'secondary_template_pdb'):
            m = modeller.Model(env, file=self.secondary_template_pdb)
            aln.append_model(m, align_codes=(self.secondary_name))

        # Align
        aln.malign()
        self.ali_fn = os.path.join(self.working_dir, f'{self.fasta_name}.ali')
        aln.write(file=self.ali_fn, alignment_format='PIR')
        aln.write(file= os.path.join(self.working_dir, f'{self.fasta_name}.pap'), alignment_format='PAP')


    
    def _build_homology_model(self, nstd_resids):
        """
        Build a homology model with Modeller.AutoModel
        """

        if hasattr(self, 'secondary_template_pdb'):
            self.model = modeller.automodel.AutoModel(self.env, 
                                         sequence=self.fasta_name,
                                         knowns=(self.name, self.secondary_name),
                                         alnfile=f'{self.fasta_name}.ali')

        elif self.cyclic:
            class CyclicModel(AutoModel):
                def special_patches(self, aln):
                    # Link between last residue (-1) and first (0) to make chain cyclic:
                    self.patch(residue_type='LINK', residues=(self.residues[-1], self.residues[0]))

            self.model = CyclicModel(self.env, 
                                     sequence=self.fasta_name,
                                     knowns=(self.name),
                                     alnfile=f'{self.fasta_name}.ali')
            
        else:
            self.model = modeller.automodel.AutoModel(self.env, 
                                             sequence=self.fasta_name,
                                             knowns=(self.name),
                                             alnfile=f'{self.fasta_name}.ali')
            
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




