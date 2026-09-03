import shutil, os, sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
import modeller
from modeller import *
from modeller.automodel import *
import mdtraj as md
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.align
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

    _find_template_gaps(self):
        Identifies which residues are actually missing from the input .pdb, so that everything else can be
        held fixed at its experimental coordinates.

    _trim_secondary_template(self):
        Cuts the secondary template down to the gaps of the input .pdb so it cannot override resolved regions.

    _superpose_to_input(self, traj, primary, secondary):
        Puts the secondary template in the frame of the input .pdb before it is trimmed.
    """

    # Heavy atom count of each standard residue, used to detect residues that are present in the
    # input .pdb but are missing sidechain atoms. Those residues cannot be held fixed because
    # Modeller has to build the absent atoms. A C-terminal OXT makes the count exceed the entry
    # here, so only counts *below* these values indicate an incomplete residue.
    _HEAVY_ATOM_COUNTS = {'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6,
                          'GLN': 9, 'GLU': 9, 'GLY': 4, 'HIS': 10, 'ILE': 8,
                          'LEU': 8, 'LYS': 9, 'MET': 8, 'PHE': 11, 'PRO': 7,
                          'SER': 6, 'THR': 7, 'TRP': 14, 'TYR': 12, 'VAL': 7}



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



    
    def run(self, pdb_out_fn: str, secondary_template_pdb: str=None, tails: List=False, nstd_resids: List=None, loops: List=False, verbose: bool=False, align_after: bool=True, cyclic: bool=False, preserve_resolved: bool=False, secondary_template_gaps_only: bool=False):
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

            preserve_resolved (bool):
                If true, only the residues that are actually absent from the input .pdb are optimized by Modeller.
                Every residue that has coordinates in the input .pdb is held fixed, so experimentally resolved
                regions - including disordered coils such as receptor N-termini - come through untouched.
                Residues that are present but missing sidechain atoms are optimized anyway, since Modeller has to
                build those atoms. Turn this on when the input .pdb resolves a flexible region whose conformation
                matters, such as a receptor N-terminus that contacts the ligand. Default is False.

            secondary_template_gaps_only (bool):
                If true, the secondary template is superposed onto the input .pdb and cut down to the residues
                that fall in gaps, so it can only contribute restraints where the input structure has nothing.
                Without this, Modeller derives restraints from both templates wherever they overlap, and a
                predicted model can pull resolved regions away from their experimental positions. Only has an
                effect when a secondary template is given. Default is False.

        """
        # Attributes
        self.pdb_out_fn = pdb_out_fn
        self.verbose = verbose
        self.nstd_resids = nstd_resids
        self.cyclic = cyclic
        self.preserve_resolved = preserve_resolved
        self.secondary_template_gaps_only = secondary_template_gaps_only

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

        # Restrict the secondary template to the gaps of the input structure. Only needs a fresh alignment in
        # the case where the secondary template was redundant and got dropped entirely.
        if hasattr(self, 'secondary_template_pdb') and self.secondary_template_gaps_only:
            if self._trim_secondary_template():
                self._align_sequences()

        # Work out which residues Modeller actually has to build
        self.model_gaps = None
        if self.preserve_resolved:
            self._find_template_gaps()

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



    def _read_pir(self, ali_fn):
        """
        Parse a PIR alignment file into a list of [code, description, sequence] entries. The trailing '*' that
        terminates each sequence is stripped and re-added by _write_pir.
        """
        blocks, code, desc, seq = [], None, None, []
        for line in open(ali_fn, 'r'):
            line = line.rstrip('\n')
            if line.startswith('>P1;'):
                if code is not None:
                    blocks.append([code, desc, ''.join(seq).rstrip('*')])
                code, desc, seq = line[len('>P1;'):].strip(), None, []
            elif code is None:
                continue
            elif desc is None:
                desc = line
            else:
                seq.append(line.strip())
        if code is not None:
            blocks.append([code, desc, ''.join(seq).rstrip('*')])

        return blocks


    def _write_pir(self, ali_fn, blocks):
        """
        Write [code, description, sequence] entries back out as a PIR alignment file.
        """
        with open(ali_fn, 'w') as f:
            for code, desc, seq in blocks:
                f.write(f'>P1;{code}\n{desc}\n')
                for i in range(0, len(seq), 75):
                    f.write(seq[i:i+75] + '\n')
                f.write('*\n\n')


    def _find_template_gaps(self):
        """
        Determine which residues Modeller actually has to build.

        A residue needs to be built when the input .pdb has no coordinates for it, or when it is present but
        missing heavy atoms. Everything else is resolved experimentally and is left alone.

        Sets:
        -----
            self.model_gaps (List[List[int]]):
                Contiguous [first, last] residue ranges, numbered against the template sequence in the .fasta
                (which is also the numbering of the model Modeller builds), that must be optimized. None if the
                alignment could not be read, in which case _build_homology_model optimizes everything.
        """
        self.model_gaps = None

        blocks = self._read_pir(self.ali_fn)
        seqs = {code: seq for code, _, seq in blocks}
        if self.fasta_name not in seqs or self.name not in seqs:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not locate target and template rows in', self.ali_fn, '- skipping gap detection', flush=True)
            return

        target, primary = seqs[self.fasta_name], seqs[self.name]

        # Position in the alignment -> residue number in the model Modeller will build
        resnum, pos_to_resnum = 0, {}
        for i, res in enumerate(target):
            if res != '-':
                resnum += 1
                pos_to_resnum[i] = resnum

        # A target residue is missing when the input .pdb contributes a gap at that column
        missing = [pos_to_resnum[i] for i in sorted(pos_to_resnum) if primary[i] == '-']

        # Residues that are present but lack heavy atoms still have to be optimized
        incomplete = self._find_incomplete_residues(primary, pos_to_resnum)

        to_build = sorted(set(missing) | set(incomplete))
        if len(to_build) == 0:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Input structure is complete, nothing to build', flush=True)
            self.model_gaps = []
        else:
            # Collapse into contiguous ranges
            ranges = [[to_build[0], to_build[0]]]
            for res in to_build[1:]:
                if res == ranges[-1][1] + 1:
                    ranges[-1][1] = res
                else:
                    ranges.append([res, res])
            self.model_gaps = ranges
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Residues to build:', ', '.join(f'{a}-{b}' if a != b else str(a) for a, b in ranges), flush=True)
            if len(incomplete) > 0:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + str(len(incomplete)), 'of those are present but missing heavy atoms:', incomplete, flush=True)

        if self.preserve_resolved:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Holding', len(pos_to_resnum) - len(to_build), 'resolved residues fixed at their input coordinates', flush=True)


    def _trim_secondary_template(self):
        """
        Cut the secondary template down to only the residues that fall in gaps of the input .pdb, and write the
        result over the working copy of the template.

        Modeller derives restraints from every template that covers a residue, so a predicted model spanning the
        whole sequence competes with - and can override - the experimental coordinates. Trimming it to the gaps
        leaves the input structure as the sole template everywhere it has coordinates.

        The alignment row of the trimmed template is edited in place rather than recomputed. A trimmed template
        is two or more distant fragments concatenated, so re-running the sequence alignment on it would place
        those fragments as one contiguous block in the wrong position.

        Returns:
        --------
            bool: True if the alignment needs to be rebuilt from scratch, which is only the case when the
                  secondary template turned out to be redundant and was dropped altogether.
        """
        blocks = self._read_pir(self.ali_fn)
        seqs = {code: seq for code, _, seq in blocks}
        if self.name not in seqs or self.secondary_name not in seqs:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not locate template rows in', self.ali_fn, '- leaving secondary template intact', flush=True)
            return False

        primary, secondary = seqs[self.name], seqs[self.secondary_name]

        # The n-th ungapped column of the secondary row is the n-th residue of the secondary .pdb
        keep, i_res = [], -1
        for i in range(len(secondary)):
            if secondary[i] == '-':
                continue
            i_res += 1
            if primary[i] == '-':
                keep.append(i_res)
        n_total = i_res + 1

        if len(keep) == n_total:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Secondary template', self.secondary_name, 'already only covers gaps', flush=True)
            return False

        if len(keep) == 0:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Secondary template', self.secondary_name, 'covers nothing the input structure is missing - dropping it', flush=True)
            delattr(self, 'secondary_template_pdb')
            return True

        trimmed_fn = os.path.join(self.working_dir, self.secondary_name + '.pdb')
        traj = md.load_pdb(self.secondary_template_pdb)
        if traj.topology.n_residues != n_total:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Alignment has', n_total, 'secondary template residues but', self.secondary_template_pdb, 'has', traj.topology.n_residues, '- leaving secondary template intact', flush=True)
            return False

        # Put the secondary template in the frame of the input structure before cutting it up. Modeller takes the
        # initial coordinates of a residue straight from whichever template covers it, so fragments left in the
        # predicted model's own frame start far from the residues they have to bond to, and the optimizer cannot
        # always pull them back once the rest of the model is held fixed.
        traj = self._superpose_to_input(traj, primary, secondary)

        sele = traj.topology.select('resid ' + ' '.join(str(i) for i in keep))
        traj.atom_slice(sele).save_pdb(trimmed_fn)
        self.secondary_template_pdb = trimmed_fn

        # Drop the discarded residues from the alignment row, and point the PIR header at the whole trimmed
        # file so that Modeller's residue count still matches the alignment.
        secondary = list(secondary)
        i_res = -1
        for i in range(len(secondary)):
            if secondary[i] == '-':
                continue
            i_res += 1
            if i_res not in keep:
                secondary[i] = '-'

        for block in blocks:
            if block[0] == self.secondary_name:
                block[1] = f'structureX:{self.secondary_name}:FIRST:@:LAST:@:::-1.00:-1.00'
                block[2] = ''.join(secondary)
        self._write_pir(self.ali_fn, blocks)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Trimmed secondary template', self.secondary_name, 'from', n_total, 'to', len(keep), 'residues so it only fills gaps in', self.name, flush=True)

        return False


    def _superpose_to_input(self, traj, primary, secondary):
        """
        Superpose the secondary template onto the input .pdb using the CA atoms of every residue the two
        templates share in the alignment. Returns the trajectory unchanged if the correspondence cannot be
        established or too few residues overlap to define a fit.

        Parameters:
        -----------
            traj (md.Trajectory):
                Secondary template.

            primary (str):
                Alignment row of the input .pdb.

            secondary (str):
                Alignment row of the secondary template.
        """
        try:
            input_traj = md.load_pdb(self.pdb_fn)
        except Exception as e:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not read', self.pdb_fn, 'to superpose secondary template:', e, flush=True)
            return traj

        input_res, sec_res = list(input_traj.topology.residues), list(traj.topology.residues)
        if len(input_res) != sum(1 for res in primary if res != '-'):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Cannot map', self.pdb_fn, 'onto the alignment - leaving secondary template in its own frame', flush=True)
            return traj

        sec_ca, input_ca = [], []
        i_input = i_sec = -1
        for i in range(len(primary)):
            if primary[i] != '-':
                i_input += 1
            if secondary[i] != '-':
                i_sec += 1
            if primary[i] == '-' or secondary[i] == '-':
                continue
            ca_input = [atom.index for atom in input_res[i_input].atoms if atom.name == 'CA']
            ca_sec = [atom.index for atom in sec_res[i_sec].atoms if atom.name == 'CA']
            if ca_input and ca_sec:
                input_ca.append(ca_input[0])
                sec_ca.append(ca_sec[0])

        if len(sec_ca) < 3:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Only', len(sec_ca), 'shared residues - leaving secondary template in its own frame', flush=True)
            return traj

        traj = traj.superpose(input_traj, atom_indices=sec_ca, ref_atom_indices=input_ca)
        rmsd = np.sqrt(np.mean(np.sum((traj.xyz[0, sec_ca] - input_traj.xyz[0, input_ca])**2, axis=1))) * 10
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Superposed secondary template onto input structure over', len(sec_ca), 'residues, CA RMSD', np.round(rmsd, 2), 'Angstrom', flush=True)

        return traj


    def _find_incomplete_residues(self, primary, pos_to_resnum):
        """
        Return the model residue numbers of residues that are present in the input .pdb but are missing heavy
        atoms. These cannot be held fixed because Modeller has to build the absent atoms.

        Parameters:
        -----------
            primary (str):
                Alignment row of the input .pdb.

            pos_to_resnum (dict):
                Mapping of alignment position to residue number in the model.
        """
        incomplete = []
        try:
            residues = [res for res in md.load_pdb(self.pdb_fn).topology.residues]
        except Exception as e:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Could not read', self.pdb_fn, 'to check for incomplete residues:', e, flush=True)
            return incomplete

        # The n-th ungapped column of the template row is the n-th residue of the input .pdb. Bail out rather
        # than guess if that correspondence does not hold (multiple chains, hetatms Modeller skipped, ...).
        n_aligned = sum(1 for res in primary if res != '-')
        if n_aligned != len(residues):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Alignment has', n_aligned, 'template residues but', self.pdb_fn, 'has', len(residues), '- skipping incomplete residue check', flush=True)
            return incomplete

        i_res = -1
        for i in range(len(primary)):
            if primary[i] == '-':
                continue
            i_res += 1
            if i not in pos_to_resnum:
                continue
            res = residues[i_res]
            n_expected = self._HEAVY_ATOM_COUNTS.get(res.name)
            if n_expected is None:
                continue
            n_heavy = sum(1 for atom in res.atoms if atom.element.symbol != 'H')
            if n_heavy < n_expected:
                incomplete.append(pos_to_resnum[i])

        return incomplete



    def _build_homology_model(self, nstd_resids):
        """
        Build a homology model with Modeller.AutoModel
        """

        if hasattr(self, 'secondary_template_pdb'):
            knowns = (self.name, self.secondary_name)
        else:
            knowns = (self.name)

        # Only optimize the residues that have to be built. Anything else keeps the coordinates it inherited
        # from the input .pdb. Falls back to optimizing everything if the gaps could not be determined.
        gaps = self.model_gaps if self.preserve_resolved else None
        if gaps is not None and len(gaps) == 0:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//No residues to build, optimizing all atoms instead', flush=True)
            gaps = None

        cyclic = self.cyclic
        if cyclic:
            self.env.patch_default = False

        class RepairModel(AutoModel):
            def select_atoms(self):
                if gaps is None:
                    return Selection(self)
                return Selection(*[self.residue_range(f'{gap[0]}:A', f'{gap[1]}:A') for gap in gaps])

            def special_patches(self, aln):
                if cyclic:
                    # Link between last residue (-1) and first (0) to make chain cyclic:
                    self.patch(residue_type='LINK', residues=(self.residues[-1], self.residues[0]))

        self.model = RepairModel(self.env,
                                 sequence=self.fasta_name,
                                 knowns=knowns,
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




