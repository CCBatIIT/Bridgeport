import os, sys, json, shutil
from datetime import datetime
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
import pathlib
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import numpy as np
from utils.utils import *
from utils.ProteinPreparer import ProteinPreparer
from RepairProtein.RepairProtein import RepairProtein
from ForceFields.ForceFieldHandler import ForceFieldHandler
from ForceFields.OpenMMJoiner import Joiner
from Ligand.Ligand import Ligand
from Ligand.Analogue import Analogue
from Ligand.MutatedPeptide import MutatedPeptide
from Minimizer.Minimizer import Minimizer
from openmm.app import *
from openmm import *
from openmm.unit import *
from openbabel import openbabel
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy

class Bridgeport():
    """
    Master class to prepare crystal structures for OpenMM simulation. 


    Methods:
    --------
        run():
            Run all methods to prepare an OpenMM system. Steps:
                1. Align input structure to reference structure.
                2. Separate ligand and protein for separate preparation steps.
                3. Add missing residues and atoms using Modeller based on provided sequence.
                4. Add hydrogens, solvent, and membrane (optional).
                5. Prepare the ligand.
                6. Generate an OpenMM system. 

            Output system .pdb and .xml file will be found in self.working_dir/systems

        get_analogue_MCS():
            Get the maximum common substructure of an analogue and an experimental ligand. 
        
        build_analogue_complex():
            Build a new input complex by replacing a ligand with an analogue.

        align():
            Align initial structure to a reference structure. The reference structure can include a structure from the OPM database for transmembrane proteins.
        
        separate_lig_prot():
            Separate ligand and protein based on chain and resname specified in input file.
        
        repair_crystal():
            Uses the RepairProtein class to replace missing residues with UCSF Modeller. 

        add_environment():
             Add water, lipids, and hydrogens to protein with the ProteinPreparer class.

        ligand_prep():
            Prepare ligand for OpenFF parameterization.

        generate_systems():
            Generate forcefields with OpenFF using the ForcefieldHandler and OpenMMJoiner classes.    
    """

    def __init__(self, input_json: str, verbose: bool=False):
        """
        Initialize Bridgeport objects.

        Parameters:
        -----------
            input_json (str):
                String path to .json file that contains inputs

            verbose (bool):
                If true, show missing and mutated residues after each iteration of sequence alignment. Default is False. 

        Returns:
        --------
            Bridgeport object.
        """
        # Make assertions
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Welcome to Bridgeport.', flush=True)
        assert os.path.exists(input_json), "Cannot find input_json."
        assert input_json.split('.')[1] == 'json', f"input_json: {input_json} is not a type .json."

        # Load input from json
        self.verbose = verbose
        self.input_params = json.load(open(input_json, 'r'))
        self.working_dir = self.input_params['working_dir']
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input parameters.', flush=True)
        for key, item in self.input_params.items():
            try: 
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + key + ':', flush=True)
                for key2, item2 in self.input_params[key].items():
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//\t' + key2 + ':', item2, flush=True)       
            except:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + key + ':', item, flush=True)   


        # Set other attributes
        if 'Ligand' in list(self.input_params.keys()):
            if 'resname' in self.input_params['Ligand'] and self.input_params['Ligand']['resname'] is not False:
                
                if 'Analogue' in self.input_params['Ligand']:
                    self.type = 'small_molecule'
                    self.name = self.input_params['Ligand']['Analogue']['name']
                    self.resname = self.input_params['Ligand']['resname']
                    self.lig_chainid = False
                        
                else:
                    self.type = 'small_molecule'
                    self.name = self.input_params['Ligand']['name']
                    self.resname = self.input_params['Ligand']['resname']
                    self.lig_chainid = False
                    
            elif 'chainid' in self.input_params['Ligand'] and self.input_params['Ligand']['chainid'] is not False:
               
                if 'Analogue' in self.input_params['Ligand']:
                    
                    if 'small_molecule_params' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['small_molecule_params'] is True:
                        self.type = 'small_molecule'
                        self.name = self.input_params['Ligand']['Analogue']['name']
                        self.resname = False
                        self.lig_chainid = self.input_params['Ligand']['chainid']
                        
                    else:
                        self.type = 'peptide'
                        self.name = self.input_params['Ligand']['Analogue']['name']
                        self.resname = False
                        self.lig_chainid = self.input_params['Ligand']['chainid']
    
                else:
                    self.type = 'peptide'
                    self.name = self.input_params['Ligand']['name']
                    self.resname = False
                    self.lig_chainid = self.input_params['Ligand']['chainid']
        else:
            self.type = 'apo'
            self.name = 'apo'
            

        # Get initial structure
        self.input_pdb_dir = self.input_params['Protein']['input_pdb_dir']
        pdb_fn = self.input_params['Protein']['input_pdb']
        self.input_pdb = os.path.join(self.input_pdb_dir, pdb_fn)
        self.chain = self.input_params["Protein"]["chain"]
    
        # Assign other dir locations
        self.lig_only_dir = os.path.join(self.working_dir, 'ligands')
        self.aligned_input_dir = os.path.join(self.working_dir, 'aligned_input_pdb')
        self.prot_only_dir = os.path.join(self.working_dir, 'proteins')


    
    def run(self):
        """
        Run all methods to prepare an OpenMM system. Steps:
            1. Align input structure to reference structure.
            2. Separate ligand and protein for separate preparation steps.
            3. Add missing residues and atoms using Modeller based on provided sequence.
            4. Add hydrogens, solvent, and membrane (optional).
            5. Prepare the ligand.
            6. Generate an OpenMM system. 

        Output system .pdb and .xml file will be found in self.working_dir/systems
        """

        
        # Align first
        self.align_to_reference()
        
        # Build analogue complex
        if self.type != 'apo' and 'Analogue' in self.input_params['Ligand']:
                self.get_analogue_MCS()
                self.build_analogue_complex()

        # Ligand and Protein Seperate
        self.separate_lig_prot()     
        
        # Make Repair Protein Optional be setting working dir to false if it shouldn't be done
        if self.input_params['RepairProtein']['working_dir'] != False:
            self.repair_protein()
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Repair Working Dir set to False - Skipping Repairs with Modeller', flush=True)
            
        #Add Water (and possibly membrane)
        self.add_environment()
        
        #Prepare Ligand
        if self.type != 'apo':
            self.ligand_prep()
        
        #Make OpenMM Systems
        self.generate_systems()

        # Choose analogue complex, if applicable
        if hasattr(self, "analogue_pdbs"):
            if "minimize" in self.input_params["Ligand"] and self.input_params["Ligand"]["minimize"] == False:
                pass
            else:
                self.choose_analogue_conformer()

        # Repositions
        self.reposition_at_origin()


    def get_analogue_MCS(self, 
                         add_atoms: List[List[int]]=None, 
                         remove_atoms: List[int]=None,
                         strict: bool=True):
        """
        Get the maximum common substructure of an analogue (from smiles) and a template ligand with an experimental structure. 

        Parameters:
        -----------
            add_atoms (List[List[int]]):
                List of atom inds as depicted to be added to common substructure. Ex: [[0, 1], [2, 4]] where atoms 0 and 2 in the analogue match atoms 1 and 4 in the template, respectively. Default is False, which will use automatically determined maximum common substructure.
                
            remove_atoms (List[int]):
                List of atoms inds to remove from the analogue. The corresponding atoms from the template structure will be removed automatically. Default is False, which will use automatically determined maximum common substructure.
        """
        # Build necessary directories
        if not os.path.exists(self.lig_only_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Making directory for ligand structures:', self.lig_only_dir, flush=True)  
            os.mkdir(self.lig_only_dir)        
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for ligand structures:', self.lig_only_dir, flush=True)  


        # Build template ligand
        if self.type == 'small_molecule':
            
            # Get template ligand from input structure
            lig_sele = mda.Universe(self.aligned_pdb).select_atoms(f'chainid {self.chain} and resname {self.resname}')
            try:
                lig_sele.write(os.path.join(self.lig_only_dir, self.input_params['Ligand']['name'] + '.pdb'))
            except:
                raise Exception(f'Could not select ligand from {self.aligned_pdb} with chainid {self.chain} and resname {self.resname}')

            # Build Ligand
            self.template = Ligand(working_dir=self.lig_only_dir,
                              name=self.input_params['Ligand']['name'],
                              resname=self.resname,
                              smiles=self.input_params['Ligand']['smiles'])
            
        elif self.type == 'peptide':

            # Get template ligand from input structure
            lig_sele = mda.Universe(self.aligned_pdb).select_atoms(f'chainid {self.lig_chainid}')
            lig_sele.write(os.path.join(self.lig_only_dir, self.input_params['Ligand']['name'] + '.pdb'))

            # Build Ligand
            self.template = Ligand(working_dir=self.lig_only_dir,
                              name=self.input_params['Ligand']['name'],
                              chainid=self.lig_chainid,
                              smiles=self.input_params['Ligand']['smiles'])

        # Prepare ligand
        try:
            self.template.prepare_ligand(small_molecule_params=True, proximityBonding=True, visualize=False) 
        except:
            self.template.prepare_ligand(small_molecule_params=True, proximityBonding=True, visualize=True)
        
        # Build analogue 
        self.analogue = Analogue(template=self.template,
                            working_dir=self.lig_only_dir,
                            name=self.input_params['Ligand']['Analogue']['name'],
                            smiles=self.input_params['Ligand']['Analogue']['smiles'],
                            verbose=self.verbose)

        # Get MCS
        if 'add_atoms' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['add_atoms'] is not False:
            add_atoms = self.input_params['Ligand']['Analogue']['add_atoms']
        if 'remove_atoms' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['remove_atoms'] is not False:
            remove_atoms = self.input_params['Ligand']['Analogue']['remove_atoms']

        self.analogue.get_MCS(add_atoms=add_atoms, remove_atoms=remove_atoms, strict=strict)

                         
    
    
    def build_analogue_complex(self,
                               n_conformers: int=1,
                               align_all: bool=False,
                               rmsd_thresh: float=3.0):
        """
        Build a new input complex by replacing a ligand with an analogue.

        Parameters:
        -----------
            align_all (bool): 
                If True, will use atoms in *add_atoms* for alignment. Default is False which will only use the automatically detected maximum common substructure.
            
            rmsd_thresh (float):
                RMSD threshold that analogue conformation must reach during alignment to be accepted as a permittable structure. Default is 3.0 Angstrom.
        """
        

        # Generate conformers
        if 'n_conformers' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['n_conformers'] is not False:
            n_conformers = self.input_params['Ligand']['Analogue']['n_conformers']
        if 'align_all' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['align_all'] is not False:
            align_all = self.input_params['Ligand']['Analogue']['align_all']
        if 'rmsd_thresh' in self.input_params['Ligand']['Analogue'] and self.input_params['Ligand']['Analogue']['rmsd_thresh'] is not False:
            rmsd_thresh = self.input_params['Ligand']['Analogue']['rmsd_thresh']

        self.analogue.generate_conformers(n_conformers=n_conformers, align_all=align_all, rmsd_thresh=rmsd_thresh)
        self.analogue_pdbs = [pdb for pdb in os.listdir(self.analogue.conformer_dir) if pdb.endswith('pdb')]

        # Get protein 
        prot_sele = mda.Universe(self.aligned_pdb).select_atoms(f'protein and chainid {self.chain}')

        
        # Combine to create new initial complex
        self.name = self.analogue.name
        self.aligned_pdb = os.path.join(self.aligned_input_dir, self.name+'.pdb')
        u = mda.core.universe.Merge(prot_sele, self.analogue.sele)
        u.select_atoms('all').write(self.aligned_pdb)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Built new inital complex.', flush=True)

        # Change input parameters
        self.resname = 'UNL'
        self.type = 'small_molecule'
        self.input_params['Ligand']['smiles'] = self.input_params['Ligand']['Analogue']['smiles']
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Changing ligand resname to:', self.resname, flush=True)


    
    def align_to_reference(self, ref_chains: List[str]=['A']):
        """
        Align initial structure to a reference structure. 
        The reference structure can include a structure from the OPM database for transmembrane proteins.

        Parameters:
        -----------
            ref_chains (List[str]):
                Chain ID of the chain to use in reference for alignment. 
        """        
        
        # Create directory for aligned proteins 
        if not os.path.exists(self.aligned_input_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Creating directory for aligned input structures:', self.aligned_input_dir, flush=True)    
            os.mkdir(self.aligned_input_dir)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for aligned input structures:', self.aligned_input_dir, flush=True)    


        # Load reference structure
        ref_pdb_path = self.input_params['Environment']['alignment_ref']
        if 'reference_chain' in self.input_params['Environment']:
            ref_chains = self.input_params['Environment']['reference_chain']
            
        if os.path.exists(ref_pdb_path):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found references structure', ref_pdb_path, 'and will align to chains', ref_chains, flush=True)  
        else:
            raise FileNotFoundError("Cannot find reference structure:", ref_pdb_path)
            
        self.ref = mda.Universe(ref_pdb_path)
        self.ref_resids = self.ref.select_atoms('chainid ' + ' or '.join(chain for chain in ref_chains)).residues.resids
        
        # Load structure to align
        if os.path.exists(self.input_pdb):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found input structure:', self.input_pdb, flush=True)

            # Slim to correct chain
            u = mda.Universe(self.input_pdb)
            
            #Make the selection
            if len(self.chain) > 1:
                chain_sele_str = f'chainid {self.chain.split()[0]}'
            else:
                chain_sele_str = f'chainid {self.chain}'
            chain_sele = u.select_atoms(chain_sele_str)

            # Get resids
            resids = chain_sele.residues.resids

            # Find matching resids
            matching_resids, matching_res_inds, matching_ref_res_inds = np.intersect1d(resids, self.ref_resids, return_indices=True)
                        
            sele_str = chain_sele_str +\
                       ' and resid ' + ' '.join(str(resids[res_ind]) for res_ind in matching_res_inds) +\
                       ' and name CA'
            ref_sele_str = 'chainid ' + ' or '.join(chain for chain in ref_chains) +\
                           ' and resid ' + ' '.join(str(resids[res_ind]) for res_ind in matching_res_inds) +\
                           ' and name CA'
            
            # Align
            try:
                _, _ = alignto(mobile=u, 
                        reference=self.ref,
                        select={'mobile': sele_str,
                              'reference': ref_sele_str})
            except:
                print('Mobile sele str:', sele_str)
                print('Mobile sele:', u.select_atoms(sele_str).residues.resids)
                print('Ref sele str:', ref_sele_str)
                print('Ref sele:', self.ref.select_atoms(ref_sele_str).residues.resids)
                raise Exception('Could not find match')
                
            # Save 
            self.aligned_pdb = os.path.join(self.aligned_input_dir, self.name+'.pdb')
            u.select_atoms('all').write(self.aligned_pdb)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved aligned structure to:', self.aligned_pdb, flush=True)

        else:
            raise FileNotFoundError(f'Could not locate {self.input_pdb}')

    
    
    def separate_lig_prot(self):
        """
        Separate ligand and protein based on chain and resname specified in input file.
        """
        # Create directories for separated .pdb files
        if not os.path.exists(self.prot_only_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Making directory for protein structures:', self.prot_only_dir, flush=True) 
            os.mkdir(self.prot_only_dir)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for protein structures:', self.prot_only_dir, flush=True)  

        self.lig_only_dir = os.path.join(self.working_dir, 'ligands')
        if not os.path.exists(self.lig_only_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Making directory for ligand structures:', self.lig_only_dir, flush=True)  
            os.mkdir(self.lig_only_dir)        
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for ligand structures:', self.lig_only_dir, flush=True)  

        # Iterate through input files
        u = mda.Universe(self.aligned_pdb)

        # Select protein by chain ID 
        prot_sele = u.select_atoms(f'protein and chainid {self.chain}')
        self.prot_pdb = os.path.join(self.prot_only_dir, self.name+'.pdb')
        prot_sele.write(self.prot_pdb)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated chain(s)', self.chain, 'from input structure', flush=True)  

        # Select ligand by resname or peptide_chain    
        if self.type != 'apo':
            if self.type == 'small_molecule':
                if self.resname is not False:
                    lig_sele = u.select_atoms(f'resname {self.resname}')
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated ligand', self.resname, 'from input structure with', lig_sele.n_atoms, 'atoms', flush=True)
                
            elif self.type == 'peptide':
                lig_sele = u.select_atoms(f'chainid {self.lig_chainid}')
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated ligand', self.lig_chainid, 'from input structure', flush=True)
    
            self.lig_pdb = os.path.join(self.lig_only_dir, self.name+'.pdb')
            lig_sele.write(self.lig_pdb)


    
    def repair_protein(self, 
                       engineered_resids: List[int]=None, 
                       secondary_template: str=None, 
                       tails: List[int]=False, 
                       loops: List[List[int]]=False):
        """
        Uses the RepairProtein class to replace missing residues with UCSF Modeller. 
        """
        # Load parameters
        params = self.input_params['RepairProtein']
        if 'engineered_resids' in params and params['engineered_resids'] is not False:
            engineered_resids = params['engineered_resids']
        if 'secondary_template' in params and params['secondary_template'] is not False:
            secondary_template = params['secondary_template']
        if 'tails' in params and params['tails'] is not False:
            self.tails = params['tails']
        

        # Run with secondary template if specified
        protein_reparer = RepairProtein(pdb_fn=self.prot_pdb,
                                        fasta_fn=params['fasta_path'], 
                                        mutated_resids=engineered_resids,
                                        working_dir=params['working_dir'])
        
        protein_reparer.run(pdb_out_fn=self.prot_pdb,
                            secondary_template_pdb=secondary_template,
                            tails=self.tails,
                            loops=False,
                            verbose=self.verbose)

        
    
    def add_environment(self, pH: float=7.0, membrane: bool=False, ion_strength: float=0.15):
        """
        Add water, lipids, and hydrogens to protein with the ProteinPreparer class.
        """        
        # See if environment parameters are present
        if "Environment" in self.input_params.keys():
            if "pH" in self.input_params["Environment"].keys():
                pH = self.input_params["Environment"]["pH"]
            if "membrane" in self.input_params["Environment"].keys():
                membrane = self.input_params['Environment']['membrane']
            if "ion_strength" in self.input_params["Environment"].keys():
                ion_strength = self.input_params["Environment"]["ion_strength"]
        
        # Add the environment with ProteinPreparer
        if membrane:
            pp = ProteinPreparer(pdb_path=self.prot_pdb,
                                 working_dir=self.prot_only_dir,
                                 pH=pH,
                                 env='MEM',
                                 ion_strength=ion_strength,
                                 verbose=self.verbose)
        else:
            pp = ProteinPreparer(pdb_path=self.prot_pdb,
                                 working_dir=self.prot_only_dir,
                                 pH=pH,
                                 env='SOL',
                                 ion_strength=ion_strength,
                                 verbose=self.verbose)            
        pp.main()

        # Remove unecessary .pqr and .log files
        for fn in os.listdir(self.prot_only_dir):
            if fn.endswith('.log') or fn.endswith('.pqr'):
                os.remove(os.path.join(self.prot_only_dir, fn))


        # Set attr
        self.env_pdb = os.path.join(self.prot_only_dir, self.name+'_env.pdb')

        # Correct resids
        pdb = md.load_pdb(self.env_pdb)
        top = pdb.topology
        if top.residue(0).resSeq == 1:
            for i in range(top.n_residues):
                top.residue(i).resSeq += self.tails[0]
            pdb[0].save_pdb(self.env_pdb)


    def ligand_prep(self,
                    small_molecule_params: bool=False,
                    sanitize: bool=True,
                    removeHs: bool=True,
                    proximityBonding: bool=False,
                    sequence: str=None,
                    nstd_resids: List[int]=None,
                    pH: float=7.0,
                    neutral_Cterm: bool=False,
                    loops=False,
                    chain=False,
                    smiles=False):
        """ 
        Prepare ligand for OpenFF parameterization.

        Parameters:
        -----------
            out_fm (Str):
                String of extension of intended format to write out. Default is .sdf. 
        """
        # Load parameters
        params = self.input_params['Ligand']
        if 'sequence' in params.keys():
            sequence = params['sequence']
        if 'small_molecule_params' in params.keys():
            small_molecule_params = params['small_molecule_params']
        if 'sanitize' in params.keys():
            sanitize = params['sanitize']
        if 'removeHs' in params.keys():
            removeHs = params['removeHs']
        if 'proximityBonding' in params.keys():
            proximityBonding = params['proximityBonding']
        if 'pH' in params.keys():
            pH = params['pH']
        if 'nstd_resids' in params.keys():
            nstd_resids = params['nstd_resids']
        if 'neutral_Cterm' in params.keys():
            neutral_Cterm = params['neutral_Cterm']
        if 'smiles' in params.keys():
            smiles = params['smiles']
        if 'loops' in params.keys():
            loops = params['loops']
        if 'chain' in params.keys():
            chain = params['chain']
        

        # Prepare based on type of ligand
        if self.type == 'small_molecule':
            ligand = Ligand(working_dir=self.lig_only_dir, 
                            name=self.name,
                            resname=self.resname,
                            smiles=smiles,
                            verbose=self.verbose)

            ligand.prepare_ligand(small_molecule_params=small_molecule_params,
                                  sanitize=sanitize,
                                  removeHs=removeHs,
                                  proximityBonding=proximityBonding)
                                  
            self.lig_sdf = ligand.sdf

        elif self.type == 'peptide':

            if 'MutatedPeptide' in self.input_params['Ligand'].keys():
                self.ligand_xmls = []
                for i in range(len(self.input_params['Ligand']['MutatedPeptide']['Mutations'])):
                    mp_params = self.input_params['Ligand']['MutatedPeptide']['Mutations'][i]
    
                    # Prepare reference peptide for mutation
                    ref_name = mp_params['mutation_resname'] + str(mp_params['mutation_resid']) + '_reference'
                    shutil.move(self.lig_pdb, os.path.join(self.lig_only_dir, ref_name + '.pdb'))
                    reference = Ligand(working_dir=self.lig_only_dir, name=ref_name, chainid=params['chain'], sequence=sequence)
                    if i == 0:
                        reference.prepare_ligand(small_molecule_params=False, removeHs=False)
    
                    # Mutate
                    ligand = MutatedPeptide(template=reference, replace_resid=mp_params['mutation_resid'], replace_resname=mp_params['mutation_resname'], replace_smiles=mp_params['mutation_smiles'], working_dir=self.lig_only_dir, name=self.name, chainid=params['chain'])
                    if 'remove_atoms' in mp_params.keys():
                        remove_atoms = mp_params['remove_atoms']
                    else:
                        remove_atoms = []
                    if 'change_atoms' in mp_params.keys():
                        change_atoms = mp_params['change_atoms']
                    else:
                        change_atoms = {}
                    if 'bonds_to_add' in mp_params.keys():
                        bonds_to_add = mp_params['bonds_to_add']
                    else:
                        bonds_to_add = None
                    if 'external_bonds' in mp_params.keys():
                        external_bonds = mp_params['external_bonds']
                    else:
                        external_bonds = None
                    ligand.run(remove_atoms=remove_atoms, change_atoms=change_atoms, bonds_to_add=bonds_to_add, external_bonds=external_bonds)
                    self.ligand_xmls.append(ligand.analogue.xml)
                    reference.pdb = ligand.pdb

                
            else:
                ligand = Ligand(working_dir=self.lig_only_dir,
                                name=self.name,
                                chainid=self.lig_chainid,
                                sequence=sequence,
                                smiles=smiles,
                                verbose=self.verbose)
    
                ligand.prepare_ligand(small_molecule_params=small_molecule_params,
                                      pH=pH,
                                      nstd_resids=nstd_resids,
                                      neutral_Cterm=neutral_Cterm,
                                      loops=loops,
                                      chain=chain)


        # If analogues were generated, prepare those too
        if hasattr(self, 'analogue_pdbs'):
            for conf_pdb in self.analogue_pdbs:
                
                # Prepare based on type of ligand
                if self.type == 'small_molecule':
                    ligand = Ligand(working_dir=self.analogue.conformer_dir, 
                                    name=conf_pdb.split('.')[0],
                                    resname=self.resname,
                                    smiles=params['smiles'],
                                    verbose=self.verbose)
        
                    ligand.prepare_ligand(small_molecule_params=small_molecule_params,
                                          sanitize=sanitize,
                                          removeHs=removeHs,
                                          proximityBonding=proximityBonding,
                                          visualize=False)
                    
                elif self.type == 'peptide':
                    ligand = Ligand(working_dir=self.analogue.conformer_dir, 
                                    name=conf_pdb.split('.')[0],
                                    chainid=self.lig_chainid,
                                    sequence=sequence,
                                    smiles=params['smiles'],
                                    verbose=self.verbose)
        
                    ligand.prepare_ligand(pH=pH,
                                          nstd_resids=nstd_resids,
                                          neutral_Cterm=neutral_Cterm,
                                          visualize=False)
        
        

    def generate_systems(self):
        """
        Generate forcefields with OpenFF using the ForcefieldHandler and OpenMMJoiner classes.
        """
        # Create systems dir
        self.sys_dir = os.path.join(self.working_dir, 'systems')
        if not os.path.exists(self.sys_dir):
            os.mkdir(self.sys_dir)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Created systems directory:', self.sys_dir, flush=True)
        
        # Iterate through files
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Building parameters for', self.name, flush=True)
        

        # Get ligand path
        if self.type != 'apo':
            if self.type == 'small_molecule':
                lig_path = self.lig_sdf
            elif self.type == 'peptide':
                lig_path = self.lig_pdb
            assert os.path.exists(lig_path), f"Cannot find path to ligand file: {lig_path}"

            # Generate protein system
            assert os.path.exists(self.env_pdb), f"Cannot find path to protein file in environment: {self.env_pdb}"
            prot_sys, prot_top, prot_pos = ForceFieldHandler(self.env_pdb).main()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Protein parameters built.', flush=True)
            
            # Generate ligand system
            if 'MutatedPeptide' in self.input_params['Ligand'].keys():
                lig_sys, lig_top, lig_pos = ForceFieldHandler(lig_path, force_field_files=self.ligand_xmls).main(use_nonbonded=False)
            else:
                lig_sys, lig_top, lig_pos = ForceFieldHandler(lig_path).main(use_nonbonded=False)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Ligand parameters built.', flush=True)
            
            # Combine systems 
            self.sys, self.top, self.pos = Joiner((lig_sys, lig_top, lig_pos),  (prot_sys, prot_top, prot_pos)).main()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'System parameters built.', flush=True)

        else:
            # Generate protein system
            assert os.path.exists(self.env_pdb), f"Cannot find path to protein file in environment: {self.env_pdb}"
            self.sys, self.top, self.pos = ForceFieldHandler(self.env_pdb).main()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Protein parameters built.', flush=True)
            

        # Get energy
        int = LangevinIntegrator(300 * kelvin, 1/picosecond, 0.001 * picosecond)
        self.sim = Simulation(self.top, self.sys, int)
        self.sim.context.setPositions(self.pos)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Initial structure potential energy:', np.round(self.sim.context.getState(getEnergy=True).getPotentialEnergy()._value, 2), flush=True)
        
        # Save combined systems
        self.final_pdb = os.path.join(self.sys_dir, self.name+'.pdb')
        self.final_xml = os.path.join(self.sys_dir, self.name+'.xml')        
        with open(self.final_pdb, 'w') as f:
            PDBFile.writeFile(self.sim.topology, self.sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        with open(self.final_xml, 'w') as f:
            f.write(XmlSerializer.serialize(self.sim.system))


        
    def reposition_at_origin(self):
        """
        OpenMM does not like having the origin in the middle :)
        """
        # Reposition at origin
        box_vectors = self.sys.getDefaultPeriodicBoxVectors()
        translate = Quantity(np.array((box_vectors[0].x,
                                       box_vectors[1].y,
                                       box_vectors[2].z))/2,
                             unit=nanometer)

        # Get positions in case they changed
        self.pos = PDBFile(self.final_pdb).positions
        
        # Get energy
        int = LangevinIntegrator(300 * kelvin, 1/picosecond, 0.001 * picosecond)
        self.sim = Simulation(self.top, self.sys, int)
        self.sim.context.setPositions(self.pos + translate)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Initial structure potential energy:', np.round(self.sim.context.getState(getEnergy=True).getPotentialEnergy()._value, 2), flush=True)
        
        # Write out
        with open(self.final_pdb, 'w') as f:
            PDBFile.writeFile(self.sim.topology, self.sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        with open(self.final_xml, 'w') as f:
            f.write(XmlSerializer.serialize(self.sim.system))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Final system coordinates saved to', self.final_pdb, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Final system parameters saved to', self.final_xml, flush=True)


        
    def choose_analogue_conformer(self):
        """
        """
        
        def _minimize_new_lig_coords(ref_traj, lig_sele, conf_path, lig_resname='UNL', min_out_pdb=None):
            temp_conf_pdb = self.name + '_temp_complex.pdb'
            
            # Get ligand positions
            lig_pos = md.load_pdb(conf_path).xyz[0]

            # Adjust in reference complex
            traj.xyz[0, lig_sele, :] = lig_pos
            traj.save_pdb(temp_conf_pdb)

            # Minimize
            row = Minimizer(self.final_xml, temp_conf_pdb, 'NA')
            if min_out_pdb != None:
                row._minimize(temp_conf_pdb, pdb_out=min_out_pdb, lig_resname='UNK', mcs=self.analogue.matching_atoms)
            else:
                row._minimize(temp_conf_pdb, lig_resname='UNK', mcs=self.analogue_mcs)

            return temp_conf_pdb, row.PE

        
        # Make directory to store minimized files
        min_sys_dir = os.path.join(self.sys_dir, self.name + '_minimized_conformers')
        if not os.path.exists(min_sys_dir):
            os.mkdir(min_sys_dir)
        
        # Remove CONECT records from self.final_pdb
        lines = [l for l in open(self.final_pdb, 'r').readlines() if not l.startswith('CONECT')]
        with open(self.final_pdb, 'w') as f:
            f.writelines(lines)
            
        # Load initial structure
        # u = mda.Universe(self.final_pdb)
        # u.select_atoms('all').write(self.final_pdb)
        top = md.Topology.from_openmm(self.sim.topology)
        traj = md.Trajectory(self.sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value, topology=top)
        lig_sele = traj.topology.select(f'resname UNK')
        assert len(lig_sele) > 0

        # Iterate through analogue conformers
        potential_energies = np.zeros(len(self.analogue_pdbs))
        for i, conf_pdb in enumerate(self.analogue_pdbs):
            conf_path = os.path.join(self.analogue.conformer_dir, conf_pdb)
            min_out_path = os.path.join(min_sys_dir, conf_pdb)
            temp_conf_pdb, potential_energies[i] = _minimize_new_lig_coords(traj, lig_sele, conf_path, min_out_pdb=min_out_path)

        # Choose minimum PE
        conf_pdb = self.analogue_pdbs[list(potential_energies).index(potential_energies.min())]
        conf_path = os.path.join(self.analogue.conformer_dir, conf_pdb)
        shutil.copy(os.path.join(min_sys_dir, conf_pdb), self.final_pdb)

        # Clean 
        if os.path.exists(temp_conf_pdb):
            os.remove(temp_conf_pdb)
        
        # Write out PE
        np.save(os.path.join(min_sys_dir, 'PEs.npy'), potential_energies)


