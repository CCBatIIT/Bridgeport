import os, sys, json
from datetime import datetime
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
import pathlib
bp_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(bp_dir, 'RepairProtein'))
sys.path.append(os.path.join(bp_dir, 'ForceFields'))
sys.path.append(os.path.join(bp_dir, 'utils'))
import numpy as np
from bp_utils import analogue_alignment
from ProteinPreparer import ProteinPreparer
from RepairProtein import RepairProtein
from ForceFieldHandler import ForceFieldHandler
from OpenMMJoiner import Joiner
from openmm.app import *
from openmm import *
from openmm.unit import *
from openbabel import openbabel
from pdbfixer import PDBFixer

class Bridgeport():
    """
    Master class to prepare crystal structures for OpenMM simulation. 

    Attributes:
    -----------
    input_json (str):
        String path to input .json file.

    input_params (dict):
        Dictionary of input options read from self.input_json. For information on how to write the input file, reference the Bridgeport README

    working_dir (str):
        String path to working directory where all subdirectories will be created and output files will be stored. 

    prot_only_dir (str):
        String path to directory where protein .pdb files will be stored.

    lig_only_dir (str):
        String path to directory where lig .pdb files will be stored. 

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
    
        _align():
            Align initial structure to a reference structure. The reference structure can include a structure from the OPM database for transmembrane proteins.
        
        _separate_lig_prot():
            Separate ligand and protein based on chain and resname specified in input file.
        
        _repair_crystal():
            Uses the RepairProtein class to replace missing residues with UCSF Modeller. 

        _add_environment():
             Add water, lipids, and hydrogens to protein with the ProteinPreparer class.

        _ligand_prep():
            Prepare ligand for OpenFF parameterization.

        _generate_systems():
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
        self.input_json = input_json
        self.input_params = json.load(open(self.input_json))
        self.working_dir = self.input_params['working_dir']
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input parameters.', flush=True)
        for key, item in self.input_params.items():
            try: 
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + key + ':', flush=True)
                for key2, item2 in self.input_params[key].items():
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//\t' + key2 + ':', item2, flush=True)       
            except:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + key + ':', item, flush=True)   

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
        # If analogue, create new intial structure
        if self.input_params['ligand']['lig_resname'] == False:
            if self.input_params['ligand']['peptide_chain'] == False:
                if 'analogue_smiles' in self.input_params['ligand'] and self.input_params['ligand']['analogue_smiles'] != False:
                    self.analogue_smiles=self.input_params['ligand']['analogue_smiles']
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found analogue with smiles:', self.analogue_smiles, flush=True)
                    if 'analogue_name' in self.input_params['ligand'] and self.input_params['ligand']['analogue_name'] != False:
                        self.analogue_name=self.input_params['ligand']['analogue_name']
                        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found analogue with name:', self.analogue_name, flush=True)
                        if 'known_structure' in self.input_params['ligand'] and self.input_params['ligand']['known_structure'] != False:
                            self.analogue_pdb = self.input_params['ligand']['known_structure']
                            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found reference ligand file:', self.analogue_pdb, flush=True)
                            if 'known_resname' in self.input_params['ligand'] and self.input_params['ligand']['known_resname'] != False:
                                self.analogue_resname = self.input_params['ligand']['known_resname']
                                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found reference ligand with resname:', self.analogue_resname, flush=True)
    
                                # Build analogue complex
                                self._build_analogue_complex()
                            
        # Run 
        self._align()
        self._separate_lig_prot()
        self._repair_crystal()
        self._add_environment()
        self._ligand_prep()
        self._generate_systems()

    def _build_analogue_complex(self):
        """
        Build a new input complex by replacing a ligand with an analogue.
        """
        # Build necessary directories
        if not os.path.exists(self.lig_only_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Making directory for ligand structures:', self.lig_only_dir, flush=True)  
            os.mkdir(self.lig_only_dir)        
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for ligand structures:', self.lig_only_dir, flush=True)  
        
        # Get known ligand
        name = self.input_params['protein']['input_pdb']
        ref_u = mda.Universe(self.analogue_pdb)
        ref_sele = ref_u.select_atoms('resname '+self.analogue_resname)
        assert ref_sele.n_atoms > 0, f"Could not find any atoms with resname {self.analogue_resname}"
        prot_sele = ref_u.select_atoms('protein')
        assert prot_sele.n_atoms > 0, f"Could not find any protein atoms in {self.analogue_pdb}"

        # Write to ligand folder
        ref_lig_pdb = os.path.join(self.working_dir, 'ligands', name)
        ref_sele.write(ref_lig_pdb)

        # General aligned analogue
        lig_path = os.path.join(self.lig_only_dir, self.analogue_name+'.pdb')
        analogue_alignment(smiles=self.analogue_smiles,
                           known_pdb=self.analogue_pdb,
                           known_resname=self.analogue_resname,
                           analogue_out_path=lig_path)
        assert os.path.exists(lig_path), f"No output file exists at {lig_path}"

        # Combine to create new initial complex
        new_input_path = os.path.join(self.working_dir, self.input_params['protein']['input_pdb_dir'], self.analogue_name+'.pdb')
        lig_sele = mda.Universe(lig_path).select_atoms('all')
        u = mda.core.universe.Merge(prot_sele, lig_sele)
        assert u.select_atoms('all').n_atoms == lig_sele.n_atoms + prot_sele.n_atoms, "Did not correctly merge ligand and protein AtomGroups."
        u.select_atoms('all').write(new_input_path)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Build new inital complex.', flush=True)
        self.input_params['protein']['input_pdb'] = self.analogue_name+'.pdb'
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Changing input_pdb to:', self.input_params['protein']['input_pdb'], flush=True)
        self.input_params['ligand']['lig_resname'] = 'UNL'
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Changing ligand resname to:', self.input_params['ligand']['lig_resname'], flush=True)
    
    def _align(self):
        """
        Align initial structure to a reference structure. 
        The reference structure can include a structure from the OPM database for transmembrane proteins.
        """        
        # Create directory for aligned proteins 
        if not os.path.exists(self.aligned_input_dir):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Creating directory for aligned input structures:', self.aligned_input_dir, flush=True)    
            os.mkdir(self.aligned_input_dir)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found directory for aligned input structures:', self.aligned_input_dir, flush=True)    


        # Load reference structure
        ref_pdb_path = self.input_params['environment']['alignment_ref']
        if os.path.exists(ref_pdb_path):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found references structure', ref_pdb_path, flush=True)  
        else:
            raise FileNotFoundError("Cannot find reference structure:", ref_pdb_path)
        ref = mda.Universe(ref_pdb_path)
        ref_resids = ref.residues.resids
        
        # Load structure to align
        input_pdb_dir = self.input_params['protein']['input_pdb_dir']
        pdb = self.input_params['protein']['input_pdb']
        input_pdb_path = os.path.join(input_pdb_dir, pdb)
        if os.path.exists(input_pdb_path):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found input structure:', input_pdb_path, flush=True)

            # Slim to correct chain
            u = mda.Universe(input_pdb_path)
            chain_sele = u.select_atoms('chainid ' + self.input_params['protein']['chains'])

            # Get resids
            resids = chain_sele.residues.resids

            # Find matching resids
            matching_resids, matching_res_inds, matching_ref_res_inds = np.intersect1d(resids, ref_resids, return_indices=True)
            sele_str = 'chainid ' + self.input_params['protein']['chains'] + ' and resid ' + ' '.join(str(resids[res_ind]) for res_ind in matching_res_inds) + ' and backbone'
            ref_sele_str = 'resid ' + ' '.join(str(ref_resids[res_ind]) for res_ind in matching_ref_res_inds) + ' and backbone'
            
            # Align
            _, _ = alignto(mobile=u, 
                    reference=ref,
                    select={'mobile': sele_str,
                          'reference': ref_sele_str})
            
            # Save 
            u.select_atoms('all').write(os.path.join(self.aligned_input_dir, pdb))

        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//WARNING: Cannot find input structure:', input_pdb_path, "Skipping for now...", flush=True)  

    
    def _separate_lig_prot(self):
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
        pdb_fn = self.input_params['protein']['input_pdb']
        pdb_path = os.path.join(self.aligned_input_dir, pdb_fn)
        u = mda.Universe(pdb_path)

        # Select protein by chain ID 
        prot_sele = u.select_atoms(f'protein and chainid {self.input_params["protein"]["chains"]}')
        prot_sele.write(os.path.join(self.prot_only_dir, pdb_fn))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated chain', self.input_params["protein"]["chains"], 'from input structure', flush=True)  

        # Select ligand by resname or peptide_chain
        lig_resname = self.input_params['ligand']['lig_resname']
        peptide_chain = self.input_params['ligand']['peptide_chain']
        assert (lig_resname == False and peptide_chain != False) or (lig_resname != False and peptide_chain == False), f"Either lig_resname or peptide_chain must be False."
        if lig_resname != False:
            lig_sele = u.select_atoms(f'resname {lig_resname}')
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated ligand', lig_resname, 'from input structure', flush=True)
        elif peptide_chain != False:
            lig_sele = u.select_atoms(f'chainid {peptide_chain}')
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Separated ligand', peptide_chain, 'from input structure', flush=True)
        lig_sele.write(os.path.join(self.lig_only_dir, pdb_fn))

    def _repair_crystal(self):
        """
        Uses the RepairProtein class to replace missing residues with UCSF Modeller. 
        """
        params = self.input_params['RepairProtein']
        file = self.input_params['protein']['input_pdb']
        if 'secondary_template' in self.input_params['RepairProtein']:
            secondary_temp = self.input_params['RepairProtein']['secondary_template']
            if secondary_temp!= False and os.path.exists(secondary_temp):
                protein_reparer = RepairProtein(pdb_fn=os.path.join(self.prot_only_dir, file),
                                                                fasta_fn=params['fasta_path'], 
                                                                working_dir=params['working_dir'])
                protein_reparer.run_with_secondary(pdb_out_fn=os.path.join(self.prot_only_dir, file),
                                                  secondary_template_pdb=secondary_temp,
                                                  tails=params['tails'],
                                                  loops=params['loops'])
            else:
                protein_reparer = RepairProtein(pdb_fn=os.path.join(self.prot_only_dir, file),
                                                                fasta_fn=params['fasta_path'], 
                                                                working_dir=params['working_dir'])
                protein_reparer.run(pdb_out_fn=os.path.join(self.prot_only_dir, file),
                                    tails=params['tails'],
                                    loops=params['loops'])
        else:
            protein_reparer = RepairProtein(pdb_fn=os.path.join(self.prot_only_dir, file),
                                                            fasta_fn=params['fasta_path'], 
                                                            working_dir=params['working_dir'])
            protein_reparer.run(pdb_out_fn=os.path.join(self.prot_only_dir, file),
                                tails=params['tails'],
                                loops=params['loops'],
                                verbose=self.verbose)
                                    
    def _add_environment(self, pH: float=7.0, membrane: bool=False, ion_strength: float=0.15):
        """
        Add water, lipids, and hydrogens to protein with the ProteinPreparer class.
        """
        # See if environment parameters are present
        if "environment" in self.input_params.keys():
            if "pH" in self.input_params["environment"].keys():
                pH = self.input_params["environment"]["pH"]
            if "membrane" in self.input_params["environment"].keys():
                membrane = self.input_params["environment"]["membrane"]
            if "ion_strength" in self.input_params["environment"].keys():
                ion_strength = self.input_params["environment"]["ion_strength"]
        
        # Iterate through proteins
        self.prot_pdbs = []
        pdb_fn = self.input_params['protein']['input_pdb']
        pdb_path = os.path.join(self.prot_only_dir, pdb_fn)
        if membrane:
            pp = ProteinPreparer(pdb_path=pdb_path,
                                 working_dir=self.prot_only_dir,
                                 pH=pH,
                                 env='MEM',
                                 ion_strength=ion_strength)
        else:
            pp = ProteinPreparer(pdb_path=pdb_path,
                                 working_dir=self.prot_only_dir,
                                 pH=pH,
                                 env='SOL',
                                 ion_strength=ion_strength)            
        self.prot_pdbs.append(pp.main())

        # Remove unecessary .pqr and .log files
        for fn in os.listdir(self.prot_only_dir):
            if fn.endswith('.log') or fn.endswith('.pqr'):
                os.remove(os.path.join(self.prot_only_dir, fn))
    
    def _ligand_prep(self, out_fm: str='sdf'):
            """ 
            Prepare ligand for OpenFF parameterization.

            Parameters:
            -----------
                out_fm (Str):
                    String of extension of intended format to write out. Default is .sdf. 
            """
            # Iterate through ligands
            lig_fn = self.input_params['protein']['input_pdb']        

            # Get crystal information from protein
            crys_line = open(os.path.join(self.prot_only_dir, lig_fn.split('.')[0]+'_env.pdb'), 'r').readlines()[1]
            assert crys_line.startswith('CRYS'), f"No crystal line found at top of {lig_fn.split('.')[0]}_env.pdb, found: {crys_line}."

            # Get path to input ligand file
            mol_path = os.path.join(self.lig_only_dir, lig_fn)

            # Use obabel if small molecule to covert to .sdf
            lig_resname = self.input_params['ligand']['lig_resname']
            if lig_resname != False:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found small molecule ligand with resname:', lig_resname, flush=True)

                #Obabel conversion block
                obConversion = openbabel.OBConversion()
                formats = [mol_path.split('.')[-1], out_fm]
                obConversion.SetInAndOutFormats(*formats)
                mol = openbabel.OBMol()
        
                #Find Input
                if os.path.isfile(mol_path):
                    obConversion.ReadFile(mol, mol_path)
                elif os.path.isfile(os.path.join(self.abs_work_dir, mol_path)):
                    obConversion.ReadFile(mol, os.path.join(self.abs_work_dir, mol_path))
                else:
                    raise FileNotFoundError('mol_fn was not found')
                    
                #Add Hydrogens
                mol.AddHydrogens()
                            
                #Writeout the protonated file in the second format
                out_fn = mol_path.split('.')[0] + '.sdf'
                obConversion.WriteFile(mol, out_fn)

                # Recursively written over original file type
                if not os.path.exists(mol_path.split('.')[0] + '.pdb'):
                    self._ligand_prep(out_fm=mol_path.split('.')[-1])
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved prepared ligand to', mol_path.split('.')[0] + '.pdb', mol_path.split('.')[0] + '.sdf', flush=True)

            # Prepare peptide ligand
            chain_id = self.input_params['ligand']['peptide_chain']
            if chain_id != False:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found peptide ligand with chainid:', chain_id, flush=True)
                if "environment" in self.input_params.keys():
                    if "pH" in self.input_params["environment"].keys():
                        pH = self.input_params["environment"]["pH"]
                    else:
                        pH = 7.0
                if "peptide_nonstandard_resids" in self.input_params['ligand']:
                    pep_nonstandard_resids = self.input_params['ligand']['peptide_nonstandard_resids']
                else:
                    pep_nonstandard_resids = None
                print('!!!pep_nonstandard_resids', pep_nonstandard_resids)
                #Repair with RepairProtein
                if 'peptide_fasta' in self.input_params['ligand']:
                    pep_fasta = self.input_params['ligand']['peptide_fasta']
                    if pep_fasta != False:
                        protein_reparer = RepairProtein(pdb_fn=mol_path,
                                                fasta_fn=pep_fasta, 
                                                working_dir=self.input_params['RepairProtein']['working_dir'])
                        protein_reparer.run(pdb_out_fn=mol_path,
                                            tails=False,
                                            nstd_resids=pep_nonstandard_resids,
                                            loops=False,
                                            verbose=self.verbose)

                # Protonate 
                if pep_nonstandard_resids == None:
                    pp = ProteinPreparer(pdb_path=mol_path,
                             working_dir=self.lig_only_dir,
                             pH=pH,
                             env='SOL',
                             ion_strength=0) 
                    prot_mol_path = pp._protonate_with_pdb2pqr()
                    print(prot_mol_path)
                    os.rename(prot_mol_path, mol_path)
                    for fn in os.listdir(self.lig_only_dir):
                        if fn.endswith('log') or fn.endswith('pqr'):
                            os.remove(os.path.join(self.lig_only_dir, fn))

                else:
                   # Neutralize C-terminus
                    if "neutral_C-term" in self.input_params['ligand']:
                        if self.input_params['ligand']['neutral_C-term'] == True:
                            pdb_lines = open(mol_path, 'r').readlines()
                            oxt_line = ''
                            for line in pdb_lines:
                                if line.find('OXT') != -1:
                                    oxt_line = line
            
                            nxt_line = [c for c in oxt_line]
                            nxt_line[13] = 'N'
                            nxt_line[-4] = 'N'
                            nxt_line = ''.join(nxt_line)
        
                            with open(mol_path, 'w') as f:
                                for line in pdb_lines:
                                    if line.find('OXT') == -1:
                                        f.write(line)
                                    else:
                                        f.write(nxt_line)

                    #Obabel conversion block
                    obConversion = openbabel.OBConversion()
                    formats = [mol_path.split('.')[-1], out_fm]
                    obConversion.SetInAndOutFormats(*formats)
                    mol = openbabel.OBMol()
            
                    #Find Input
                    if os.path.isfile(mol_path):
                        obConversion.ReadFile(mol, mol_path)
                    elif os.path.isfile(os.path.join(self.abs_work_dir, mol_path)):
                        obConversion.ReadFile(mol, os.path.join(self.abs_work_dir, mol_path))
                    else:
                        raise FileNotFoundError('mol_fn was not found')
                        
                    #Add Hydrogens
                    mol.AddHydrogens()
                                
                    #Writeout the protonated file in the second format
                    out_fn = mol_path.split('.')[0] + '.sdf'
                    obConversion.WriteFile(mol, out_fn)
    
                    # Recursively written over original file type
                    if not os.path.exists(mol_path.split('.')[0] + '.pdb'):
                        self._ligand_prep(out_fm=mol_path.split('.')[-1])
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved prepared ligand to', mol_path.split('.')[0] + '.pdb', mol_path.split('.')[0] + '.sdf', flush=True)

 

            # Add crys line
            lig_lines = open(os.path.join(self.lig_only_dir, lig_fn), 'r').readlines()
            lig_lines.insert(0, crys_line)
            with open(os.path.join(self.lig_only_dir, lig_fn), 'w') as f:
                for line in lig_lines:
                    f.write(line)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved prepared ligand to', os.path.join(self.lig_only_dir, lig_fn), flush=True)
            f.close()

    def _generate_systems(self):
        """
        Generate forcefields with OpenFF using the ForcefieldHandler and OpenMMJoiner classes.
        """
        # Create systems dir
        self.sys_dir = os.path.join(self.working_dir, 'systems')
        if not os.path.exists(self.sys_dir):
            os.mkdir(self.sys_dir)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Created systems directory:', self.sys_dir, flush=True)
        
        # Get names for file to iterate through
        name = self.input_params['protein']['input_pdb'].split('.')[0] 
        
        # Iterate through files
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Building parameters for', name, flush=True)
        # Generate protein system
        prot_path = os.path.join(self.prot_only_dir, name+'_env.pdb')
        assert os.path.exists(prot_path), f"Cannot find path to protein file in environment: {prot_path}"
        prot_sys, prot_top, prot_pos = ForceFieldHandler(prot_path).main()
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Protein parameters built.', flush=True)

        # Generate ligand system
        if self.input_params['ligand']['lig_resname'] != False:
            lig_path = os.path.join(self.lig_only_dir, name+'.sdf')
        elif self.input_params['ligand']['peptide_chain'] != False:
            if "peptide_nonstandard_resids" in self.input_params['ligand']:
                if self.input_params['ligand']['peptide_nonstandard_resids'] != False:
                    lig_path = os.path.join(self.lig_only_dir, name+'.sdf')
                else:    
                    lig_path = os.path.join(self.lig_only_dir, name+'.pdb')
            else:    
                lig_path = os.path.join(self.lig_only_dir, name+'.pdb')

        assert os.path.exists(lig_path), f"Cannot find path to ligand file: {lig_path}"
        lig_sys, lig_top, lig_pos = ForceFieldHandler(lig_path).main()
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Ligand parameters built.', flush=True)

        # Combine systems 
        sys, top, pos = Joiner((lig_sys, lig_top, lig_pos),
                               (prot_sys, prot_top, prot_pos)).main()
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'System parameters built.', flush=True)

        # Reposition at origin
        box_vectors = sys.getDefaultPeriodicBoxVectors()
        translate = Quantity(np.array((box_vectors[0].x,
                                       box_vectors[1].y,
                                       box_vectors[2].z))/2,
                             unit=nanometer)

        # Get energy
        int = LangevinIntegrator(300 * kelvin, 1/picosecond, 0.001 * picosecond)
        sim = Simulation(top, sys, int)
        sim.context.setPositions(pos + translate)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Initial structure potential energy:', np.round(sim.context.getState(getEnergy=True).getPotentialEnergy()._value, 2), flush=True)

        # Write out
        with open(os.path.join(self.sys_dir, name+'.pdb'), 'w') as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        with open(os.path.join(self.sys_dir, name+'.xml'), 'w') as f:
            f.write(XmlSerializer.serialize(sim.system))
        
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Final system coordinates saved to', os.path.join(self.sys_dir, name+'.pdb'), flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Final system parameters saved to', os.path.join(self.sys_dir, name+'.xml'), flush=True)

    

        




                    

    
    
    
    
    
                    
    
                                        
    
    
    
                
    
    
    
