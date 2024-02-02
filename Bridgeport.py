import os, sys, json
from datetime import datetime
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
sys.path.append('RepairProtein')
import numpy as np
from ProteinPreparer import ProteinPreparer
from RepairProtein import RepairProtein

class Bridgeport():
    """
    Master class to prepare crystal structures for OpenMM simulation. 

    Attributes:
    -----------

    Methods:
    --------
    """

    def __init__(self, input_json: str):
        """
        Initialize Bridgeport objects.

        Parameters:
        -----------
            input_json (str):
                String path to .json file that contains inputs

        Returns:
        --------
            Bridgeport object.
        """
        # Make assertions
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Welcome to Bridgeport.', flush=True)
        assert os.path.exists(input_json), "Cannot find input_json."
        assert input_json.split('.')[1] == 'json', f"input_json: {input_json} is not a type .json."

        # Load input from json
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

    def _align(self):
        """
        Align initial structure to a reference structure. 
        The reference structure can include a structure from the OPM database for transmembrane proteins.
        """
        # Create directory for aligned proteins 
        self.aligned_input_dir = os.path.join(self.working_dir, 'aligned_input_pdbs')
        if not os.path.exists(self.aligned_input_dir):
            os.mkdir(self.aligned_input_dir)

        # Load reference structure
        ref = mda.Universe(self.input_params['environment']['alignment_ref'])
        ref_resids = ref.residues.resids
        
        # Load structure to align
        input_pdb_dir = self.input_params['protein']['input_pdb_dir']
        for i, pdb in enumerate(self.input_params['protein']['input_pdbs']):
            input_pdb_path = os.path.join(input_pdb_dir, pdb)

            # Slim to correct chain
            u = mda.Universe(input_pdb_path)
            chain_sele = u.select_atoms('chainid ' + self.input_params['protein']['chains'][i])

            # Get resids
            resids = chain_sele.residues.resids

            # Find matching resids
            matching_resids, matching_res_inds, matching_ref_res_inds = np.intersect1d(resids, ref_resids, return_indices=True)
            # print('!!!', matching_resids, matching_res_inds, matching_ref_res_inds)
            # sele = u.select_atoms('chainid ' + self.input_params['protein']['chains'][i] + ' and resid ' + ' '.join(str(resids[res_ind]) for res_ind in matching_res_inds) + ' and backbone')
            sele_str = 'chainid ' + self.input_params['protein']['chains'][i] + ' and resid ' + ' '.join(str(resids[res_ind]) for res_ind in matching_res_inds) + ' and backbone'
            # ref_sele = ref.select_atoms('resid ' + ' '.join(str(ref_resids[res_ind]) for res_ind in matching_ref_res_inds) + ' and backbone')
            ref_sele_str = 'resid ' + ' '.join(str(ref_resids[res_ind]) for res_ind in matching_ref_res_inds) + ' and backbone'
            
            # Align
            _, _ = alignto(mobile=u, 
                    reference=ref,
                    select={'mobile': sele_str,
                          'reference': ref_sele_str})
            
            # Save 
            u.select_atoms('all').write(os.path.join(self.aligned_input_dir, pdb))

    def _separate_lig_prot(self):
        """
        Separate ligand and protein based on chain and resname specified in input file.
        """
        # Create directories for separated .pdb files
        self.prot_only_dir = os.path.join(self.working_dir, 'proteins')
        if not os.path.exists(self.prot_only_dir):
            os.mkdir(self.prot_only_dir)
        self.lig_only_dir = os.path.join(self.working_dir, 'ligands')
        if not os.path.exists(self.lig_only_dir):
            os.mkdir(self.lig_only_dir)        

        # Iterate through input files
        for i, pdb_fn in enumerate(self.input_params['protein']['input_pdbs']):
            pdb_path = os.path.join(self.aligned_input_dir, pdb_fn)
            u = mda.Universe(pdb_path)

            # Select protein by chain ID 
            prot_sele = u.select_atoms(f'protein and chainid {self.input_params["protein"]["chains"][i]}')
            prot_sele.write(os.path.join(self.prot_only_dir, pdb_fn))

            # Select ligand by resname or peptide_chain
            lig_resname = self.input_params['ligand']['lig_resnames'][i]
            peptide_chain = self.input_params['ligand']['peptide_chains'][i]
            assert (lig_resname == False and peptide_chain != False) or (lig_resname != False and peptide_chain == False), f"Either lig_resnames or peptide_chains must be False at indice {i}."
            if lig_resname != False:
                lig_sele = u.select_atoms(f'resname {lig_resname}')
            elif peptide_chain != False:
                lig_sele = u.select_atoms(f'chainid {peptide_chain}')
            lig_sele.write(os.path.join(self.lig_only_dir, pdb_fn))

    def _repair_crystal(self):
        """
        Uses the RepairProtein class to replace missing residues with UCSF Modeller. 
        """
        params = self.input_params['RepairProtein']
        for i, file in enumerate(self.input_params['protein']['input_pdbs']):
                protein_reparer = RepairProtein(pdb_fn=os.path.join(self.prot_only_dir, file),
                                                fasta_fn=params['fasta_path'], 
                                                working_dir=params['working_dir'])
                protein_reparer.run(pdb_out_fn=os.path.join(self.prot_only_dir, file),
                                    tails=params['tails'],
                                    loops=None)
                                    #loops=params['loops'])
                
    def _add_environment(self, pH: float=7.0, membrane: bool=False, ion_strength: float=0.15):
        """
        Add water, lipids, and hydrogens to protein.
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
        for i, prot_pdb_fn in enumerate(os.listdir(self.prot_only_dir)):
            if membrane:
                pp = ProteinPreparer(protein_pdb_fn=prot_pdb_fn,
                                 pH=pH,
                                 env='MEM',
                                 ion_strength=ion_strength)
            else:
                pp = ProteinPreparer(protein_pdb_fn=prot_pdb_fn,
                                 pH=pH,
                                 env='SOL',
                                 ion_strength=ion_strength)
            
            pp.main()
            





                

                                    



            



