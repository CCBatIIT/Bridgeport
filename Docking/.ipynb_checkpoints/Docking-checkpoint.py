import os, sys
from typing import List
from vina import Vina
import numpy as np
import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
sys.path.append('../utils')
from bp_utils import return_max_common_substructure
import warnings
# suppress some MDAnalysis warnings about PSF files
warnings.filterwarnings('ignore')

class Docking():
    """
    Docks ligands and protein structures using AutoDock Vina 1.2.3

    Conda environments:
    -------------------
        "vina":
            Conda enviroment named "vina" that contains Vina 1.2.3 and the required dependecies to run Vina.

    Attributes:
    rec_path (str):
        Path to the .pdbqt file of the receptor.
    
    lig_path (str):
        Path to the .pdbqt file of the ligand.
    
    config_dir (str):
        Path to the directory to store AutoDock Vina configuration files. Default is './'.
    
    box_center (List[float]):
        3D coordinates [x, y, z] representing the center of the docking box.
    
    box_dim (List[float]):
        Dimensions [x, y, z] representing the size of the docking box in each direction.
    
    lig_out_dir (str):
        Directory path to store the output of the docking process. Defaults to './'.
    
    pdb_dir (str):
        Subdirectory within lig_out_dir where separated .pdb files of docked poses are stored.
    
    scores_dir (str):
        Subdirectory within lig_out_dir where docking scores are stored.
    
    lig_name (str):
        Name of the ligand, derived from lig_path.
    
    Methods:
        init(self, receptor_path: str, ligand_path: str, config_dir: str='./'):
        Initializes a Docking object with specified receptor and ligand paths, and an optional configuration directory.
    
    set_box(self, box_center: List[float], box_dim: List[float]):
        Sets the docking box dimensions and center coordinates.
    
    dock(self, lig_out_dir: str='./', n_poses: int=20, exhaustiveness: int=8, min_rmsd: float=1.0):
        Performs the docking operation. Outputs docked poses, separated pdb files, and scores to specified directories. Utilizes a configuration file prepared by _write_autodock.
    
    compare(self, ref_pdb: str, ref_chainid: str, ref_lig_sele_str: str):
        Compares docked poses to a reference structure by calculating the RMSD of the maximum common substructure. Returns an array of RMSD values.
    
    _write_autodock(self, config_dir: str, receptor_path: str, ligand_path: str, lig_name: str, num_poses: int, exhaustiveness: int, min_rmsd: float):
        Internal method to write the AutoDock Vina configuration file based on provided parameters and the docking box coordinates. Returns the path to the configuration file.

    """

    def __init__(self, receptor_path: str, ligand_path: str, config_dir: str='./'):
        """
        Initialize Docking object.

        Parameters:
        -----------
            receptor_path (str):
                String path to .pdbqt file of receptor.

            ligand_path (str):
                String path to .pdbqt file of ligand.

            config_dir (str):
                String path to directory to store Autodock vina configuration files. Default is './'.
        """

        # Set attributes
        self.rec_path = receptor_path
        self.lig_path = ligand_path
        self.config_dir = config_dir

    def set_box(self, box_center: List[float], box_dim: List[float]):
        """
        Set box for docking. 

        Parameters:
        -----------
            box_center (List[float]):
                List of 3-D cartesian coordinates in format [x, y, z] that correspond to the center of the box.
            
            box_dim (List[float]):
                List of vector magnitudes in formax [x, y, z] that correspond to the length of the box vector centered at box_center. 
        """
        
        self.box_center = box_center
        self.box_dim = box_dim

    def dock(self, 
             lig_out_dir: str='./',
             n_poses: int=20,
             exhaustiveness: int=8, 
             min_rmsd: float=1.0):
        """
        Dock ligands to receptors.

        Parameters:
        -----------
            lig_out_dir (str):
                String path to directory to store docked poses. Default is './'.
                    Vina output (.pdbqt) can be found in {lig_out_dir}/pdbqt
                    Separated .pdb files can be found in {lig_out_dir}/pdb
                    Docking scores can be found in {lig_out_dir}/scores

            n_poses (int):
                Number of docked poses to output. Default is 20.

            exhaustiveness (int):
                Level of exhaustiveness. Default is 8. 

            min_rmsd (float):
                To save a new docked pose, the new pose must have a RMSD > min_rmsd compared to all previous poses. Units is Angstrom. Default is 1.0 Angstrom. 
        """
        # Assign attributes
        self.lig_out_dir = lig_out_dir

        # Make necessary directories
        pdbqt_dir = os.path.join(self.lig_out_dir, 'pdbqt')
        if not os.path.exists(pdbqt_dir):
            os.mkdir(pdbqt_dir)
            
        self.pdb_dir = os.path.join(self.lig_out_dir, 'pdb')
        if not os.path.exists(self.pdb_dir):
            os.mkdir(self.pdb_dir)

        self.scores_dir = os.path.join(self.lig_out_dir, 'scores')
        if not os.path.exists(self.scores_dir):
            os.mkdir(self.scores_dir)
            
        # Get receptor name
        rec_name = self.rec_path.split('/')[-1].split('.')[0]
        print("Receptor " + rec_name)
            
        # Get ligand name
        self.lig_name = self.lig_path.split('/')[-1].split('.')[0]
        print("Ligand " + self.lig_name) 
        
        config_path = self._write_autodock(config_dir= self.config_dir,
                             receptor_path=self.rec_path, 
                             ligand_path=self.lig_path,
                             lig_name = self.lig_name,
                             num_poses=n_poses,
                             exhaustiveness=exhaustiveness,
                             min_rmsd=min_rmsd)

        pdbqt_path = os.path.join(pdbqt_dir, self.lig_name + '.pdbqt')
        os.system(f'vina --config {config_path} --out {pdbqt_path}')

        # Save scores 
        scores = []
        with open(pdbqt_path, 'r') as f:
            for line in f:
                if line.find('REMARK VINA RESULT') != -1:
                    scores.append(float(line.split()[3]))
        scores_path = os.path.join(self.scores_dir, self.lig_name + '_scores.txt')
        np.savetxt(scores_path, np.array(scores))

        # Convert output back to pdb
        os.system(f'obabel -ipdbqt {pdbqt_path} -opdb -O {self.pdb_dir}/{self.lig_name}_.pdb -m')

    def compare(self, ref_pdb: str, ref_chainid: str, ref_lig_sele_str):
        """
        Compare the docked poses to a referenc analogue via RMSD of maximum common substructure.

        Parameters:
        -----------
            ref_pdb (str):
                String path to .pdb file with both receptor and ligand used for reference.

            ref_chainid (str):
                Chainid of protein to align to docking structure.

            ref_lig_sele_str (str):
                MDAnalysis selection string to parse the ligand from ref_pdb file. 

        Returns:
        --------
            rmsds (np.array):
                Array of RMSDs of docked pose and reference ligand based on the maximum common substructure.
        """

        # Create MDA selections to align proteins
        docking_u = mda.Universe(self.rec_path)
        docking_prot = docking_u.select_atoms('name CA')
        resids = docking_prot.residues.resids
        assert os.path.exists(ref_pdb), f"Could not find ref_pdb: {ref_pdb}"
        ref_u = mda.Universe(ref_pdb)
        ref_prot = ref_u.select_atoms(f'name CA and chainid {ref_chainid} and resid {" ".join(str(r) for r in resids)}')
        assert ref_prot.n_atoms > 0, f"Parsing ref_pdb: {ref_pdb} with chainid: {ref_chainid} returned an empty selection."

        # Align ref strucutre to structure used for docking
        _, _ = alignto(ref_prot, docking_prot)
        
        #Create rdkit molecules
        ref_sele = ref_u.select_atoms(ref_lig_sele_str)
        assert ref_sele.n_atoms > 0, f"Parsing ref_pdb: {ref_pdb} with selection string: {ref_lig_sele_str} returned an empty selection."
        ref_sele.write('reference_analogue.pdb')
        ref_mol = Chem.MolFromPDBFile('reference_analogue.pdb')
        pose_pdb = os.path.join(self.pdb_dir, self.lig_name + '_1.pdb')
        pose_mol = Chem.MolFromPDBFile(pose_pdb)

        # Get maximum common substructure
        ref_match_inds, pose_match_inds = return_max_common_substructure(ref_mol, pose_mol)
        ref_sele_atoms = [ref_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in ref_match_inds]
        ref_sele_resids = [ref_mol.GetAtoms()[i].GetPDBResidueInfo().GetResidueNumber() for i in ref_match_inds]
        pose_sele_atoms = [pose_mol.GetAtoms()[i].GetMonomerInfo().GetName().strip() for i in pose_match_inds]

        # Get reference selection for MCS
        ref_align_sele = ref_sele.select_atoms('')
        for ref_atom, ref_resid in zip(ref_sele_atoms, ref_sele_resids):
            ref_align_sele = ref_align_sele + ref_sele.select_atoms('resid '+ str(ref_resid) + ' and name '+ ref_atom)

        # Iterate through poses
        n_poses = len(os.listdir(self.pdb_dir))
        rmsds = np.empty(n_poses)
        for i in range(1, n_poses+1):
            pose_path = os.path.join(self.pdb_dir, self.lig_name + '_' + str(i) + '.pdb')

            pose_u = mda.Universe(pose_path)
            pose_align_sele = pose_u.select_atoms('')
            for pose_atom in pose_sele_atoms:
                pose_align_sele = pose_align_sele + pose_u.select_atoms('name ' + pose_atom)
        
            RMSD = rmsd(ref_align_sele.positions.copy(), pose_align_sele.positions.copy())
            rmsds[i-1] = RMSD
            
        # Save rmsds
        rmsds_path = os.path.join(self.scores_dir, self.lig_name + '_rmsds.txt')
        
        return rmsds
    
    def _write_autodock(self, config_dir, receptor_path, ligand_path, lig_name, num_poses, exhaustiveness, min_rmsd):
        config_path = os.path.join(config_dir, lig_name)
        with open(config_path, "w") as f:
            f.write("#CONFIGURATION FILE (options not used are commented) \n")
            f.write("\n")
            f.write("#INPUT OPTIONS \n")
            f.write(f"receptor = {receptor_path} \n")
            f.write(f"ligand = {ligand_path} \n")
            f.write("#flex = [flexible residues in receptor in pdbqt format] \n")
            f.write("#SEARCH SPACE CONFIGURATIONS \n")
            f.write("#Center of the box (values cx, cy and cz) \n")
        # -->CHANGE THE FOLLOWING DATA WITH YOUR BOX CENTER COORDINATES
            f.write(f"center_x = {self.box_center[0]} \n")
            f.write(f"center_y = {self.box_center[1]} \n")
            f.write(f"center_z = {self.box_center[2]} \n")
        # -->CHANGE THE FOLLOWING DATA WITH YOUR BOX DIMENSIONS
            f.write("#Size of the box (values szx, szy and szz) \n")
            f.write(f"size_x = {self.box_dim[0]} \n")
            f.write(f"size_y = {self.box_dim[1]} \n")
            f.write(f"size_z = {self.box_dim[2]} \n")
        #MORE OPTIONS
            f.write("#OUTPUT OPTIONS \n")
            f.write("#out = \n")
            f.write("#log = \n")
            f.write("\n")
            f.write("#OTHER OPTIONS \n")
            f.write("cpu = 4\n")
            f.write(f"exhaustiveness = {exhaustiveness}\n")
            f.write(f"num_modes = {num_poses}\n")
            f.write(f"min_rmsd = {min_rmsd}\n")
            f.write("#energy_range = \n")
            f.write("#seed = ")

        return config_path






