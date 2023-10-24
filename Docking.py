import os
from typing import List

class Docking():
    """
    Docks ligands and protein structures using AutoDock Vina 1.2.3

    Conda environments:
    -------------------
        "vina":
            Conda enviroment named "vina" that contains Vina 1.2.3 and the required dependecies to run Vina.
        "mgltools":
            Conda environment name "mgltools" that contains mgltools 1.5.7

    Attributes:
    -----------

    Methods:
    --------

    """

    def __init__(self, pdb_path: str, lig_paths: List[str]):
        """
        Initialize Docking object.

        Parameters:
        -----------
            pdb_path (str):
                String containing path to protein structure (pdb) to use for docking. 

            lig_path (List[str]):
                List of strings contraining path to ligand structure to use for docking. 
        """

        # Set attributes
        self.input_prot_path = pdb_path
        self.input_lig_paths = lig_paths

        # Convert to pdbqt 
        self.receptor_path = './receptor.pdbqt'
        os.system(f'prepare_receptor4.py -r {self.input_prot_path} -o {self.receptor_path} -C -U nphs_lps -v')
        
        




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

