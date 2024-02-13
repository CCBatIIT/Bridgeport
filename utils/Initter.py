import textwrap, sys, os, glob, shutil
import numpy as np
import mdtraj as md
import MDAnalysis as mda
from openbabel import openbabel
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *


class Initter():
    """
    A class for getting the input files ready for further processing
    A given structure file must be split into a pdb file of just the protein, as well as an sdf file of the ligand.
    The ligand is protonated at this step, but the protein is not.
    """
    def __init__(self, pdb_fn, ligand_resname, working_directory, align_to=None):
        """
        Parameters:
            pdb_fn - The pdb file of the protein (crystal structure pdb file)
            ligand_resname - declaration of the ligand resname
            align_to - By default no alignment, if provided as pdb file, will align
        """
        self.pdb_fn = pdb_fn
        self.ligand_string = ligand_resname
        self.abs_work_dir = os.path.join(os.getcwd(), working_directory)
        if not os.path.isdir(self.abs_work_dir):
            os.mkdir(self.abs_work_dir)
    
    def main(self):
        #receptor_path, ligand_path = self._seperate_crys_ligand(self.pdb_fn,
        #                                                        self.ligand_string)
        receptor_path, ligand_path = self._seperate_crys_using_MDA(self.pdb_fn,
                                                                   self.ligand_string)
        
        print(receptor_path, os.path.isfile(receptor_path))
        print(ligand_path, os.path.isfile(ligand_path))
        
        ligand_sdf_path, ligand_protonated = self._obabel_ligand_prep(ligand_path,
                                                                      ['pdb', 'sdf'],
                                                                      out_fn='AUTO',
                                                                      add_Hs=True,
                                                                      rewrite_with_Hs=True)
        print(ligand_protonated, os.path.isfile(ligand_protonated))
        print(ligand_sdf_path, os.path.isfile(ligand_sdf_path))

        return receptor_path, ligand_sdf_path
    
    def _seperate_crys_ligand(self,
                              crys_pdb_fn: str,
                              ligand_string: str,
                              align_to: str = None,
                              receptor_string: str = 'protein',
                              ligand_pdb_fn: str = 'ligand_crys.pdb',
                              receptor_pdb_fn: str = 'receptor_crys.pdb'):
        """
        Loads the provided crystal structure, and seperates the atoms of resname LIGAND STRING into their own file
        Arguments:
            crys_pdb_fn: crystal structure to be seperated
            ligand_string: MDAnalysis selection string for ligand
            receptor_string: MDAnalysis selection string for receptor (default 'protein')
            ligand_pdb_fn: Write path for MDAnalysis ligand (default 'ligand.pdb')
            receptor_pdb_fn: Write path for MDAnalysis receptor (default 'receptor.pdb')
        Returns:
            receptor_pdb_fn: filename of the pdb file containing the receptor only
            ligand_pdb_fn: filename of the pdb file containing the atoms from the LIGAND_STRING selection
        """
        
        u = md.load(crys_pdb_fn)
        
        if align_to is not None:
            ref = md.load(align_to)
            u = u.superpose(ref)

        ligand_indices = u.topology.select(self.ligand_string)
        protein_indices = u.topology.select(receptor_string)

        lig_top = u.topology.subset(ligand_indices)
        rec_top = u.topology.subset(protein_indices)

        lig_pos = u.xyz[:, ligand_indices, :]
        rec_pos = u.xyz[:, protein_indices, :]

        ligand_pdb_fn = os.path.join(self.abs_work_dir, ligand_pdb_fn)
        receptor_pdb_fn = os.path.join(self.abs_work_dir, receptor_pdb_fn)
        
        with md.formats.PDBTrajectoryFile(ligand_pdb_fn, 'w') as f:
            f.write(lig_pos[0] *10, lig_top)

        with md.formats.PDBTrajectoryFile(receptor_pdb_fn, 'w') as g:
            g.write(rec_pos[0]*10, rec_top)

        return receptor_pdb_fn, ligand_pdb_fn

    def _seperate_crys_using_MDA(self,
                                 crys_pdb_fn: str,
                                 ligand_string: str,
                                 receptor_string: str = 'protein',
                                 ligand_pdb_fn: str = 'ligand_crys.pdb',
                                 receptor_pdb_fn: str = 'receptor_crys.pdb'):
        """
        Loads the provided crystal structure, and seperates the atoms of resname LIGAND STRING into their own file
        Arguments:
            crys_pdb_fn: crystal structure to be seperated
            ligand_string: MDAnalysis selection string for ligand
            receptor_string: MDAnalysis selection string for receptor (default 'protein')
            ligand_pdb_fn: Write path for MDAnalysis ligand (default 'ligand.pdb')
            receptor_pdb_fn: Write path for MDAnalysis receptor (default 'receptor.pdb')
        Returns:
            receptor_pdb_fn: filename of the pdb file containing the receptor only
            ligand_pdb_fn: filename of the pdb file containing the atoms from the LIGAND_STRING selection
        """
        u = mda.Universe(crys_pdb_fn)
        all_atoms = u.select_atoms('all')
        receptor = all_atoms.select_atoms(receptor_string)
        ligand = all_atoms.select_atoms(ligand_string)
        receptor.write(os.path.join(self.abs_work_dir, receptor_pdb_fn))
        ligand.write(os.path.join(self.abs_work_dir, ligand_pdb_fn))
        return os.path.join(self.abs_work_dir, receptor_pdb_fn), os.path.join(self.abs_work_dir, ligand_pdb_fn)

    def _obabel_ligand_prep(self,
                            mol_fn: str,
                            formats: list,
                            out_fn: str = 'AUTO',
                            add_Hs: bool = True,
                            rewrite_with_Hs: bool = True):
        """
        Convert a file to another format using openbabel.  Neither add_Hs, nore rewrite_with_Hs should be True if the input has hydrogens.
        Arguments:
            mol_fn: in_file_name : (THis should be infile.xxx where xxx = formats[0])
            formats: 2-list of babel formats (in, out)
            out_fn: Filename for the ligand with Hydrogens (if AUTO, it will be MOL_FN with the extension swapped to format[-1])
            add_Hs: Bool: Whether adding hydrogens to the input file should be done (default True)
            rewrite_with_Hs: Bool: Additionally rewrites the protonated ligand in the original file format (default True)
        Returns:
            str: The path of the converted file
        Example:
            obabel_conversion('ligand.pdb', ['pdb', 'sdf']) will attempt to protonate (by default) and convert a pdb file to sdf (named ligand.sdf)
        """
        #Assertion checks
        assert len(formats) == 2
        #Obabel conversion block
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(*formats)
        mol = openbabel.OBMol()
        #Find INput
        if os.path.isfile(mol_fn):
            obConversion.ReadFile(mol, mol_fn)
        elif os.path.isfile(os.path.join(self.abs_work_dir, mol_fn)):
            obConversion.ReadFile(mol, os.path.join(self.abs_work_dir, mol_fn))
        else:
            raise FileNotFoundError('mol_fn was not found')
        #Add Hydrogens
        if add_Hs:
            mol.AddHydrogens()
        print(mol.NumAtoms(), 'Atoms', mol.NumBonds(), 'Bonds', mol.NumResidues(), 'Residues')
        #Output file name parsing
        if out_fn == 'AUTO':
            out_fn = os.path.splitext(mol_fn)[0] + '.' + formats[-1]
        #Actually writeout the protonated file in the second format
        obConversion.WriteFile(mol, out_fn)
        
        #Rewrite original format with Hydrogens (if necessary)
        if rewrite_with_Hs:
            #recursively use this function to convert from format 0 to format 0 again
            org_form_new_fn = os.path.splitext(mol_fn)[0] + '_H.' + formats[0]
            org_form_wHs_fn = self._obabel_ligand_prep(mol_fn, [formats[0], formats[0]], out_fn=org_form_new_fn, add_Hs=True, rewrite_with_Hs=False)[0]
            return (out_fn, org_form_wHs_fn)
        else:
            return (out_fn, None)