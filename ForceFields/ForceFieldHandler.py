import textwrap, sys, os, pathlib, json, warnings
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem

#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *


class ForceFieldHandler():
    """
    A Class for parameterization of a structure file. Returns the necessary elements to construct an openmm simulation, System, Topology, and Positions.
    SDF File - Ligand to be parameterized with OpenFF
    PDB File - Environment File to be parameterized with Amber FF14SB, Lipid17, OPC3 XML Files
    - Override the defaults with optional user specified files. Extensions must be offxml for an sdf file or xml for a pdb file.
    
    Default Usage:
    --------------
        system, topology, positions = ForceFieldHandler(INPUTS).main()
    
    Parameters:
    -----------
        structure_file - Input, either and an sdf or pdb file
        force_field_files - Optional Input - offxml or xml files. Can use included forcefields with either OpenFF or OpenMM
    
    Returns:
    -------
        system, topology, positions as a 3-tuple
    
    Attributes:
    -----------
        default_xmls (dict): 
            Default XML files for OpenFF and OpenMM.
            
        structure_file (str): 
            Input file path, either an SDF or PDB file.
            
        working_mode (str): 
            Mode of operation based on file extension, either 'OpenFF' or 'OpenMM'.
            
        xmls (list): 
            List of XML files to be used for parameterization.
    
    Methods:
    --------
        __init__(self, structure_file, force_field_files=None, use_defaults: bool=True): 
            Initializes the ForceFieldHandler object with a structure file and optional force field files.
            
        _parse_file(self, file_fn): 
            Determines the working mode ('OpenFF' or 'OpenMM') based on the file extension.
        
        main(self, use_rdkit: bool=False): 
            Main method for parameterization. Returns a tuple containing the system, topology, and positions.
        
        generate_custom_xml(self, out_xml, name): 
            Generates a custom XML file for parameterization.
        
        neutralizeMol(mol): 
            Neutralizes the given molecule by setting radical electrons and formal charges to zero.
    """
    
    def __init__(self, structure_file, force_field_files=None, use_defaults: bool=True):
        """
        Parameters:
            structure_file: string: Path to an SDF or PDB file containing the structure to be parameterized.
            force_field_files: [string]: Default Nonetype - provide a list of strings to use user-defined force field files
            use_defaults: bool: Default True - change to False when providing user defined force field files (when force_field_files != None)
        Returns:
            None
        """
        self.default_xmls = {'OpenFF': ['openff-2.1.0.offxml'], 
                        'OpenMM': ['amber14/protein.ff14SB.xml', 
                                   'amber14/lipid17.xml',
                                   f'{pathlib.Path(__file__).parent.resolve()}/wat_opc3.xml']}
        self.structure_file = structure_file
        
        # Parse the structure file to see if the user is in OpenFF or OpenMM mode
        self.working_mode = self._parse_file(structure_file)
        
        #If force field files were provided, check their extensions, if not use the default
        if force_field_files is None and use_defaults == True:
            self.xmls = self.default_xmls[self.working_mode]
        elif type(force_field_files) != list:
            raise Exception('force_field_files parameter must be specified as a list of strings')
        else:
            mode_parse = [self._parse_file(ff_file) for ff_file in force_field_files]
            mode_check = [elem == self.working_mode for elem in mode_parse]
            if use_defaults == True:
                self.xmls = self.default_xmls['OpenMM']
            else:
                self.xmls = []
            for ff in force_field_files:
                self.xmls.insert(0, ff)

            if False in mode_check:
                bad_index = mode_check.index(False)
                bad_file = force_field_files[bad_index]
                raise Exception(f'{bad_file} was found incompatible with the structure file')

    def _parse_file(self, file_fn):
        """
        A short method to determine the type of force field file to use based on the extension of the structure file provided.
        Currently SDF structure files -> OFFXML (OpenFF) forcefield files
                  PDB structure files -> XML (OpenMM) forcefield files
        Current support is only for these two files types

        Parameters:
            file_fn: string: path to the structure file
        Returns
            mode: string: either "OpenFF" or "OpenMM", this string indicates which type of default force field to use
        """
        ext = os.path.splitext(file_fn)[-1]
        supported_openff_types = ['.sdf', '.offxml']
        supported_openmm_types = ['.pdb', '.xml']

        if ext in supported_openff_types:
            mode = 'OpenFF'
        elif ext in supported_openmm_types:
            mode = 'OpenMM'
        else:
            raise Exception(f'The extension {ext} was not recognized!')
        return mode

    def main(self, use_nonbonded: bool=True):
        """
        The intended main usage case.  Parameterize ligands from an SDF file with OpenFF (.offxml) parameters and
        environment/protein from a PDB file with OpenMM (.xml) parameters.

        Paremeters:
            use_rdkit: bool: Default=False - When a custom force field for the ligand is necessary, and the ligand is 
                a pdb file, this should be True.  Takes an intermediate step to load an RDKit molecule from the pdb file
                and then load an OpenFF molecule from the RDKit molecule - as opposed to loading the OpenFF molecule directly
                from the structure file.
        Returns:
            (sys, top, positions): tuple - A 3-tuple of OpenMM System, OpenMM Topology, and coordinate array of positions
        """
        if self.working_mode == 'OpenFF':
            try:
                mol = openff.toolkit.Molecule.from_file(self.structure_file)
            except:
                warnings.warn('WARNING: ATTEMPTING TO ALLOW UNDEFINED STEREOCHEMISTRY. CHECK OUTPUT STRUCTURE CAREFULLY')
                mol = openff.toolkit.Molecule.from_file(self.structure_file, allow_undefined_stereo=True)
            ff = openff.toolkit.ForceField(*self.xmls)
            cubic_box = openff.units.Quantity(30 * np.eye(3), openff.units.unit.angstrom)
            self.interchange = openff.interchange.Interchange.from_smirnoff(topology=[mol], force_field=ff, box=cubic_box)
            positions = np.array(self.interchange.positions) * nanometer
            sys = self.interchange.to_openmm_system()
            top = self.interchange.to_openmm_topology()
    
        elif self.working_mode == 'OpenMM':
            ff = ForceField(*self.xmls)
            pdb = PDBFile(self.structure_file)
            top, positions = pdb.getTopology(), pdb.getPositions()
            if use_nonbonded:
                sys = ff.createSystem(top, nonbondedMethod=PME) 
            else:
                sys = ff.createSystem(top)

        return (sys, top, positions)
