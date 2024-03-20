import textwrap, sys, os, pathlib, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

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
    A Class for parameterization of a structure file. Returns the necessary elements to construct an openmm simulation, System, Topology, and Positions. SDF File - Ligand to be parameterized with OpenFF PDB File - Environment File to be parameterized with Amber FF14SB, Lipid17, OPC3 XML Files - Override the defaults with optional user specified files. Extensions must be offxml for an sdf file or xml for a pdb file.
    
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
        Supported formats (SDF, PDB)
        Currently parses sdf files as openff format and pdb files for openmm
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

    def main(self, use_rdkit: bool=False):
        """
        The intended main usage case is to parameterize ligands with an SDF file and openff parameters
        and do do a protein, lipid, solvent system with openmm parameters.
        """
        if self.working_mode == 'OpenFF':
            if use_rdkit:
                rdkit_mol = Chem.MolFromPDBFile(self.structure_file, removeHs=False, proximityBonding=False)
                display(rdkit_mol)
                mol = openff.toolkit.Molecule.from_rdkit(rdkit_mol, hydrogens_are_explicit=True, allow_undefined_stereo=True)
            else:
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
            sys = ff.createSystem(top, nonbondedMethod=PME) # Make this an adjustable parameter later

        return (sys, top, positions)

    def generate_custom_xml(self, out_xml: str, name: str):
        """
        Generate a custom .xml to pass to Handler.

        Parameters:
        -----------
            out_xml (str): 
                String path to output.xml file. 
        """

        # Invoke main to create interchange object
        self.working_mode = 'OpenFF'
        self.xmls = self.default_xmls[self.working_mode]
        _, _, _ = self.main(use_rdkit=True)

        # Write out to .prmtop
        out_pre = out_xml.split('.xml')[0]
        out_prmtop = out_pre + '.prmtop'
        self.interchange.to_prmtop(out_prmtop)

        # Write write_xml_pretty_input.json
        input_json = f'{pathlib.Path(__file__).parent.resolve()}/write_xml_pretty_input.json'
        data = json.load(open(input_json, 'r'))

        data['fname_prmtop'] = out_prmtop
        data['fname_xml'] = out_xml
        data['ff_prefix'] = str(name)

        json.dump(data, open(input_json, 'w'), indent=6)
        
        # Use write_xml_pretty.py to convert .prmtop to .xml
        os.system(f'python {pathlib.Path(__file__).parent.resolve()}/write_xml_pretty.py -i {input_json}')  

# def neutralizeMol(mol):
#     for a in mol.GetAtoms():
#         if a.GetNumRadicalElectrons()==1:
#              a.SetNumRadicalElectrons(0)         
#         if a.GetFormalCharge()!=0:
#              a.SetFormalCharge(0)         
#     return mol
