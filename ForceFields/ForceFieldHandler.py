import textwrap, sys, os, pathlib
import numpy as np
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
    A Class for parameterization of a structure file.  Returns the necessary elements to 
    construct an openmm simulation, System, Topology, and Positions.
        SDF File - Ligand to be parameterized with OpenFF
        PDB File - Environment File to be parameterized with Amber FF14SB, Lipid17, OPC3
        XML Files - Override the defaults with optional user specified files.
                    Extensions must be offxml for an sdf file or xml for a pdb file.
        
    Default Usage
        system, topology, positions = ForceFieldHandler(INPUTS).main()
    
    Parameters:
        structure_file - Input, either and an sdf or pdb file
        force_field_files - Optional Input - offxml or xml files.  Can use included forcefields with either OpenFF or OpenMM

    Returns:
        system, topology, positions as a 3-tuple
        
    """
    
    def __init__(self, structure_file, force_field_files=None):
        default_xmls = {'OpenFF': ['openff-2.1.0.offxml'], 
                        'OpenMM': ['amber14/protein.ff14SB.xml', 
                                   'amber14/lipid17.xml', 
                                   f'{pathlib.Path(__file__).parent.resolve()}/wat_opc3.xml']}
        self.structure_file = structure_file
        # Parse the structure file to see if the user is in OpenFF or OpenMM mode
        self.working_mode = self._parse_file(structure_file)
        
        #If force field files were provided, check their extensions, if not use the default
        if force_field_files is None:
            self.xmls = default_xmls[self.working_mode]
        elif type(force_field_files) != list:
            raise Exception('force_field_files parameter must be specified as a list of strings')
        else:
            mode_parse = [self._parse_file(ff_file) for ff_file in force_field_files]
            mode_check = [elem == self.working_mode for elem in mode_parse]
            if False in mode_check:
                bad_index = mode_check.index(False)
                bad_file = force_field_files[bad_index]
                raise Exception(f'{bad_file} was found incompatible with the structure file')
            self.xmls = force_field_files
    
    
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

    def main(self):
        """
        The intended main usage case is to parameterize ligands with an SDF file and openff parameters
        and do do a protein, lipid, solvent system with openmm parameters.
        """
        if self.working_mode == 'OpenFF':
            mol = openff.toolkit.Molecule.from_file(self.structure_file, allow_undefined_stereo=True)
            ff = openff.toolkit.ForceField(*self.xmls)
            cubic_box = openff.units.Quantity(30 * np.eye(3), openff.units.unit.angstrom)
            interchange = openff.interchange.Interchange.from_smirnoff(topology=[mol], force_field=ff, box=cubic_box)
            positions = np.array(interchange.positions) * nanometer
            sys = interchange.to_openmm_system()
            top = interchange.to_openmm_topology()
        
        elif self.working_mode == 'OpenMM':
            ff = ForceField(*self.xmls)
            pdb = PDBFile(self.structure_file)
            top, positions = pdb.getTopology(), pdb.getPositions()
            sys = ff.createSystem(top, nonbondedMethod=PME) # Make this an adjustable parameter later

        return (sys, top, positions)