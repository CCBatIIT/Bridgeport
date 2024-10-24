#MotorRow
import os, shutil, sys
import mdtraj as md
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from datetime import datetime
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
from Minimizer_utils import *
from typing import List
import math

class Minimizer():
    """
 
    """
    
    def __init__(self, system_xml, pdb_file, working_directory):
        """
        Parse the xml into an openmm system; sets the self.system attribute from the xml file; sets the self.topology attribute from pdb_file

        Parameters:
            system_xml: string: path to the xml file that was generated using Bridgeport
            pdb_file: string: path to the pdb file that was generated using Bridgeport
            working directory: string: the directory to store simulation data in

        Returns:
            None
        """
        #If the working dir is absolute, leave it alone, otherwise make it abs
        if os.path.isabs(working_directory):
            self.abs_work_dir = working_directory
        else:
            self.abs_work_dir = os.path.join(os.getcwd(), working_directory)
        #Ensure that the working dir exists, and if not create it
        if not os.path.isdir(self.abs_work_dir):
            os.mkdir(self.abs_work_dir)
        #Get the system xml file (we want to create a system fresh from this every time)
        if system_xml is None or os.path.isabs(system_xml):
            pass
        else:
            shutil.copy(system_xml, os.path.join(self.abs_work_dir, system_xml))
            system_xml = os.path.join(self.abs_work_dir, system_xml)

        self.system_xml = system_xml
        
        #Get the pdbfile, store the topology (and initial positions i guess)
        if os.path.isabs(pdb_file):
            pass
        else:
            shutil.copy(pdb_file, os.path.join(self.abs_work_dir, pdb_file))
            pdb_file = os.path.join(self.abs_work_dir, pdb_file)
        
        pdb = PDBFile(pdb_file)
        self.topology = pdb.topology

        
    def main(self, pdb_in):
        """
        Run the standard five step equilibration
        0 - Minimization
        1 - NVT with Heavy Restraints on the Protein and Membrane (Z) coords
        2 - NVT with no restraints
        3 - NPT with MonteCarlo Membrane Barostat
        4 - NPT with MonteCarlo Barostat
        5 - NPT with MonteCarlo Barostat

        Parameters:
            pdb_in: string: path to the pdb file (same as init) - the initial structure

        Returns:
            state_fn: string: path to the XML serialized state file
            pdb_fn: string: path to the final structure of the equilibration simulation.
        """
        #IF the pdb is absolute, store other files in that same directory (where the pdb is)
        if os.path.isabs(pdb_in):
            pass
        else:
            shutil.copy(pdb_in, os.path.join(self.abs_work_dir, pdb_in))
            pdb_in = os.path.join(self.abs_work_dir, pdb_in)
        
        #Minimize
        state_fn, pdb_fn = self._minimize(pdb_in)
        
        return state_fn, pdb_fn
    

    def _describe_state(self, sim: Simulation, name: str = "State"):
        """
        Report the energy of an openmm simulation

        Parameters:
            sim: Simulation: The OpenMM Simulation object to report the energy of
            name: string: Default="State" - An optional identifier to help distinguish what energy is being reported
        """
        state = sim.context.getState(getEnergy=True, getForces=True)
        self.PE = round(state.getPotentialEnergy()._value, 2)
        max_force = round(max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces()), 2)
        print(f"{name} has energy {self.PE} kJ/mol ", f"with maximum force {max_force} kJ/(mol nm)")
      
        
    def _write_state(self, sim: Simulation, xml_fn: str):
        """
        Serialize the State of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the State of
            xml_fn: string: The path to the xmlfile to write the serialized State to
        """
        state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        contents = XmlSerializer.serialize(state)
        with open(xml_fn, 'w') as f:
            f.write(contents)
        print(f'Wrote: {xml_fn}')
 
    
    def _write_system(self, sim: Simulation, xml_fn: str):
        """
        Serialize the System of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the System of
            xml_fn: string: The path to the xmlfile to write the serialized System to
        """
        with open(xml_fn, 'w') as f:
            f.write(XmlSerializer.serialize(sim.system))
        print(f'Wrote: {xml_fn}')


    def _write_structure(self, sim: Simulation, pdb_fn: str):
        """
        Write the structure of an OpenMM Simulation to a PDB file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to write the structure of
            pdb_fn: string: The path to the PDB file to write the structure to
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        print(f'Wrote: {pdb_fn}')
        

    def _minimize(self, pdb_in:str, pdb_out:str=None, state_xml_out:str=None, temp=300.0, dt=2.0, lig_resname: str=None, mcs: List[str]=None, fc_pos: float=40.0):
        """
        Minimizes the structure of pdb_in
        
        Parameters:
            pdb_in - the structure to be minimized
        
        Returns:
            pdb_out - FilePath to the output structure
        """
        start = datetime.now()
        system, _, positions = unpack_infiles(self.system_xml, pdb_in)

        # Add restraint if specified 
        if mcs != None and lig_resname != None:
            crds, prt_heavy_atoms, mem_heavy_atoms, lig_heavy_atoms = get_positions_from_pdb(pdb_in, lig_resname=lig_resname)
            lig_heavy_atom_inds = np.array(lig_heavy_atoms)[:,0].astype(int)
            lig_heavy_atom_names = np.array(lig_heavy_atoms)[:,1]
            mcs_atom_inds = parse_atom_inds(lig_heavy_atom_inds, lig_heavy_atom_names, mcs)

            system = restrain_atoms(system, crds, prt_heavy_atoms, fc_pos)
            system = restrain_atoms(system, crds, mem_heavy_atoms, fc_pos)
            system = restrain_atoms(system, crds, mcs_atom_inds, fc_pos) 

        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(self.topology, system, integrator)
        simulation.context.setPositions(positions)
        self._describe_state(simulation, "Original state")
        simulation.minimizeEnergy()
        self._describe_state(simulation, "Minimized state")
        end = datetime.now() - start
        print(f'Minimization completed in {end}')
        
        if pdb_out is None:
            pdb_out = os.path.join(self.abs_work_dir, f'minimized.pdb')
        self._write_structure(simulation, pdb_out)

        if state_xml_out is None:
            state_xml_out = os.path.join(self.abs_work_dir, f'minimized.xml')
        self._write_state(simulation, state_xml_out)
            
        return state_xml_out, pdb_out

