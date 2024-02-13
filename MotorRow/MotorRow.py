import os, shutil
import mdtraj as md
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from datetime import datetime

class MotorRow():
    """
    A Class for Equilibration of Membrane Proteins
    Follows a five-step protocol
        0 - Minimization
        1 - "nvt": {"fc_pos": 300.0, "totalSteps": 125000}
        2 - "nvt": {"totalSteps": 125000}
        3 - "npt": {"totalSteps": 2500000, "Barostat" : "MonteCarloMembraneBarostat", "Pressure" : 1.0}
        4 - "npt": {"totalSteps": 2500000, "Barostat" : "MonteCarloBarostat", "Pressure" : 1.0}
        5 - "npt": {"totalSteps": 10000000, "Barostat" : "MonteCarloBarostat", "Pressure" : 1.0}

        Common - dt=2.0fs ; Temp=300K ; Platform=OpenCL ; 1000 step stdout ; 5000 step dcd ; 
    """
    
    def __init__(self, system_xml, pdb_file, working_directory):
        """
        Parse the xml into openmm an openmm system
        Sets the self.system parameter
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
        if os.path.isabs(system_xml):
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
        """
        #IF the pdb is absolute, store other files in that same directory (where the pdb is)
        if os.path.isabs(pdb_in):
            pass
        else:
            shutil.copy(pdb_in, os.path.join(self.abs_work_dir, pdb_in))
            pdb_in = os.path.join(self.abs_work_dir, pdb_in)
        
        #Minimize
        state_fn, pdb_fn = self._minimize(pdb_in)
        #NVT Restraints
        state_fn, pdb_fn = self._run_step(state_fn, 1, nsteps=125000, positions_from_pdb=pdb_fn)
        #NVT no Restraints
        state_fn, pdb_fn = self._run_step(state_fn, 2, nsteps=125000)
        #NPT Membrane Barostat
        state_fn, pdb_fn = self._run_step(state_fn, 3, nsteps=250000)
        #NPT
        state_fn, pdb_fn = self._run_step(state_fn, 4, nsteps=250000)
        #NPT
        state_fn, pdb_fn = self._run_step(state_fn, 5, nsteps=250000)
        
        return state_fn, pdb_fn
    

    def _describe_state(self, sim: Simulation, name: str = "State"):
        """
        Report the energy of an openmm simulation
        """
        state = sim.context.getState(getEnergy=True, getForces=True)
        max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
        print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
              f"with maximum force {round(max_force, 2)} kJ/(mol nm)")


    def _unpack_infiles(self, xml, pdb):
        """
        Parse XML and PDB into Openmm System Topology adn Positions
        """
        print(f'Unpacking {xml}, {pdb}')
        pdb = PDBFile(pdb)
        with open(xml) as f:
            system = XmlSerializer.deserialize(f.read())
        return system, pdb.topology, pdb.positions
       

    def _write_state(self, sim: Simulation, xml_fn: str):
        """
        Serialize the openmm State as an xml file
        """
        state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        contents = XmlSerializer.serialize(state)
        with open(xml_fn, 'w') as f:
            f.write(contents)
        print(f'Wrote: {xml_fn}')
 
    
    def _write_system(self, sim: Simulation, xml_fn: str):
        """
        Serialize the openmm system as an xml file
        """
        with open(xml_fn, 'w') as f:
            f.write(XmlSerializer.serialize(sim.system))
        print(f'Wrote: {xml_fn}')


    def _write_structure(self, sim: Simulation, pdb_fn:str=None):
        """
        Writes the structure of the given simulation object to pdb_fn
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        print(f'Wrote: {pdb_fn}')


    def _get_positions_from_pdb(self, fname_pdb):
        nameMembrane = ['DPP', 'POP']
        with open(fname_pdb, 'r') as f_pdb:
            l_pdb = f_pdb.read().split('\n')
            
        coords = []
        prt_heavy_atoms = []
        mem_heavy_atoms = []
        iatom = 0
        
        for line in l_pdb[:-1]:
            if line[:6] in ['ATOM  ', 'HETATM']:
                words = line[30:].split()
                x = float(words[0])
                y = float(words[1])
                z = float(words[2])
    
                coords.append(Vec3(x, y, z))
    
                if line[17:20] in nameMembrane and words[-1] != 'H':
                    mem_heavy_atoms.append(iatom)
                elif line[:6] in ['ATOM  '] and words[-1] != 'H':
                    prt_heavy_atoms.append(iatom)
                
                iatom += 1
    
        return np.array(coords), prt_heavy_atoms, mem_heavy_atoms
        

    def _minimize(self, pdb_in:str, pdb_out:str=None, state_xml_out:str=None, temp=300.0, dt=2.0):
        """
        Minimizes the structure of pdb_in
        
        Parameters:
            pdb_in - the structure to be minimized
        
        Returns:
            pdb_out - FilePath to the output structure
        """
        start = datetime.now()
        system, _, positions = self._unpack_infiles(self.system_xml, pdb_in)
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(self.topology, system, integrator)
        simulation.context.setPositions(positions)
        self._describe_state(simulation, "Original state")
        simulation.minimizeEnergy()
        self._describe_state(simulation, "Minimized state")
        end = datetime.now() - start
        print(f'Minimization completed in {end}')
        
        if pdb_out is not None:
            pass
        else:
            pdb_out = os.path.join(self.abs_work_dir, f'minimized.pdb')
        self._write_structure(simulation, pdb_out)
        
        if state_xml_out is not None:
            pass
        else:
            state_xml_out = os.path.join(self.abs_work_dir, f'minimized_state.xml')
        self._write_state(simulation, state_xml_out)
        return state_xml_out, pdb_out


    def _run_step(self, state_in:str, stepnum:int, state_xml_out:str=None, pdb_out:str=None,
                  fc_pos:float=300.0, nsteps=125000, temp=300.0, dt=2.0, nstdout=1000,
                  fn_stdout=None, ndcd=5000, fn_dcd=None, press=1.0, positions_from_pdb:str=None):
        """
        Run different simulations based on the step number
        1 - NVT with Heavy Restraints on the Protein and Membrane (Z) coords
        2 - NVT with no restraints
        3 - NPT with MonteCarlo Membrane Barostat
        4 - NPT with MonteCarlo Barostat
        5 - NPT with MonteCarlo Barostat
        """
        #Before ANY STEP
        start = datetime.now()
        #Establish State
        with open(self.system_xml) as f:
            system = XmlSerializer.deserialize(f.read())
        
        #print(f'Forces as loaded from XML: {system.getForces()}')
        #print(f'Box Vectors as loaded from system: {system.getDefaultPeriodicBoxVectors()}')
        
        #STEP SPECIFIC ACTIONS
        if stepnum == 1:
            assert positions_from_pdb is not None
            crds, prt_heavy, mem_heavy = self._get_positions_from_pdb(positions_from_pdb)
            prt_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x0,y0,z0)^2')
            prt_rest.addGlobalParameter('fc_pos', fc_pos)
            prt_rest.addPerParticleParameter('x0')
            prt_rest.addPerParticleParameter('y0')
            prt_rest.addPerParticleParameter('z0')
            for iatom in prt_heavy:
                x, y, z = crds[iatom]/10
                prt_rest.addParticle(iatom, [x, y, z])
            system.addForce(prt_rest)
            #Membrane Restraint
            mem_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x,y,z0)^2')
            mem_rest.addGlobalParameter('fc_pos', fc_pos)
            mem_rest.addPerParticleParameter('z0')
            for iatom in mem_heavy:
                x, y, z = crds[iatom]/10
                mem_rest.addParticle(iatom, [z])
            system.addForce(mem_rest)

        elif stepnum == 2:
            pass

        elif stepnum == 3:
            system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
                                                       MonteCarloMembraneBarostat.XYIsotropic,
                                                       MonteCarloMembraneBarostat.ZFree, 100))
        elif stepnum == 4:
            #system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
            #                                           MonteCarloMembraneBarostat.XYIsotropic,
            #                                           MonteCarloMembraneBarostat.ZFree, 100))
            system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

        elif stepnum == 5:
            #system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
            #                                           MonteCarloMembraneBarostat.XYIsotropic,
            #                                           MonteCarloMembraneBarostat.ZFree, 100))
            system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

        else:
            raise NotImplementedError('How did that happen?')
        
        #Any Step Establish Simulation
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        try:
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            simulation = Simulation(self.topology, system, integrator, platform, properties)
        except:
            simulation = Simulation(self.topology, system, integrator)
        
        #If it is an NVT step, positions should be set from the pdb out of previous step
        #otherwise positions (and box vecs) should be set via loadState
        if stepnum == 1:
            assert positions_from_pdb is not None
            pdb = PDBFile(positions_from_pdb)
            simulation.context.setPositions(pdb.positions)
            simulation.context.setVelocitiesToTemperature(temp)
        else:
            simulation.loadState(state_in)
        
        if fn_stdout is None:
            fn_stdout = os.path.join(self.abs_work_dir, f'step{stepnum}.stdout')
        
        if fn_dcd is None:
            fn_dcd = os.path.join(self.abs_work_dir, f'step{stepnum}.dcd')
        
        SDR = app.StateDataReporter(fn_stdout, nstdout, step=True, time=True,
                                    potentialEnergy=True, temperature=True, progress=False,
                                    remainingTime=True, speed=False, volume=True,
                                    totalSteps=nsteps, separator=' : ')
        simulation.reporters.append(SDR)
        DCDR = app.DCDReporter(fn_dcd, ndcd)
        simulation.reporters.append(DCDR)
        print(f'Starting Step {stepnum} with forces {simulation.system.getForces()}')
        print(f'Starting Step {stepnum} with box_vectors {simulation.system.getDefaultPeriodicBoxVectors()}')
        simulation.step(nsteps)
        self._describe_state(simulation, f'Step {stepnum}')
        end = datetime.now() - start
        print(f'Step {stepnum} completed after {end}')
        print(f'Box Vectors after this step {simulation.system.getDefaultPeriodicBoxVectors()}')
        
        if pdb_out is not None:
            pass
        else:
            pdb_out = os.path.join(self.abs_work_dir, f'Step_{stepnum}.pdb')
        self._write_structure(simulation, pdb_out)

        if state_xml_out is not None:
            pass
        else:
            state_xml_out = os.path.join(self.abs_work_dir, f'Step_{stepnum}.xml')
        self._write_state(simulation, state_xml_out)
        
        for i in range(3):
            print('########################################################################################')
        return state_xml_out, pdb_out
