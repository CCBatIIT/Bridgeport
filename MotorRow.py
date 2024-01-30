import os
import mdtraj as md
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *


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
    
    def __init__(self, xml_file, working_directory):
        """
        Parse the xml into openmm an openmm system
        Sets the self.system parameter
        """
        #Load XML file
        self.xml = xml_file
                
        if os.path.isabs(working_directory):
            self.abs_work_dir = working_directory
        else:
            self.abs_work_dir = os.path.join(os.getcwd(), working_directory)

        if not os.path.isdir(self.abs_work_dir):
            os.mkdir(self.abs_work_dir)

        
    def main(self, pdb_in)):
        """
        Run the standard five step equilibration
        0 - Minimization
        1 - NVT with Heavy Restraints on the Protein and Membrane (Z) coords
        2 - NVT with no restraints
        3 - NPT with MonteCarlo Membrane Barostat
        4 - NPT with MonteCarlo Barostat
        5 - NPT with MonteCarlo Barostat
        """
        
        pdb1 = self._minimize(pdb_in)
        pdb2 = self._run_step(pdb1, 1)
        pdb3 = self._run_step(pdb2, 2)
        pdb4 = self._run_step(pdb3, 3)
        pdb5 = self._run_step(pdb4, 4)
        prod_pdb = self._run_step(pdb5, 5)
        
        return prod_pdb
    
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
        pdb = PDBFile(pdb_in)
        with open(xml_file) as f:
            system = XmlSerializer.deserialize(f.read())
        return system, pdb.topology, pdb.positions
        
    
    def _write_structure(self, sim: Simulation, pdb_fn: str):
        """
        Writes the structure of the given simulation object to pdb_fn
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
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
        
    def _minimize(self, pdb_in: str, pdb_out: str = None, temp = 300.0, dt = 2.0):
        """
        Minimizes the structure of pdb_in
        
        Parameters:
            pdb_in - the structure to be minimized
        
        Returns:
            pdb_out - FilePath to the output structure
        """
        system, topology, positions = self._unpack_infiles(self.xml, pdb_in)
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Original state")
        simulation.minimizeEnergy()
        describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Minimized state")
        
        if pdb_out is not None:
            write_structure(simulation, pdb_out)
        else:
            pdb_out = os.path.join(os.path.split(pdb_in)[0], f'minimized.pdb')
            write_structure(simulation, pdb_out)
        
        return pdb_out


    def _run_step(self, pdb_in:str, stepnum:int, pdb_out:str=None,
                  fc_pos:float=300.0, nsteps=125000, temp=300.0, dt=2.0,
                  nstdout=1000, fn_stdout=None, ndcd=5000, fn_dcd=None, press=1.0):
        """
        Run different simulations based on the step number
        1 - NVT with Heavy Restraints on the Protein and Membrane (Z) coords
        2 - NVT with no restraints
        3 - NPT with MonteCarlo Membrane Barostat
        4 - NPT with MonteCarlo Barostat
        5 - NPT with MonteCarlo Barostat
        """
        #Before ANY STEP
        system, topology, positions = self._unpack_infiles(self.xml, pdb_in)
        crds, prt_heavy, mem_heavy = self._get_positions_from_pdb(pdb_in)
        
        #STEP SPECIFIC ACTIONS
        if stepnum == 1:
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
            system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

        elif stepnum == 5:
            system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

        else:
            raise NotImplementedError('How did that happen?')
        
        #AFTER ANY STEP
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        try:
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            simulation = Simulation(topology, system, integrator, platform, properties)
        except:
            simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temp)
        
        if fn_stdout=None:
            fn_stdout = os.path.join(self.abs_work_dir, f'step{stepnum}.stdout')
        
        if fn_dcd=None:
            fn_dcd = os.path.join(self.abs_work_dir, f'step{stepnum}.dcd')
        
        SDR = app.StateDataReporter(fn_stdout, nstdout, step=True, time=True,
                                    potentialEnergy=True, temperature=True, progress=False,
                                    remainingTime=True, speed=False, volume=False,
                                    totalSteps=nsteps, separator=' : '))
        simulation.reporters.append(SDR)
        DCDR = app.DCDReporter(fn_dcd, ndcd)
        simulation.reporters.append(DCD)
        simulation.step(nsteps)

        if pdb_out is not None:
            write_structure(simulation, pdb_out)
        else:
            pdb_out = os.path.join(os.path.split(pdb_in)[0], f'Step_{stepnum}.pdb')
            write_structure(simulation, pdb_out)

        return pdb_out