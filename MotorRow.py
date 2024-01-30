


import os

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
        with open(sys.argv[2]) as f:
            self.system = XmlSerializer.deserialize(f.read())
        
        if os.path.isabs(working_directory)
            self.abs_work_dir = working_directory
        else:
            self.abs_work_dir = os.path.join(os.getcwd(), working_directory)

        if not os.path.isdir(self.abs_work_dir):
            os.mkdir(self.abs_work_dir)

        
    def _describe_state(self, sim: Simulation, name: str = "State"):
        """
        Report the energy of an openmm simulation
        """
        state = sim.context.getState(getEnergy=True, getForces=True)
        max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
        print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
              f"with maximum force {round(max_force, 2)} kJ/(mol nm)")


    def _write_structure(self, sim: Simulation, pdb_fn: str):
        """
        Writes the structure of the given simulation object to pdb_fn
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
        print(f'Wrote: {pdb_fn}')
        
    def _step0(self, pdb_in: str, pdb_out: str = None, temp = 300.0, dt = 2.0):
        """
        Minimizes the structure of pdb_in
        
        Parameters:
            pdb_in - the structure to be minimized
        
        Returns:
            pdb_out - FilePath to the output structure
        """
        pdb = PDBFile(pdb_in)
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(pdb.topology, self.system, integrator)
        simulation.context.setPositions(pdb.positions)
        describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Original state")
        simulation.minimizeEnergy()
        describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Minimized state")
        if pdb_out is not None:
            write_structure(simulation, pdb_out)
        else:
            pdb_out = os.path.splitext(pdb_in)[0] + '_min' + os.path.splitext(pdb_in)[-1]
        return pdb_out
    
    def _step1(self, pdb_in: str, pdb_out: str = None, nsteps=125000, 
               temp = 300.0, dt = 2.0, nstdout=1000, ndcd=5000):
        """
        Run 250 ps with strong restraints on the protein and membrane heavy atoms
        """
        pdb = PDBFile(pdb_in)
        #RESTRAINT STUFF
        #END RESTRAINT STUFF
        integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
        simulation = Simulation(pdb.topology, sys, integrator)
        simulation.context.setPositions(pdb.positions)

    
    def _step2(self):
    def _step3(self):
    def _step4(self):
    def _step5(self):

  
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
