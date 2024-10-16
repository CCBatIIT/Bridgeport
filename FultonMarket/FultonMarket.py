from openmm import *
from openmm.app import *
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState
from openmmtools.multistate import ParallelTemperingSampler, MultiStateReporter
from openmmtools.utils.utils import TrackedQuantity
import tempfile
import os, sys
sys.path.append('../MotorRow')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import netCDF4 as nc
from typing import List
from datetime import datetime
import mdtraj as md
from shorten_replica_exchange import truncate_ncdf
from Randolph import Randolph
import faulthandler
faulthandler.enable()

geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
spring_constant_unit = (unit.joule)/(unit.angstrom*unit.angstrom*unit.mole)
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))

class FultonMarket():
    """
    Replica exchange
    """

    def __init__(self, input_pdb: str, input_system: str, input_state: str=None):
        """
        Initialize a Fulton Market obj. 

        Parameters:
        -----------
            input_pdb (str):
                String path to pdb to run simulation. 

            input_system (str):
                String path to OpenMM system (.xml extension) file that contains parameters for simulation. 

            input_state (str):
                String path to OpenMM state (.xml extension) file that contains state for reference. 


        Returns:
        --------
            FultonMarket obj.
        """
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Welcome to FultonMarket.', flush=True)
        
        # Unpack .pdb
        self.input_pdb = input_pdb
        self.pdb = PDBFile(input_pdb)
        self.init_positions = self.pdb.getPositions(asNumpy=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input_pdb:', input_pdb, flush=True)


        # Unpack .xml
        self.system =XmlSerializer.deserialize(open(input_system, 'r').read())
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input_system:', input_system, flush=True)


        # Build state
        if input_state != None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input_state:', input_state, flush=True)

    def run(self, 
            total_sim_time: float, 
            iteration_length: float, 
            dt: float=2.0,
            T_min: float=300, 
            T_max: float=367.447, 
            n_replicates: int=12, 
            sim_length=50,
            init_overlap_thresh: float=0.5, 
            term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'),
            restrained_atoms_dsl:str=None,
            spring_centers2_pdb:str=None,
            K=83.68):
        """
        Run parallel tempering replica exchange.
        if restrained_atoms_dsl is provided, then parallel tempering will also be restrained to the initial positions with spring constants K
        if spring_centers2_pdb is also provided, then the spring centers will shift between the initial positions and spring_centers2 at high temperature

        Parameters:
        -----------
            total_sim_time (float):
                Aggregate simulation time from all replicates in nanoseconds.

            iteration_length (float):
                Specify the amount of time between swapping replicates in nanoseconds. 

            dt (float):
                Timestep for simulation. Default is 2.0 femtoseconds.

            T_min (float):
                Minimum temperature in Kelvin. This state will serve as the reference state. Default is 300 K.

            T_max (float):
                Maximum temperature in Kelvin. Default is 360 K.

            n_replicates (int):
                Number of replicates, meaning number of states between T_min and T_max. States are automatically built at with a geometeric distribution towards T_min. Default is 12.

            init_overlap_thresh (float):
                Acceptance rate threshold during first 50 ns simulation to cause restart. Default is 0.50. 

            term_overlap_thresh (float):
                Terminal acceptance rate. If the minimum acceptance rate every falls below this threshold simulation with restart. Default is 0.35.

            output_dir (str):
                String path to output directory to store files. Default is 'FultonMarket_output' in the current working directory.

            restrained_atoms_dsl (str):
                MDTraj selection string, selected atoms will be restrained

            K_max (Float in units (joule)/(angstrom*angstrom*mole)):
                Strongest restraint (low temperature state), restraints weaken as temperature rises
        """

        # Store variables
        self.total_sim_time = total_sim_time
        self.restrained_atoms_dsl = restrained_atoms_dsl
        self.temperatures = [temp*unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
        
        if restrained_atoms_dsl is not None: #Leave the top 20% of states unrestrained
            self.spring_constants = [K * spring_constant_unit for i in range(n_replicates)]
            assert len(self.spring_constants) == len(self.temperatures)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Running with restraints', flush=True)
        else:
            self.spring_constants = None
        
        if spring_centers2_pdb is not None:
            # Unpack .pdb
            self.spring_centers = self.make_interpolated_positions_array(self.input_pdb, spring_centers2_pdb, n_replicates) #All atoms, not just protein
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Second Spring Centers and Made the Shifting Center Schedule', flush=True)
            
            # Additionally, in this Umbrella Sampling mode - the temps should all be the max
            self.temperatures = [T_max * unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
            
        elif self.spring_constants is not None:
            self.spring_centers = np.array([self.init_positions for i in range(n_replicates)])
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Restraining All States to the Initial Positions', flush=True)
        else:
            self.spring_centers = None

        ref_state = states.ThermodynamicState(system=self.system, temperature=self.temperatures[0], pressure=1.0*unit.bar)
        self.output_ncdf = os.path.join(output_dir, 'output.ncdf')
        checkpoint_ncdf = os.path.join(output_dir, 'output_checkpoint.ncdf')
        self.save_dir = os.path.join(output_dir, 'saved_variables')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found total simulation time of', total_sim_time, 'nanoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found iteration length of', iteration_length, 'nanoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found timestep of', dt, 'femtoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found number of replicates', n_replicates, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found initial acceptance rate threshold', init_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found terminal acceptance rate threshold', term_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found output_dir', output_dir, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Temperature Schedule', [np.round(T._value, 1) for T in self.temperatures], 'Kelvin', flush=True)
        if self.restrained_atoms_dsl is not None:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Restraint Schedule', [np.round(T._value, 1) for T in self.spring_constants], spring_constant_unit, flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Spring Center Schedule', np.round(rmsd(self.spring_centers[:], self.spring_centers[0]), 2), 'nm', flush=True)
            

        # Loop through short 50 ns simulations to allow for .ncdf truncation
        self._configure_experiment_parameters(sim_length=sim_length)
        while self.sim_no < self.total_n_sims:
            
            params = dict(sim_no=self.sim_no,
                          sim_time=self.sim_time,
                          system=self.system,
                          ref_state=ref_state,
                          temperatures=self.temperatures,
                          spring_constants=self.spring_constants,
                          init_positions=self.init_positions,
                          init_box_vectors=self.init_box_vectors,
                          output_dir=output_dir,
                          output_ncdf=self.output_ncdf,
                          checkpoint_ncdf=checkpoint_ncdf,
                          iter_length=iteration_length,
                          dt=dt)
             
            # Initialize Randolph
            if self.sim_no > 0:
                self._load_initial_args() #sets positions, velocities, box_vecs, temperatures, and spring_constants
                params['init_velocities'] = self.init_velocities
                params['init_positions'] = self.init_positions
                params['init_box_vectors'] = self.init_box_vectors
                params['temperatures'] = self.temperatures
                if restrained_atoms_dsl is not None:
                    params['spring_constants'] = self.spring_constants

            elif hasattr(self, 'context'):
                params['context'] = self.context

            if self.restrained_atoms_dsl is not None:
                params['restrained_atoms_dsl'] = self.restrained_atoms_dsl
                params['mdtraj_topology'] = md.Topology.from_openmm(self.pdb.topology)
                params['spring_centers'] = self.spring_centers
            
            simulation = Randolph(**params)
            
            # Run simulation
            simulation.main(init_overlap_thresh=init_overlap_thresh, term_overlap_thresh=term_overlap_thresh)

            # Save simulation
            if self.spring_centers is None and restrained_atoms_dsl is None:
                self.temperatures = simulation.save_simulation(self.save_dir)
            else:
                self.temperatures, self.spring_constants, self.spring_centers = simulation.save_simulation(self.save_dir)
            
            # Delete output.ncdf files if not last simulation 
            if not self.sim_no+1 == self.total_n_sims:
                os.remove(self.output_ncdf)
                os.remove(checkpoint_ncdf)

            # Update counter
            self.sim_no += 1
    

    
    def _load_initial_args(self):
        # Get last directory
        load_no = self.sim_no - 1
        load_dir = os.path.join(self.save_dir, str(load_no))
        
        # Load args (not in correct shapes
        self.temperatures = np.load(os.path.join(load_dir, 'temperatures.npy'))
        self.temperatures = [t*unit.kelvin for t in self.temperatures]
        if self.spring_constants is not None:
            self.spring_constants = np.load(os.path.join(load_dir, 'spring_constants.npy'))
            self.spring_constants = [s*spring_constant_unit for s in self.spring_constants]
            self.spring_centers = np.load(os.path.join(load_dir, 'spring_centers.npy'))
        
        try:
            init_positions = np.load(os.path.join(load_dir, 'positions.npy'))[-1] 
            init_box_vectors = np.load(os.path.join(load_dir, 'box_vectors.npy'))[-1] 
            init_velocities = np.load(os.path.join(load_dir, 'velocities.npy')) 
            state_inds = np.load(os.path.join(load_dir, 'states.npy'))[-1]
        except:
            try:
                init_positions = np.load(os.path.join(load_dir, 'positions.npy'))[-1]
                init_box_vectors = np.load(os.path.join(load_dir, 'box_vectors.npy'))[-1]
                init_velocities = None
                state_inds = np.load(os.path.join(load_dir, 'states.npy'))[-1]
            except:
                init_velocities, init_positions, init_box_vectors, state_inds = self._recover_arguments()
        
        # Reshape 
        reshaped_init_positions = np.empty((init_positions.shape))
        reshaped_init_box_vectors = np.empty((init_box_vectors.shape))
        for state in range(len(self.temperatures)):
            rep_ind = np.where(state_inds == state)[0]
            reshaped_init_box_vectors[state] = init_box_vectors[rep_ind] 
            reshaped_init_positions[state] = init_positions[rep_ind] 
        if init_velocities is not None:
            reshaped_init_velocities = np.empty((init_velocities.shape))
            for state in range(len(self.temperatures)):
                rep_ind = np.where(state_inds == state)[0]
                reshaped_init_velocities[state] = init_velocities[rep_ind] 
        # Convert to quantities    
        self.init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=reshaped_init_positions, mask=False, fill_value=1e+20), unit=unit.nanometer))
        self.init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=reshaped_init_box_vectors, mask=False, fill_value=1e+20), unit=unit.nanometer))
        if init_velocities is not None:
            self.init_velocities = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=reshaped_init_velocities, mask=False, fill_value=1e+20), unit=(unit.nanometer / unit.picosecond)))
        else:
            self.init_velocities = None
            self.context = None

    def _configure_experiment_parameters(self, sim_length=50):
        # Assert that no empty save directories have been made
        assert all([len(os.listdir(os.path.join(self.save_dir, dir))) >= 5 for dir in os.listdir(self.save_dir)]), "You may have an empty save directory, please remove empty or incomplete save directories before continuing :)"
        
        # Configure experiment parameters
        self.sim_no = len(os.listdir(self.save_dir))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found n_sims_completed to be', self.sim_no, flush=True)
        self.sim_time = sim_length # ns
        self.total_n_sims = np.ceil(self.total_sim_time / self.sim_time)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated total_n_sims to be', self.total_n_sims, flush=True)
    
    def _recover_arguments(self):
        ncfile = nc.Dataset(self.output_ncdf, 'r')
        
        # Read
        velocities = ncfile.variables['velocities'][-1].data
        positions = ncfile.variables['positions'][-1].data
        box_vectors = ncfile.variables['box_vectors'][-1].data
        state_inds = ncfile.variables['states'][-1].data
        
        ncfile.close()
        
        return velocities, positions, box_vectors, state_inds

    def make_interpolated_positions_array(self, spring_centers1_pdb, spring_centers2_pdb, num_replicates):
        """
        Create a positions array linearly interpolating from spring_centers1_pdb to spring_centers2_pdb
        """
        #Get the important coordinates from two pdbs, aligning them
        traj1, traj2 = md.load(spring_centers1_pdb), md.load(spring_centers2_pdb)
        prot_inds1, prot_inds2 = traj1.top.select('protein'), traj2.top.select('protein')
        assert np.alltrue(prot_inds1 == prot_inds2)
        not_prot_inds1 = traj1.top.select('not protein')
        traj2 = traj2.superpose(traj1, atom_indices=prot_inds1)
        xyz1, xyz2 = traj1.xyz[0], traj2.xyz[0]
        #Create the array
        positions_array = np.empty((num_replicates, xyz1.shape[0], 3))
        lambdas = np.linspace(1,0,num_replicates)
        gammas = 1 - lambdas
        for i in range(num_replicates):
            positions_array[i, prot_inds1] = lambdas[i]*xyz1[prot_inds1] + gammas[i]*xyz2[prot_inds2]
            positions_array[i, not_prot_inds1] = xyz1[not_prot_inds1]
        
        return positions_array

