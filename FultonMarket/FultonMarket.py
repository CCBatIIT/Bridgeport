from openmm import *
from openmm.app import *
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.multistate import ParallelTemperingSampler, ReplicaExchangeSampler, MultiStateReporter
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

import faulthandler
faulthandler.enable()


geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
spring_constant_unit = (unit.joule)/(unit.angstrom*unit.angstrom*unit.mole)


class FultonMarket():
    """
    Replica exchange

    Default is Parallel Tempering (PT), additionally can perform PT with restraints (PTwRE)
    PT uses a ParallelTemperingSampler
    PTwRE uses the more general ReplicaExchangeSampler

    Methods:
        init - Initialize a Fulton Market Object with pdb, xml (system), and xml (state) files
        run - Run parallel tempering replica exchange.
        _save_simulation - Save the important information from a simulation and then truncate the output.ncdf file to preserve disk space.
        _configure_simulation_parameters - Configure simulation times to meet aggregate simulation time.
        plot_energies - plot the energy of each state in the simulation
        _build_simulation - assign Integrator and Report, create a new simulation or continue a previous one as necessary
        _simulate - perform the simulation, running cycles
        _run_cycle - run one cycle, performing an interpolation of parameters should acceptance rates be too low
        _eval_acc_rates - Evaluate acceptance rates
        _interpolate_states - add new states between two which have poor acceptance rates (below the threshold)
        _restrain_atoms_by_dsl - In the case of a restrained simulation - puts restraints on selected atoms in each thermodynamic state of the PTwRE simulation
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
        self.system = XmlSerializer.deserialize(open(input_system, 'r').read())
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input_system:', input_system, flush=True)


        # Build state
        if input_state != None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context

    def run(self, total_sim_time: float, iteration_length: float,
            dt: float=2.0, T_min: float=300, T_max: float=360, n_replicates: int=12,
            init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'),
            restrained_atoms_dsl=None, K_max:unit.Quantity=unit.Quantity(83.68, spring_constant_unit)):
        """
        PT - Default - Run parallel temporing replica exchange.
        PTwRE - By assigning a string to the restrained_atoms_dsl argument, those atoms will be restrained (default is not to do this)

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
                Terminal acceptance rate. If the minimum acceptance rate ever falls below this threshold simulation will restart. Default is 0.35.

            output_dir (str):
                String path to output directory to store files. Default is 'FultonMarket_output' in the current working directory.

            restrained_atoms_dsl:
                If restraints are to be used, supply an MDTraj selection string for the atoms which are to be restrained (default is not to do this)
            
            K_max (unit.Quantity):
                If restrained_atoms_dsl is not None, then establish restraints (geometrically distributed from 0 to K_max)
                Highest temp is unrestrained, lowest temp is fully restrained (K_max)
        """

        # Store variables and assign units
        self.total_sim_time = total_sim_time * unit.nanosecond
        self.iter_length = iteration_length * unit.nanosecond
        self.dt = dt * unit.femtosecond
        self.T_min = T_min * unit.kelvin
        self.T_max = T_max *unit.kelvin
        self.n_replicates = n_replicates
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh
        self.output_dir = output_dir
        self.output_ncdf = os.path.join(self.output_dir, 'output.ncdf')
        self.checkpoint_ncdf = os.path.join(self.output_dir, 'output_checkpoint.ncdf')
        self.restrained_atoms_dsl = restrained_atoms_dsl
        if self.restrained_atoms_dsl is not None:
            self.K_max = K_max
                        
        self.save_dir = os.path.join(self.output_dir, 'saved_variables')
        self.interpolate = False
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found total simulation time of', self.total_sim_time, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found iteration length of', self.iter_length, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found timestep of', self.dt, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found minimum temperature', self.T_min, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found maximum temperature', self.T_max, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found number of replicates', self.n_replicates, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found initial acceptance rate threshold', self.init_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found terminal acceptance rate threshold', self.term_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found output_dir', self.output_dir, flush=True)
        
        # Loop through short 50 ns simulations to allow for .ncdf truncation
        self._configure_experiment_parameters()
        while self.n_sims_remaining > 0:

            # Configure simulation times
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Configuring Simulation Parameters', flush=True)
            self._configure_simulation_parameters()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done', flush=True)
            
            # Set reference state
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Assigning Reference State', flush=True)
            self.ref_state = states.ThermodynamicState(system=self.system, temperature=self.temperatures[0], pressure=1.0*unit.bar)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done', flush=True)

            # Set up simulation
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Building Simulation', flush=True)
            self._build_simulation()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done', flush=True)
    
            # Run simulation
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Running Simulation', flush=True)
            self._simulate() 
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done', flush=True)

            # Save simulation
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saving SimulationDone', flush=True)
            self._save_simulation()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done', flush=True)

            # Update counter
            self.n_sims_remaining -= 1
    

    def _save_simulation(self):
        """
        Save the important information from a simulation and then truncate the output.ncdf file to preserve disk space.
        """
        # Determine save no. 
        prev_saves = [int(dir) for dir in os.listdir(self.save_dir)]
        if len(prev_saves) > 0:
            new_save_no = max(prev_saves) + 1
        else:
            new_save_no = 0
        save_no_dir = os.path.join(self.save_dir, str(new_save_no))
        if not os.path.exists(save_no_dir):
            os.mkdir(save_no_dir)

        # Truncate output.ncdf
        ncdf_copy = os.path.join(self.output_dir, 'output_copy.ncdf')
        pos, box_vectors, states, energies, temperatures = truncate_ncdf(self.output_ncdf, ncdf_copy, self.reporter, False)
        np.save(os.path.join(save_no_dir, 'positions.npy'), pos.data)
        del pos
        np.save(os.path.join(save_no_dir, 'box_vectors.npy'), box_vectors.data)
        del box_vectors
        np.save(os.path.join(save_no_dir, 'states.npy'), states.data)
        del states
        np.save(os.path.join(save_no_dir, 'energies.npy'), energies.data)
        del energies
        np.save(os.path.join(save_no_dir, 'temperatures.npy'), temperatures)
        del temperatures
        
        # Truncate output_checkpoint.ncdf
        checkpoint_copy = os.path.join(self.output_dir, 'output_checkpoint_copy.ncdf')
        truncate_ncdf(self.checkpoint_ncdf, checkpoint_copy, self.reporter, True)

        # Write over previous .ncdf files
        os.system(f'mv {ncdf_copy} {self.output_ncdf}')
        os.system(f'mv {checkpoint_copy} {self.checkpoint_ncdf}')

        # Close reporter object
        if hasattr(self, 'reporter'):
            if self.reporter.is_open():
                self.reporter.close()

    def _configure_experiment_parameters(self):
        # Assert that no empty save directories have been made
        assert all([len(os.listdir(os.path.join(self.save_dir, dir))) == 5 for dir in os.listdir(self.save_dir)]), "You may have an empty save directory, please remove empty or incomplete save directories before continuing :)"
        
        # Configure experiment parameters
        self.n_sims_completed = len(os.listdir(self.save_dir))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found n_sims_completed to be', self.n_sims_completed, flush=True)
        self.sim_time = 50 * unit.nanosecond
        self.n_sims_remaining = np.ceil(self.total_sim_time / self.sim_time) - self.n_sims_completed
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated n_sims_remaining to be', self.n_sims_remaining, flush=True)

    def _configure_simulation_parameters(self):
        """
        Configure simulation times to meet aggregate simulation time. 
        """            

        # Read number replicates if different than argument
        if os.path.exists(self.output_ncdf):
            self.n_replicates = nc.Dataset(self.output_ncdf).variables['states'].shape[1] 
        
        # Configure times/steps
        sim_time_per_rep = self.sim_time / self.n_replicates
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated simulation per replicate to be', sim_time_per_rep, flush=True)
        
        steps_per_rep = int(np.ceil(sim_time_per_rep / self.dt))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per replicate to be', steps_per_rep, 'steps', flush=True)        
        
        self.n_steps_per_iter = int(np.ceil(self.iter_length / self.dt))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per iteration to be', self.n_steps_per_iter, 'steps', flush=True) 
        
        self.n_iters = int(np.ceil(steps_per_rep / self.n_steps_per_iter))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iterations to be', self.n_iters, 'iterations', flush=True) 
        
        self.n_cycles = int(np.ceil(self.n_iters / 5))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of cycles to be', self.n_cycles, 'cycles', flush=True) 
        
        self.n_iters_per_cycle = int(np.ceil(self.n_iters / self.n_cycles))
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iters per cycle to be', self.n_iters_per_cycle, 'iterations', flush=True) 

        # Configure replicates
        self.temperatures = geometric_distribution(self.T_min, self.T_max, self.n_replicates)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of replicates to be', [np.round(t._value,1) for t in self.temperatures], 'Kelvin', flush=True) 

        # Configure Restraints if necessary
        if self.restrained_atoms_dsl is not None:
            self.spring_constants = list(reversed(geometric_distribution(unit.Quantity(0, spring_constant_unit), self.K_max, self.n_replicates)))
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated spring constants of replicates to be', [np.round(t._value,1) for t in self.spring_constants], spring_constant_unit, flush=True)
        
    def _build_simulation(self, interpolate=False):
        """
        construct the simulation by assigning mcmc move, RE sampler,
        """ 
        
        print('STARTING BUILD SIMULATION')
        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)

        # Set up simulation
        if hasattr(self, 'simulation'):
            del self.simulation

        # Setup reporter
        atom_inds = tuple([i for i in range(self.system.getNumParticles())])
        if hasattr(self, 'reporter'):
            del self.reporter
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=10, analysis_particle_indices=atom_inds)
        
        # Load from checkpoint, if available
        if os.path.exists(self.output_ncdf) and self.interpolate == False:
            self.reporter.open()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Loaded reporter', flush=True) 
            if self.restrained_atoms_dsl is None:
                self.simulation = ParallelTemperingSampler.from_storage(self.reporter)
            else:
                self.simulation = ReplicaExchangeSampler.from_storage(self.reporter)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Loaded simulation from', self.output_ncdf, flush=True) 
            ncfile = nc.Dataset(self.output_ncdf)
            n_iters_completed = ncfile.dimensions['iteration'].size - 1
            ncfile.close()
            self.current_cycle = int(np.floor(n_iters_completed / self.n_iters_per_cycle))
            self.restart = True
            
        else:                                        
            # Create simulation
            if os.path.exists(self.output_ncdf):
                os.remove(self.output_ncdf)
            
            if self.interpolate == False and hasattr(self, 'context'):
                self.sampler_states = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors).from_context(self.context)
            elif self.interpolate == False:
                self.sampler_states = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors)
            
            if self.restrained_atoms_dsl is None:
                self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
                self.simulation.create(self.ref_state, self.sampler_states, self.reporter, temperatures=self.temperatures, n_temperatures=len(self.temperatures))
            else:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Creating Thermodynamic States', flush=True)
                thermodynamic_states = [ThermodynamicState(system=self.system, temperature=T) for T in self.temperatures]
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done Creating Thermodynamic States', flush=True)
                topo = md.Topology.from_openmm(self.pdb.topology)
                for thermo_state, spring_cons in zip(thermodynamic_states, self.spring_constants):
                    self._J_restrain_atoms_by_dsl(thermo_state, self.sampler_states, topo, self.restrained_atoms_dsl, spring_cons)
                    if (1 + thermodynamic_states.index(thermo_state)) % (self.n_replicates // 4) == 0:
                        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Assigning Restraints is {round(100*(1 + thermodynamic_states.index(thermo_state))/self.n_replicates, 2)}% Complete', flush=True)
                self.simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
                self.simulation.create(thermodynamic_states=thermodynamic_states, sampler_states=self.sampler_states, storage=self.reporter)
            self.restart = False


    def _simulate(self):
        """
        Perform entire simulation
        """

        # Continue until self.n_cycles reached
        if not self.restart:
            self.current_cycle = 0
        while self.current_cycle <= self.n_cycles:

            # Minimize
            if self.current_cycle == 0 and not self.restart and not self.interpolate:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing...', flush=True)
                self.simulation.minimize()
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing finished.', flush=True)

            # Advance 1 cycle
            self._run_cycle()
            


    def _run_cycle(self):
        """
        Run one cycle
        """

        # Take steps
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'CYCLE', self.current_cycle, 'advancing', self.n_iters_per_cycle, 'iterations', flush=True) 
        if self.simulation.is_completed:
            self.simulation.extend(self.n_iters_per_cycle)
        else:
            self.simulation.run(self.n_iters_per_cycle)

        # Eval acceptance rates
        if self.n_sims_completed == 0:
            insert_inds = self._eval_acc_rates(self.init_overlap_thresh)
        else:
            insert_inds = self._eval_acc_rates(self.term_overlap_thresh)

        # Interpolate, if necessary
        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self.interpolate = True
            self._build_simulation()
            self._configure_simulation_parameters()
        else:
            self.current_cycle += 1
    


    def _eval_acc_rates(self, acceptance_rate_thresh: float=0.40):
        "Evaluate acceptance rates"        
        
        # Get temperatures
        temperatures = [s.temperature._value for s in self.reporter.read_thermodynamic_states()[0]]
        
        # Get mixing statistics
        accepted, proposed = self.reporter.read_mixing_statistics()
        accepted = accepted.data
        proposed = proposed.data
        acc_rates = np.mean(accepted[1:] / proposed[1:], axis=0)
        acc_rates = np.nan_to_num(acc_rates) # Adjust for cases with 0 proposed swaps
    
        # Iterate through mixing statistics to flag acceptance rates that are too low
        insert_inds = [] # List of indices to apply new state. Ex: (a "1" means a new state between "0" and the previous "1" indiced state)
        for state in range(len(acc_rates)-1):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Mixing between', np.round(temperatures[state], 2), 'and', np.round(temperatures[state+1], 2), ':', acc_rates[state, state+1], flush=True) 
            rate = acc_rates[state, state+1]
            if rate < acceptance_rate_thresh:
                insert_inds.append(state+1)
    
        return np.array(insert_inds)

    def _interpolate_states(self, insert_inds: np.array):
        """
        Add new temperatures (and spring constants) to a new state - if acceptance rates are too low between two states
        """
        # Add new states
        prev_temps = [s.temperature._value for s in self.reporter.read_thermodynamic_states()[0]]
        new_temps = [temp for temp in prev_temps]
        for displacement, ind in enumerate(insert_inds):
            temp_below = prev_temps[ind-1]
            temp_above = prev_temps[ind]
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state at', np.mean((temp_below, temp_above)), flush=True) 
            new_temps.insert(ind + displacement, np.mean((temp_below, temp_above)))

        # Add new restraints
        if self.restrained_atoms_dsl is not None:
            prev_spring_cons = [s._value for s in self.spring_constants]
            new_spring_cons = [cons for cons in prev_spring_cons]
            for displacement, ind in enumerate(insert_inds):
                cons_below = prev_spring_cons[ind-1]
                cons_above = prev_spring_cons[ind]
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state with Spring Constant', np.mean((cons_below, cons_above)), flush=True) 
                new_spring_cons.insert(ind + displacement, np.mean((cons_below, cons_above)))
            self.spring_constants = [cons * spring_constant_unit for cons in new_spring_cons]
            
        self.temperatures = [temp*unit.kelvin for temp in new_temps]
        self.n_replicates = len(self.temperatures)

        # Read sampler states
        sampler_states = self.reporter.read_sampler_states(self.reporter.read_last_iteration())

        # Add sampler_states for new temperatures
        self.sampler_states = []
        displacement = 0
        for state in range(len(self.temperatures)):
            if state-displacement in insert_inds:
                self.sampler_states.append(sampler_states[state-displacement-1])
                displacement += 1
            else:
                self.sampler_states.append(sampler_states[state-displacement])
            

        # # Read initial positions from this simulation
        # ncfile = nc.Dataset(self.output_ncdf)
        # init_positions = ncfile.variables['positions'][0].data
        # init_box_vectors = ncfile.variables['box_vectors'][0].data
        # state_inds = ncfile.variables['states'][0].data
        # ncfile.close()

        # # Reshape positions, box_vectors by state rather than by replicate
        # reshaped_init_positions = np.empty((init_positions.shape))
        # reshaped_init_box_vectors = np.empty((init_box_vectors.shape))
        # for state in range(reshaped_init_positions.shape[0]):
        #     reshaped_init_positions[state] = init_positions[np.where(state_inds == state)[0]]
        #     reshaped_init_box_vectors[state] = init_box_vectors[np.where(state_inds == state)[0]]

        # # Reshape positions, box_vectors into new shape to accomodate interpolation
        # self.init_positions = np.empty((len(self.temperatures), reshaped_init_positions.shape[1], reshaped_init_positions.shape[2]))
        # self.init_box_vectors = np.empty((len(self.temperatures), 3, 3))
        # displacement = 0
        # for state in range(len(self.temperatures)):
        #     if state-displacement in insert_inds:
        #         self.init_positions[state] = reshaped_init_positions[state-displacement-1].copy()
        #         self.init_box_vectors[state] = reshaped_init_box_vectors[state-displacement-1].copy()
        #         displacement += 1
        #     else:
        #         self.init_positions[state] = reshaped_init_positions[state-displacement].copy()
        #         self.init_box_vectors[state] = reshaped_init_box_vectors[state-displacement].copy()
                

    def _J_restrain_atoms_by_dsl(self, thermodynamic_state, sampler_state, topology, atoms_dsl, spring_constant):
        """
        Unceremoniously Ripped from the OpenMMTools github, simply to change sigma to K
        Apply a soft harmonic restraint to the given atoms.
        This modifies the ``ThermodynamicState`` object.
        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state with the system. This will be modified.
        sampler_state : openmmtools.states.SamplerState
            The sampler state with the positions.
        topology : mdtraj.Topology or openmm.Topology
            The topology of the system.
        atoms_dsl : str
           The MDTraj DSL string for selecting the atoms to restrain.
        spring_constant : openmm.unit.Quantity, optional
            Controls the strength of the restrain. The smaller, the tighter
            (units of distance, default is 3.0*angstrom).
        """

        # Make sure the topology is an MDTraj topology.
        if isinstance(topology, md.Topology):
            mdtraj_topology = topology
        else:
            mdtraj_topology = md.Topology.from_openmm(topology)
        
        #Determine indices of the atoms to restrain
        restrained_atom_indices = mdtraj_topology.select(atoms_dsl)
        if len(restrained_atom_indices) == 0:
            raise Exception('No Atoms To Restrain!')
        
        #Assign Spring Constant, ensuring it is the appropriate unit
        K = spring_constant  # Spring constant.
        if type(K) != unit.Quantity:
            K = K * spring_constant_unit
        elif K.unit != spring_constant_unit:
            raise Exception('Improper Spring Constant Unit')
        
        #Energy and Force for Restraint
        energy_expression = '(K/2)*periodicdistance(x, y, z, x0, y0, z0)^2'
        restraint_force = openmm.CustomExternalForce(energy_expression)
        restraint_force.addGlobalParameter('K', K)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')
        for index in restrained_atom_indices:
            parameters = self.init_positions[index,:]
            restraint_force.addParticle(index, parameters)
        thermodynamic_state.system.addForce(restraint_force)
        

