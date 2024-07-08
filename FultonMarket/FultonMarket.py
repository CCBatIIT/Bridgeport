from openmm import *
from openmm.app import *
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState
from openmmtools.multistate import ParallelTemperingSampler, MultiStateReporter
import tempfile
import os, sys
sys.path.append('../MotorRow')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import netCDF4 as nc
from typing import List
from datetime import datetime
import mdtraj as md

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
        integrator = LangevinIntegrator(300, 0.01, 2)
        sim = Simulation(self.pdb.topology, self.system, integrator)
        sim.loadState(input_state)
        self.context = sim.context

    def run(self, total_sim_time: float, iteration_length: float, dt: float=2.0, T_min: float=300, T_max: float=360, n_replicates: int=12, init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35, init_overlap_perc: float=0.2, output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/')):
        """
        Run parallel temporing replica exchange. 

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
                Acceptance rate threshold during "init_overlap_perc" of the simulation time to cause restart. Default is 0.50.

            term_overlap_thresh (float):
                Terminal acceptance rate. If the minimum acceptance rate every falls below this threshold simulation with restart. Default is 0.35.

            init_overlap_perc: (float):
                Percentage of simulation time to evaluate acceptance rates with init_overlap_thresh. For example 0.2 (default) represents first 20% of simulation. 

            output_dir (str):
                String path to output directory to store files. Default is 'FultonMarket_output' in the current working directory.
        """

        # Store variables
        self.total_sim_time = total_sim_time
        self.iter_length = iteration_length
        self.dt = dt 
        self.T_min = T_min
        self.T_max = T_max
        self.n_replicates = n_replicates
        self.init_overlap_thresh = init_overlap_thresh
        self.init_overlap_perc = init_overlap_perc
        self.term_overlap_thresh = term_overlap_thresh
        self.output_dir = output_dir
        self.output_ncdf = os.path.join(self.output_dir, 'output.ncdf')

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found total simulation time of', self.total_sim_time, 'nanoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found iteration length of', self.iter_length, 'nanoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found timestep of', self.dt, 'femtoseconds', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found minimum temperature', self.T_min, 'Kelvin', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found maximum temperature', self.T_max, 'Kelvin', flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found number of replicates', self.n_replicates, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found initial acceptance rate threshold', self.init_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found initial acceptance rate threshold holding percentage', self.init_overlap_perc, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found terminal acceptance rate threshold', self.term_overlap_thresh, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found output_dir', self.output_dir, flush=True)

        # Configure simulation times
        self._configure_simulation_parameters()

        # Set reference state
        self.ref_state = states.ThermodynamicState(system=self.system, temperature=self.temperatures[0], pressure=1.0*unit.bar)

        # Set up simulation
        self._build_simulation()

        # Run Simulations
        self._simulate()        

        # Save data
        self._write_trajectory()
        

    def _configure_simulation_parameters(self):
        """
        Configure simulation times to meet aggregate simulation time. 
        """            
        
        # Configure times/steps
        sim_time_per_rep = self.total_sim_time / self.n_replicates
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated simulation per replicate to be', np.round(sim_time_per_rep, 6), 'nanoseconds', flush=True)
        
        steps_per_rep = np.ceil(sim_time_per_rep * 1e6 / self.dt)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per replicate to be', np.round(steps_per_rep,0), 'steps', flush=True)        
        
        self.n_steps_per_iter = self.iter_length * 1e6 / self.dt
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per iteration to be', np.round(self.n_steps_per_iter, 0), 'steps', flush=True) 
        
        self.n_iters = np.ceil(steps_per_rep / self.n_steps_per_iter)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iterations to be', self.n_iters, 'iterations', flush=True) 
        
        self.n_cycles = np.ceil(self.n_iters / 5)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of cycles to be', self.n_cycles, 'cycles', flush=True) 
        
        self.n_iters_per_cycle = np.ceil(self.n_iters / self.n_cycles)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iters per cycle to be', self.n_iters_per_cycle, 'iterations', flush=True) 

        # Configure replicates
        self.temperatures = [temp*unit.kelvin for temp in np.logspace(np.log10(self.T_min),np.log10(self.T_max), self.n_replicates)]
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of replicates to be', [np.round(t._value,1) for t in self.temperatures], flush=True) 



    def plot_energies(self, figsize=(10,2)):
        # Get information
        ncfile = nc.Dataset(self.output_ncdf)
        temperatures = self.temperatures
        energies = ncfile.variables['energies'][:].data
        states = ncfile.variables['states'][:].data
    
        # Create plotting obj
        fig, ax = plt.subplots(dpi=300, figsize=figsize)
    
        # Plot by state
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(temperatures)))
        for state, temp in enumerate(temperatures):
            state_inds = np.where(states == state)
            state_energies = energies[state_inds[0], state_inds[1], state][2:] # Remove first two iteration
            sns.kdeplot(state_energies, color=cmap[state], ax=ax, linewidth=0.5, label=np.round(temp,2))
            
    
        ax.set_xlabel('Energy (kT)')
        ax.legend(bbox_to_anchor=(1,1), ncol=np.ceil(len(temperatures)/15), fontsize=5)
        plt.show()
        

    
    def _build_simulation(self, interpolate=False):

        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)
        
        # Set up simulation
        self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        self.simulation._global_citation_silence = True

        # Setup reporter
        atom_inds = tuple([i for i in range(self.system.getNumParticles())])
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=10, analysis_particle_indices=atom_inds)
        
        # Load from checkpoint, if available
        if os.path.exists(self.output_ncdf) and interpolate == False:
            self.reporter.open()
            self.simulation = self.simulation.from_storage(self.output_ncdf)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Loading simulation from', self.output_ncdf, flush=True) 
            ncfile = nc.Dataset(self.output_ncdf)
            n_iters_completed = ncfile.dimensions['iteration'].size - 1
            self.current_cycle = int(np.floor(n_iters_completed / self.n_iters_per_cycle))
            self.restart = True
            
        else:                                        
            # Create simulation
            if hasattr(self, 'context'):
                sampler = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors).from_context(self.context)
            else:
                sampler = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors)
            self.simulation.create(self.ref_state,
                                  sampler,
                                  self.reporter, 
                                  temperatures=self.temperatures,
                                  n_temperatures=len(self.temperatures))
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
            if self.current_cycle == 0 and not self.restart:
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
        self.simulation.run(self.n_iters_per_cycle)

        # Eval acceptance rates
        perc_through = self.current_cycle / self.n_cycles
        if perc_through <= self.init_overlap_perc:
            insert_inds = self._eval_acc_rates(self.init_overlap_thresh)
        else:
            insert_inds = self._eval_acc_rates(self.term_overlap_thresh)

        # Interpolate, if necessary
        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.current_cycle = 0
            self._build_simulation(interpolate=True)
            self._configure_simulation_parameters
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
    
        # Add new states
        prev_temps = [s.temperature._value for s in self.reporter.read_thermodynamic_states()[0]]
        new_temps = [temp for temp in prev_temps]
        for displacement, ind in enumerate(insert_inds):
            temp_below = prev_temps[ind-1]
            temp_above = prev_temps[ind]
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state at', np.mean((temp_below, temp_above)), flush=True) 
            new_temps.insert(ind + displacement, np.mean((temp_below, temp_above)))
    
        self.temperatures = [temp*unit.kelvin for temp in new_temps]
        self.n_replicates = len(self.temperatures)


    def _write_trajectory(self):
        """
        Write .dcd from .ncdf file
        """
        self.output_pdb = os.path.join(self.output_dir, 'output.pdb')
        self.output_dcd = os.path.join(self.output_dir, 'output.dcd')

        # Get positions from ncdf
        ncfile = nc.Dataset(self.output_ncdf)
        self.ref_positions = ncfile.variables['positions'][:,0].data
        self.ref_vectors = ncfile.variables['box_vectors'][:,0].data

        # Store w/ mdtraj
        traj = md.load_pdb(self.input_pdb)
        traj.xyz = self.ref_positions.copy()
        traj.unitcell_vectors = self.ref_vectors.copy()
        traj[0].save_pdb(self.output_pdb)
        traj.save_dcd(self.output_dcd)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved output.pdb to ', self.output_pdb, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved output.dcd to ', self.output_dcd, flush=True)



        
            
                
            
