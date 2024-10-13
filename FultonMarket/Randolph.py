from openmm import *
from openmm.app import *
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
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
from copy import deepcopy
from shorten_replica_exchange import truncate_ncdf

class Randolph():
    """
    """
    
    def __init__(self, 
                 sim_no: int,
                 sim_time: int,
                 system: openmm.System, 
                 ref_state: ThermodynamicState, 
                 temperatures: np.array,
                 init_positions: np.array, 
                 init_box_vectors: np.array, 
                 output_dir: str,
                 output_ncdf: str, 
                 checkpoint_ncdf: str,
                 iter_length: int,
                 dt: float, 
                 init_velocities=None,
                 context=None):
        """
        """
        # Assign attributes
        self.sim_no = sim_no
        self.sim_time = sim_time
        self.system = system
        self.output_dir = output_dir
        self.output_ncdf = output_ncdf
        self.checkpoint_ncdf = checkpoint_ncdf
        self.temperatures = temperatures.copy()
        self.ref_state = ref_state
        self.n_replicates = len(self.temperatures)
        self.init_positions = init_positions
        self.init_box_vectors = init_box_vectors
        self.init_velocities = init_velocities
        self.iter_length = iter_length
        self.dt = dt
        self.context = context

        
        # Configure simulation parameters
        self._configure_simulation_parameters()
        
        # Build simulation
        self._build_simulation()
        

    
    def main(self, init_overlap_thresh: float, term_overlap_thresh: float):
        """
        Was previously _simulation()
        """
        # Assign attributes
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        # Continue until self.n_cycles reached
        self.current_cycle = 0
        while self.current_cycle <= self.n_cycles:

            # Minimize
            if self.sim_no == 0 and self.current_cycle == 0:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing...', flush=True)
                self.simulation.minimize()
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing finished.', flush=True)

            # Advance 1 cycle
            self._run_cycle()    
            
            
            
    def save_simulation(self, save_dir):
        """
        Save the important information from a simulation and then truncate the output.ncdf file to preserve disk space.
        """
        # Determine save no. 
        save_no_dir = os.path.join(save_dir, str(self.sim_no))
        if not os.path.exists(save_no_dir):
            os.mkdir(save_no_dir)

        # Truncate output.ncdf
        ncdf_copy = os.path.join(self.output_dir, 'output_copy.ncdf')
        pos, velos, box_vectors, states, energies, temperatures = truncate_ncdf(self.output_ncdf, ncdf_copy, self.reporter, False)
        np.save(os.path.join(save_no_dir, 'positions.npy'), pos.data)
        np.save(os.path.join(save_no_dir, 'velocities.npy'), velos.data)
        np.save(os.path.join(save_no_dir, 'box_vectors.npy'), box_vectors.data)
        np.save(os.path.join(save_no_dir, 'states.npy'), states.data)
        np.save(os.path.join(save_no_dir, 'energies.npy'), energies.data)
        np.save(os.path.join(save_no_dir, 'temperatures.npy'), temperatures)

        # Truncate output_checkpoint.ncdf
        checkpoint_copy = os.path.join(self.output_dir, 'output_checkpoint_copy.ncdf')
        truncate_ncdf(self.checkpoint_ncdf, checkpoint_copy, self.reporter, True)

        # Write over previous .ncdf files
        os.system(f'mv {ncdf_copy} {self.output_ncdf}')
        os.system(f'mv {checkpoint_copy} {self.checkpoint_ncdf}')

        # Close reporter object
        try:
            self.reporter.close()
        except:
            pass    
        
        return [t*unit.kelvin for t in temperatures]
            
        
        
    def _configure_simulation_parameters(self):
        """
        Configure simulation times to meet aggregate simulation time. 
        """            

        # Read number replicates if different than argument
        self.n_replicates = len(self.temperatures)
        
        # Configure times/steps
        sim_time_per_rep = self.sim_time / self.n_replicates
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
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of', self.n_replicates, 'replicates to be', [np.round(t,1) for t in self.temperatures], flush=True)        

        
        
    def _build_simulation(self):
        """
        """
        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)
        
        # Set up simulation
        self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        self.simulation._global_citation_silence = True

        # Remove existing .ncdf files
        if os.path.exists(self.output_ncdf):
            os.remove(self.output_ncdf)
        
        # Setup reporter
        atom_inds = tuple([i for i in range(self.system.getNumParticles())])
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=10, analysis_particle_indices=atom_inds)
        
        # Initialize sampler states if starting from scratch, otherwise they should be determinine in interpolation or passed through from Fulton Market
        if self.init_velocities is not None:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Setting initial positions with the "Velocity" method', flush=True)
            self.sampler_states = [SamplerState(positions=self.init_positions[i], box_vectors=self.init_box_vectors[i], velocities=self.init_velocities[i]) for i in range(self.n_replicates)]
        elif self.context is not None:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Setting initial positions with the "Context" method', flush=True)
            self.sampler_states = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors).from_context(self.context)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Setting initial positions with the "No Context" method', flush=True)
            if self.sim_no > 0:
                self.sampler_states = [SamplerState(positions=self.init_positions[i], box_vectors=self.init_box_vectors[i]) for i in range(self.n_replicates)]
            else:
                self.sampler_states = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors)

            
        self.simulation.create(thermodynamic_state=self.ref_state,
                               sampler_states=self.sampler_states,
                               storage=self.reporter, 
                               temperatures=self.temperatures,
                               n_temperatures=len(self.temperatures))

            
            
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
        if self.sim_no == 0:
            insert_inds = self._eval_acc_rates(self.init_overlap_thresh)
        else:
            insert_inds = self._eval_acc_rates(self.term_overlap_thresh)

        # Interpolate, if necessary
        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self._configure_simulation_parameters()
            self._build_simulation()
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
        # Read sampler states

        # Add pos, box_vecs, velos for new temperatures
        self.init_positions = np.insert(self.init_positions, insert_inds, [self.init_positions[ind-1] for ind in insert_inds], axis=0)
        self.init_box_vectors = np.insert(self.init_box_vectors, insert_inds, [self.init_box_vectors[state-1] for ind in insert_inds], axis=0)
        self.init_velocities = np.insert(self.init_velocities, insert_inds, [self.init_velocities[state-1] for ind in insert_inds], axis=0)
