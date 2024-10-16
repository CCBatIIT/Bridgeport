from openmm import *
from openmm.app import *
from openmmtools.utils.utils import TrackedQuantity
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
from copy import deepcopy
from shorten_replica_exchange import truncate_ncdf


spring_constant_unit = (unit.joule)/(unit.angstrom*unit.angstrom*unit.mole)

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
                 sampler_states=None,
                 context=None,
                 spring_constants:np.array=None,
                 restrained_atoms_dsl:str=None,
                 mdtraj_topology:md.Topology=None,
                 spring_centers:np.array=None):
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
        
        #Restraints if necessary
        self.restrained_atoms_dsl = restrained_atoms_dsl
        self.spring_constants = spring_constants
        self.mdtraj_topology = mdtraj_topology
        self.spring_centers = spring_centers
        
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

            # Minimize TODO: Reinsert
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
        
        if self.restrained_atoms_dsl is not None:
            spring_constants = np.array([np.round(t._value,2) for t in self.spring_constants])
            np.save(os.path.join(save_no_dir, 'spring_constants.npy'), spring_constants)
            np.save(os.path.join(save_no_dir, 'spring_centers.npy'), self.spring_centers)

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
        
        if self.restrained_atoms_dsl is not None:    
            return [t*unit.kelvin for t in temperatures], [t*spring_constant_unit for t in self.spring_constants], self.spring_centers
        else:
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
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of', self.n_replicates,
                                      'replicates to be', [np.round(t._value,1) for t in self.temperatures], flush=True)
        
        if self.restrained_atoms_dsl is not None:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated spring_constants of', self.n_replicates,
                                          'replicates to be', [np.round(t._value,1) for t in self.spring_constants], flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated spring_centers of', self.n_replicates,
                                          'replicates to be', [self.spring_centers[i].shape for i in range(self.spring_centers.shape[0])], flush=True)
        
    def _build_simulation(self):
        """
        """
        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)
        
        # Set up simulation
        if self.restrained_atoms_dsl is None:
            self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        else:
            self.simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=self.n_iters) #This is the case for PTwR and US
        self.simulation._global_citation_silence = True

        # Remove existing .ncdf files
        if os.path.exists(self.output_ncdf):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Removing', self.output_ncdf, flush=True)
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
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Setting initial positions individual to each state', flush=True)
                self.sampler_states = [SamplerState(positions=self.init_positions[i], box_vectors=self.init_box_vectors[i]) for i in range(self.n_replicates)]
            else:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Setting initial positions the same to each state', flush=True)
                self.sampler_states = SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors)
            
        if self.restrained_atoms_dsl is None:
            self.simulation.create(thermodynamic_state=self.ref_state, sampler_states=self.sampler_states,
                                   storage=self.reporter, temperatures=self.temperatures, n_temperatures=self.n_replicates)
        else:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Creating {len(self.temperatures)} Thermodynamic States', flush=True)
            thermodynamic_states = [ThermodynamicState(system=self.system, temperature=T) for T in self.temperatures]
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Done Creating Thermodynamic States', flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Assigning {len(self.spring_constants)} Restraints', flush=True)
            assert len(self.temperatures) == len(self.spring_constants)
            #In this case, iterate over the n_replicate spring_centers and assign different ones to each thermo_state
            for thermo_state, spring_cons, spring_center in zip(thermodynamic_states, self.spring_constants, self.spring_centers):
                self._restrain_atoms_by_dsl(thermo_state, self.mdtraj_topology, self.restrained_atoms_dsl, spring_cons, spring_center)
            
            self.simulation.create(thermodynamic_states=thermodynamic_states, sampler_states=self.sampler_states, storage=self.reporter)
        
        
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

        # Add new restraints if in PTwRE
        if self.restrained_atoms_dsl is not None:
            prev_spring_cons = [s._value for s in self.spring_constants]
            new_spring_cons = [cons for cons in prev_spring_cons]
            for displacement, ind in enumerate(insert_inds):
                cons_below = prev_spring_cons[ind-1]
                cons_above = prev_spring_cons[ind]
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state with Spring Constant', np.mean((cons_below, cons_above)), flush=True) 
                new_spring_cons.insert(ind + displacement, np.mean((cons_below, cons_above)))
            self.spring_constants = [cons * spring_constant_unit for cons in new_spring_cons]
            assert len(self.spring_constants) == len(self.temperatures)
            
            prev_spring_centers = self.spring_centers
            new_spring_centers = self.spring_centers
            for displacement, ind in enumerate(insert_inds):
                center_below = prev_spring_centers[ind - 1]
                center_above = prev_spring_centers[ind]
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state with new Spring Center', flush=True)
                new_center = 0.5*(center_above + center_below)
                new_spring_centers = np.insert(new_spring_centers, ind + displacement, new_center, axis=0)
            self.spring_centers = new_spring_centers
            assert self.spring_centers.shape[0] == len(self.temperatures)
        
        self.n_replicates = len(self.temperatures)

        # Only interpolate inital positions and box_vectors if not first simulation
        if self.sim_no > 0:
            
            # Add pos, box_vecs, velos for new temperatures
            self.init_positions = np.insert(self.init_positions, insert_inds, [self.init_positions[ind-1] for ind in insert_inds], axis=0)
            self.init_box_vectors = np.insert(self.init_box_vectors, insert_inds, [self.init_box_vectors[ind-1] for ind in insert_inds], axis=0)
            if self.init_velocities is not None:
                self.init_velocities = np.insert(self.init_velocities, insert_inds, [self.init_velocities[ind-1] for ind in insert_inds], axis=0)

            # Convert to quantities    
            self.init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=self.init_positions, mask=False, fill_value=1e+20), unit=unit.nanometer))
            self.init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=self.init_box_vectors, mask=False, fill_value=1e+20), unit=unit.nanometer))
            if self.init_velocities is not None:
                self.init_velocities = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=self.init_velocities, mask=False, fill_value=1e+20), unit=(unit.nanometer / unit.picosecond)))



    def _restrain_atoms_by_dsl(self, thermodynamic_state, topology, atoms_dsl, spring_constant, spring_center):
        """
        Unceremoniously Ripped from the OpenMMTools github, simply to change sigma to K
        Apply a soft harmonic restraint to the given atoms.
        This modifies the ``ThermodynamicState`` object.
        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state with the system. This will be modified.
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
            parameters = spring_center[index,:]
            restraint_force.addParticle(index, parameters)
        a_stupid_copied_system = thermodynamic_state.system
        a_stupid_copied_system.addForce(restraint_force)
        thermodynamic_state.system = a_stupid_copied_system

