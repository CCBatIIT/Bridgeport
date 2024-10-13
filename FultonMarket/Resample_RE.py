# Imports
import os, sys, math, glob
from datetime import datetime
import netCDF4 as nc
import numpy as np
from pymbar import timeseries, MBAR
import scipy.constants as cons
import mdtraj as md
#import dask.array as da
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt

fprint = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))

class RE_Analyzer():
    """
    Analysis class for Replica Exchange Simulations written with Fulton Market

    methods:
        init: input_dir
    """
    def __init__(self, input_dir:str):
        """
        Obtain Numpy arrays, determine indices of interpolations, and set state_inds
        """
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]
        self.input_dir = input_dir
        self.stor_dir = os.path.join(input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir)
        fprint(f"Found storage directory at {self.stor_dir}")
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'), key=lambda x: int(x.split('/')[-1]))

        self.state_inds = [np.load(os.path.join(storage_dir, 'states.npy'), mmap_mode='r') for storage_dir in self.storage_dirs]
        fprint(f"Shapes of temperature arrays: {[(i, temp.shape) for i, temp in enumerate(self.obtain_temps())]}")

        self.interpolation_inds = self.determine_interpolation_inds()

    
    def obtain_temps(self):
        """
        Obtain a list of temperature arrays associated with each simulation in the run

        Returns:
            temps: [np.array]: list of arrays for temperatures
        """
        return [np.round(np.load(os.path.join(storage_dir, 'temperatures.npy'), mmap_mode='r'), decimals=2) for storage_dir in self.storage_dirs]
        
    
    def _reshape_energy(self, storage_dir=None, sim_num=None, reduce=True):
        """
        Reshape the energy array of storage_dir from (iter, replicate, state) to (iter, state, state)
        Providing the storage_dir argument overrides the cycle number argument
        cycle number is provided as an integer, one of the directories in input_dir/saved_variables/

        Input:
            storage_dir as string or sim_num as string/int
                the saved_variables directory as a string or the integer representing the saved variables dir

        Returns:
            reshaped_energy: np.array: energy array of the same shae as energies.npy
                Reshaped from (iter, replicate, state) to (iter, state, state)
        """
        assert storage_dir is not None or sim_num is not None
        if storage_dir is None and sim_num is not None:
            storage_dir = self.storage_dirs[[stor_dir.endswith(str(sim_num)) for stor_dir in self.storage_dirs].index(True)]
        elif storage_dir is not None:
            assert storage_dir in self.storage_dirs
        
        energy_arr = np.load(os.path.join(storage_dir, 'energies.npy'), mmap_mode='r')
        state_arr = np.load(os.path.join(storage_dir, 'states.npy'), mmap_mode='r')
        reshaped_energy = np.empty(energy_arr.shape)
        for state in range(energy_arr.shape[1]):
            for iter_num in range(energy_arr.shape[0]):
                reshaped_energy[iter_num, state, :] = energy_arr[iter_num, np.where(state_arr[iter_num] == state)[0], :]
        
        if reduce:
            temps = np.round(np.load(os.path.join(storage_dir, 'temperatures.npy'), mmap_mode='r'), decimals=2)
            reshaped_energy = reshaped_energy / get_kT(temps)
        
        return reshaped_energy
    
    
    def obtain_reshaped_energies(self, reduce=True):
        """
        Iterate self._reshape_energy over the storage directories, provide the result as a list of arrays

        Returns:
            reshaped_energies: [np.array]
        """
        reshaped_energies = []
        for i in range(len(self.storage_dirs)):
            reshaped_energies.append(self._reshape_energy(sim_num=i, reduce=reduce))
        return reshaped_energies


    def concatenate(self, array_list:[np.array], axis=0):
        """
        Shorthand for typical concatenations done here, default over axis zero
        """
        return np.concatenate(array_list, axis=axis)
        
        
    def state_trajectory(self, state_no, top_file=None):
        """
        Only for non-interpolated positions array sets, for now
    
        State_no is the thermodynamics state to retrieve
        If pdb file is provided (top_file), then an MdTraj trajectory will be returned
        If top_file is None - the numpy array of positions will be returned
        """
        positions, box_vecs = self.obtain_positions_box_vecs()
        total_iters = np.sum([state_arr.shape[0] for state_arr in self.state_inds])
        pos = np.empty((total_iters, positions[0].shape[2], 3))
        box_vec = np.empty((total_iters, 3, 3))
    
        sim_counter = 0
        for pos_arr, box_arr, state_arr in zip(positions, box_vecs, self.state_inds):
            pos[sim_counter : state_arr.shape[0] + sim_counter] = pos_arr[np.where(state_arr == state_no)]
            box_vec[sim_counter : state_arr.shape[0] + sim_counter] = box_arr[np.where(state_arr == state_no)]
            sim_counter += state_arr.shape[0]
    
        if top_file is None:
            return pos
        else:
            traj = md.load_pdb(top_file)
            traj.xyz = pos.copy()
            traj.unitcell_vectors = box_vec.copy()
            traj.save_dcd('temp.dcd')
            traj = md.load('temp.dcd', top=top_file)
            traj.image_molecules()
            return traj
    
    def obtain_state_specific_energies(self, energies=None, concat=True, reduce=True):
        """
        Obtain energies of each replicate in its own state (iters, state, state) -> (iters, state)
        Optionally reduce energies based on temperatures, and concatenate the list of arrays to a single array
        """
        if energies is None:
            energies = self.obtain_reshaped_energies(reduce=reduce)
        
        if type(energies) == list:
            from_numpy = False
        elif type(energies) == np.ndarray:
            from_numpy = True
        else:
            raise Exception('(>_<)')

        if from_numpy:
            specific_energies = np.empty(energies.shape[:-1])
            for j in range(specific_energies.shape[1]):
                specific_energies[:, j] = energies[:, j, j]
        else:
            specific_energies = []
            for ener_arr in energies:
                spec_ener = np.empty(ener_arr.shape[:-1])
                for j in range(spec_ener.shape[1]):
                    spec_ener[:, j] = ener_arr[:, j, j]
                specific_energies.append(spec_ener)
            if concat:
                specific_energies = self.concatenate(specific_energies)

        return specific_energies

    
    def gather_uncorrelated_samples(self, A_t):
        """
        Gather a series of uncorrelated samples from a correlated energy timeseries
        """
        from pymbar import timeseries
        t0, g, Neff_max = timeseries.detect_equilibration(A_t) # compute indices of uncorrelated timeseries
        A_t_equil = A_t[t0:]
        indices = timeseries.subsample_correlated_data(A_t_equil, g=g)
        A_n = A_t_equil[indices]
        return t0, g, Neff_max, indices, A_n


    def determine_equilibration(self, ave_energies=None, depth=100):
        """
        Automated equilibration detection
        suggests an equilibration index (with respect to the whole simulation) by detecting equilibration for the average energies
        starting from each of the first 50 frames
        returns the likely best index of equilibration
        """
        if ave_energies is None:
            ave_energies = self.average_energy()
        t0s_inds = []
        for i in range(depth):
            A_t = ave_energies[i:]   
            t0, _, _ = timeseries.detect_equilibration(A_t)
            t0s_inds.append([i, t0+i])
        t0s_inds = np.array(t0s_inds)
        counts = np.bincount(t0s_inds[:, 1])
        t0 = np.argmax(counts)
        return t0, ave_energies[t0:]


    def average_energy(self, energies=None, reduce=True):
        """
        Returns a one dimensional array of the average energy of replicates against simulation time
        """
        if energies is None:
            energies = self.obtain_state_specific_energies(concat=True, reduce=reduce)

        return np.mean(energies, axis=1)


    def free_energy_difference(self, t0=None, uncorr_indices=None, energies=None):
        """
        Calculate Free Energy differences using the MBAR method
        """
        raise NotImplementedError


    def obtain_positions_box_vecs(self):
        """
        Extract the positions and box_vectors from numpy arrays
        Returns
            positions:[np.array]
            box_vectors:[np.array]
        """
        # Extract trajectory information
        positions = []
        box_vectors = []
        for sim_no, storage_dir in enumerate(self.storage_dirs):
            positions.append(np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r'))
            box_vectors.append(np.load(os.path.join(storage_dir, 'box_vectors.npy'), mmap_mode='r'))
        
        return positions, box_vectors
    
    
    def determine_interpolation_inds(self):
        """
        determine the indices (with respect to the last simulation) which are missing from other simulations
        """
        missing_indices = []
        temps = self.obtain_temps()
        final_temps = temps[-1]
        for i, temp_arr in enumerate(temps):
            sim_inds = []
            for i, temp in enumerate(final_temps):
                if temp not in temp_arr:
                    sim_inds.append(i)
            missing_indices.append(sim_inds)
        return missing_indices


    def obtain_MBAR_weights(self, backfilled_energies=None, t0=None):
        """
        Given an energy matrix, and an equilibration time, determine the MBAR weights of post equilibration samples

        To determine weights of all samples, feed t0 = 0
        """
        if backfilled_energies is None:
            backfilled_energies, _ = self.backfill_energies()
        if t0 is None:
            t0, _ = self.determine_equilibration()
        # Get MBAR weights    
        u_kln = deepcopy(backfilled_energies[t0:])
        N_k = np.array([backfilled_energies.shape[0] - t0 for i in range(backfilled_energies.shape[1])])
        mbar = MBAR(u_kln.T, N_k, initialize='BAR')
        weights = mbar.weights()
        return weights, backfilled_energies, t0


    def obtain_resampled_configs_indices(self, n_frames=500, weights=None, backfilled_energies=None, t0=None):
        """
        """
        if backfilled_energies is None:
            backfilled_energies, _ = self.backfill_energies()
        if t0 is None:
            t0, _ = self.determine_equilibration()
        if weights is None:
            weights, _, _ = self.obtain_MBAR_weights(backfilled_energies=backfilled_energies, t0=t0)
        
        # Reshape state, iters to match weights
        flat_inds = np.array([[state, ind] for ind in range(t0, backfilled_energies.shape[0]) for state in range(backfilled_energies.shape[1])])
        print('Flattened inds', flat_inds.shape, flush=True)
        
        # Resample based on weights
        resampled = np.random.choice(np.arange(0, len(flat_inds), 1), size=n_frames, replace=False, p=weights[:,0])
        resampled_configs = flat_inds[resampled]
        return resampled_configs


    def make_positions_map(self, backfilled_energies=None, backfill_indices=None):
        """
        """
        if backfilled_energies is None or backfill_indices is None:
            backfilled_energies, backfill_indices = self.backfill_energies()

        # Extract trajectory information
        positions, _ = self.obtain_positions_box_vecs()
        
        # Make maps
        positions_map = np.empty((backfilled_energies.shape[1], backfilled_energies.shape[0], 3), dtype=int) # Dimensions are (states, iters, corresponding[sim_no, state, iter])
        sim_counter = 0
        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            shift = 0
            for state_no in range(backfilled_energies.shape[1]):
                if state_no in sim_interpolate_inds:
                    backfilled_index = sim_interpolate_inds.index(state_no)
                    for iter, backfilled_ind in enumerate(backfill_indices[sim_no][backfilled_index]):
                        counter = 0
                        for fill_sim_ind in filled_sim_inds:
                            if backfilled_ind >= counter and backfilled_ind < positions[fill_sim_ind].shape[0]+counter:
                                positions_map[state_no, iter+sim_counter] = np.array([fill_sim_ind, state_no-shift, backfilled_ind-counter])
                                break
                            else:
                                counter += positions[fill_sim_ind].shape[0]
                    shift += 1
                else:
                    for iter in range(positions[sim_no].shape[0]):
                        positions_map[state_no, iter+sim_counter] = np.array([sim_no, state_no-shift, iter])
            sim_counter += positions[sim_no].shape[0]
        return positions_map


    def write_resampled_trajectory(self, pdb_in_fn, resampled_configs=None, positions_map=None,
                                   n_frames=500, output_dcd=None, output_pdb=None, write_result=True):
        """
        """
        
        if resampled_configs is None:
            resampled_configs = self.obtain_resampled_configs_indices(n_frames=n_frames)
        if positions_map is None:
            positions_map = self.make_positions_map()
        
        # Extract trajectory information
        positions, box_vectors = self.obtain_positions_box_vecs()
        
        #Parse input pdb
        if not os.path.isfile(os.path.join(self.input_dir, pdb_in_fn)):
            if not os.path.isabs(pdb_in_fn):
                raise Exception('Could not find input pbd')
            os.system(f'scp {pdb_in_fn} {os.path.join(self.input_dir, pdb_in_fn)}')
        pdb_in_fn = os.path.join(self.input_dir, pdb_in_fn)
        
        # Write new trajectory from resampled frames
        write_dir = os.path.join(self.input_dir, 'resampled')
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
        name = pdb_in_fn.split('/')[-1][:-4] #take off prefacing directories and extension
        traj = md.load_pdb(pdb_in_fn)
        
        pos = np.empty((n_frames, positions[0].shape[2], 3))
        box_vec = np.empty((n_frames, 3, 3))
        
        if output_pdb is None:
            output_pdb = os.path.join(write_dir, 'resampled.pdb')
        if output_dcd is None:
            output_dcd = os.path.join(write_dir, 'resampled.dcd')
        
        # Iterate through resampled frames
        for i, (state, iter) in enumerate(resampled_configs):
        
            # Get indices using positions_map
            frame_sim_no, frame_state_no, frame_sim_iter = positions_map[state, iter]
        
            try:
                rep_ind = np.where(self.state_inds[frame_sim_no][frame_sim_iter] == frame_state_no)[0][0]
            except:
                print(state, iter)
                print(frame_sim_no, frame_state_no, frame_sim_iter, '\n', self.state_inds[frame_sim_no][frame_sim_iter], frame_state_no)
                raise Exception()
        
            pos[i] = np.array(positions[frame_sim_no][frame_sim_iter][rep_ind])
            box_vec[i] = box_vectors[frame_sim_no][frame_sim_iter][rep_ind]
        
        traj.xyz = pos.copy()
        traj.unitcell_vectors = box_vec.copy()
        traj.save_dcd(output_dcd)
        
        traj = md.load(output_dcd, top=pdb_in_fn)
        traj.image_molecules()
        if write_result:
            traj[0].save_pdb(output_pdb)
            traj.save_dcd(output_dcd)

        return traj


    def resample_from_post_equilibration(self, pdb_in_fn, n_frames=500,
                                         output_dcd=None, output_pdb=None):
        
        backfilled_energies, backfill_indices = self.backfill_energies()
        t0, _ = self.determine_equilibration()
        
        weights, _, _ = self.obtain_MBAR_weights(backfilled_energies=backfilled_energies,
                                                 t0=t0)
        
        resampled_configs = self.obtain_resampled_configs_indices(n_frames=n_frames,
                                                                  weights=weights,
                                                                  backfilled_energies=backfilled_energies,
                                                                  t0=t0)
        
        
        positions_map = self.make_positions_map(backfilled_energies=backfilled_energies,
                                                backfill_indices=backfill_indices)
        
        resampled_traj = self.write_resampled_trajectory(pdb_in_fn,
                                                         resampled_configs=resampled_configs,
                                                         positions_map=positions_map,
                                                         n_frames=n_frames,
                                                         output_dcd=output_dcd,
                                                         output_pdb=output_pdb,
                                                         write_result=True)
        return resampled_traj


    
        
    def resample_an_interval(self, pdb_in_fn, start, stop, n_frames=500,
                            output_dcd=None, output_pdb=None):
        fprint(f'Resampling on interval {start} - {stop}')
        #Get energies
        backfilled_energies, backfill_indices = self.backfill_energies()
        energies_on_interval = backfilled_energies[start:stop]
        #Average energy of states in their own state, on the interval
        specific_energies = np.empty(energies_on_interval.shape[:-1])
        for j in range(specific_energies.shape[1]):
            specific_energies[:, j] = energies_on_interval[:, j, j]
        ave_energies = np.mean(specific_energies, axis=1)
        fprint(f'Energies Retrieved')
        t0, _ = self.determine_equilibration(ave_energies=ave_energies)
        fprint(f'Auto equil is {t0}')
        weights, _, _ = self.obtain_MBAR_weights(backfilled_energies=energies_on_interval,
                                                 t0=t0)
        fprint(f'Weights Retrieved')
        resampled_configs = self.obtain_resampled_configs_indices(n_frames=n_frames,
                                                                  weights=weights,
                                                                  backfilled_energies=energies_on_interval,
                                                                  t0=t0)
        
        resampled_configs[:, 1] += start
        fprint(f'Resampled Configs')
        
        positions_map = self.make_positions_map(backfilled_energies=backfilled_energies,
                                                backfill_indices=backfill_indices)
        fprint(f'Positions Map Made')
        
        
        resampled_traj = self.write_resampled_trajectory(pdb_in_fn,
                                                         resampled_configs=resampled_configs,
                                                         positions_map=positions_map,
                                                         n_frames=n_frames,
                                                         output_dcd=output_dcd,
                                                         output_pdb=output_pdb,
                                                         write_result=True)
        fprint(f'Done Resampling {n_frames} on interval {start}:{stop}')
        return resampled_traj


    def backfill_energies(self, energies:[np.array]=None):
        """
        Under development

        Currently works fine for non-interpolated groups of simulations (all temps same size)
        """
        if energies is None:
            energies = self.obtain_reshaped_energies()
        
        
        temps = self.obtain_temps()
        filled_sims = [True if not self.interpolation_inds[i] else False for i in range(len(self.interpolation_inds))]
        filled_sim_inds = [i for i in range(len(filled_sims)) if filled_sims[i] == True]
        final_temps = temps[-1]
        
        # #Make an interpolation map
        # interpolation_map = [np.arange(final_temps.shape[0]) for i in range(len(temps))]
        # for i, interpolation_ind_set in enumerate(interpolation_inds):
        #     for ind in interpolation_ind_set:
        #         interpolation_map[i] = interpolation_map[i][interpolation_map[i] != ind]

        # backfilled_energies = []

        # for sim_no, sim_interpolate_inds in enumerate(interpolation_inds):
        #     #Create an array for this simulations energies, in the final simulation's shape on axis 1, 2
        #     sim_energies = np.zeros((energies[sim_no].shape[0], final_temps.shape[0], final_temps.shape[0]))
        #     #Fill this array with the values that exist
        #     for i, ind in enumerate(interpolation_map[sim_no]):
        #         sim_energies[:, ind, interpolation_map[sim_no]] = energies[sim_no][:, i, :]
        #     #Fill in rows and columns (
        #     #TODO
        #     backfilled_energies.append(sim_energies)

        # return self.concatenate(backfilled_energies)
        
        # Resample from MBAR weights
        backfilled_energies = []
        backfill_indices = [[] for i in range(len(self.interpolation_inds))]
        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            sim_energies = np.zeros((energies[sim_no].shape[0], final_temps.shape[0], final_temps.shape[0])) # reformat into dimensions (iterations, state, state)
            if len(sim_interpolate_inds) > 0:
                fill_states = np.array(sim_interpolate_inds)
                sim_keep_inds = np.array([i for i in range(len(final_temps)) if i not in sim_interpolate_inds])
                shift = 0
                for state_no in range(sim_energies.shape[1]):
                    # Proceed if state needs to be filled
                    if state_no in fill_states:

                        # Get MBAR weights for filled energy state
                        temp_energies = np.concatenate([energies[i] for i in filled_sim_inds])
                        temp_state_energies = np.array([np.concatenate([energies[i] for i in filled_sim_inds])[:, state_no, state_no]])
                        N_k = np.array([temp_state_energies.shape[1]])
                        mbar = MBAR(temp_state_energies, N_k=N_k, initialize='BAR')
                        weights = mbar.weights().reshape(temp_state_energies.shape[1])
                        
                        # Resample    
                        resampled_inds = np.random.choice(range(len(weights)), size=len(energies[sim_no]), replace=False, p=weights)
                        backfill_indices[sim_no].append(list(resampled_inds))
                        sim_energies[:,state_no,:] = np.array([temp_energies[resampled_ind, state_no, :] for resampled_ind in resampled_inds]).copy()
                        
                        # Add to shift
                        shift += 1

                    # If no resampling is needed, copy original energies
                    else:
                        sim_energies[:, state_no, sim_keep_inds] = energies[sim_no][:, state_no-shift, :].copy()
                        sim_temps = temps[sim_no]
                        for sim_interpolate_i in sim_interpolate_inds:
                            sim_energies[:,state_no,sim_interpolate_i] = (energies[sim_no][:,state_no-shift,state_no-shift] * sim_temps[sim_interpolate_i-shift] / final_temps[sim_interpolate_i]).copy() # mu_2 = m_1 * (T_1 / T_2)
                            

                # Add backfilled simulation energies
                backfilled_energies.append(sim_energies)
                    
            else:
                backfilled_energies.append(energies[sim_no])


        return np.concatenate(backfilled_energies, axis=0), backfill_indices