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

fprint = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + my_string, flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]

class RE_Analyzer():
    """
    Analysis class for Replica Exchange Simulations written with Fulton Market

    methods:
        init: input_dir
    """
    def __init__(self, input_dir):
        """
        Obtain Numpy arrays, determine indices of interpolations,
        """
        self.input_dir = input_dir
        self.stor_dir = os.path.join(input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir)
        fprint(f"Found storage directory at {self.stor_dir}")
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'), key=lambda x: int(x.split('/')[-1]))
        #determine if resampling is necessary
        fprint(f"Shapes of temperature arrays: {[(i, temp.shape) for i, temp in enumerate(self.obtain_temps())]}")

    
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
        return np.concatenate(array_list, axis=axis)


    def obtain_state_specific_energies(self, energies=None, concat=True, reduce=True):
        """
        Obtain energies of each replicate in its own state (iters, state, state) -> (iters, state)
        Optionally reduce energies based on temperatures, and concatenate the list of arrays to a single array
        """
        if energies is None:
            energies = self.obtain_reshaped_energies(reduce=reduce)
        specific_energies = []
        for ener_arr in energies:
            spec_ener = np.empty(ener_arr.shape[:-1])
            for iter_num in range(spec_ener.shape[0]):
                for j in range(spec_ener.shape[1]):
                    spec_ener[iter_num, j] = ener_arr[iter_num, j, j]
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
        pass


    def determine_interpolation_inds(self):
        """
        determine the indices (with respect to the last simulation) which are missing from other simulations
        """
        missing_indices = []
        temps = self.obtain_temps()
        final_temps = temps[-1]
        for i, temp_arr in enumerate(temps):
            sim_inds = []
            for temp in final_temps:
                if temp not in temp_arr:
                    sim_inds.append(np.where(final_temps == temp)[0][0])
            missing_indices.append(sim_inds)
        return missing_indices


    def obtain_positions_arrays(self, storage_dir=None, sim_num=None):
        """
        """
        assert storage_dir is not None or sim_num is not None
        if storage_dir is None and sim_num is not None:
            storage_dir = self.storage_dirs[[stor_dir.endswith(str(sim_num)) for stor_dir in self.storage_dirs].index(True)]
        elif storage_dir is not None:
            assert storage_dir in self.storage_dirs
        
        positions = np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r')
        state_arr = np.load(os.path.join(storage_dir, 'states.npy'), mmap_mode='r')
        
        
        
        # reshaped_energy = np.empty(energy_arr.shape)
        # for state in range(energy_arr.shape[1]):
        #     for iter_num in range(energy_arr.shape[0]):
        #         reshaped_energy[iter_num, state, :] = energy_arr[iter_num, np.where(state_arr[iter_num] == state)[0], :]
        
        # if reduce:
        #     temps = np.round(np.load(os.path.join(storage_dir, 'temperatures.npy'), mmap_mode='r'), decimals=2)
        #     reshaped_energy = reshaped_energy / get_kT(temps)
        
        return positions
    
    
    def resample_energies(self, energies:[np.array]=None):
        """
        """
        if energies is None:
            energies = self.obtain_reshaped_energies()
        temps = self.obtain_temps()
        interpolation_inds = self.determine_interpolation_inds()
        filled_sims = [i for i in range(len(interpolation_inds)) if not interpolation_inds[i]]
        filled_energies = []
        for i in filled_sims:
            filled_energies.append(energies[i])
        filled_energies = self.concatenate(filled_energies)
        filled_temps = temps[-1]
        print(filled_energies.shape, filled_temps.shape)

        # interpolation_inds = [[] for i in range(len(energies))]
        # filled_sims = [True for i in range(len(energies))]
        # sim_temps = [[] for i in range(len(energies))]
        # for sim_no, sim_energies in enumerate(energies):
        #     # Find interpolation indices
        #     sim_temps[sim_no] = np.logspace(np.log10(T_min), np.log10(T_max), len(energies[sim_no][1]))

        #     if len(sim_temps[sim_no]) != len(final_temps):
        #         while len(sim_temps[sim_no]) < len(final_temps):
        #             for state_ind, (t_0, t_f) in enumerate(zip(sim_temps[sim_no], final_temps)):
        #                 if np.round(t_0, 4) != np.round(t_f, 4):
        #                     interpolation_inds[sim_no].append(state_ind)
        #                     sim_temps[sim_no] = np.insert(sim_temps[sim_no], state_ind, t_f)
        #                     filled_sims[sim_no] = False
        #                     break
        