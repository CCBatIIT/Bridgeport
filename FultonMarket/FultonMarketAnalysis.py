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
from typing import List
import seaborn as sns
from sklearn.decomposition import PCA
from pymbar.timeseries import detect_equilibration

fprint = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))



class FultonMarketAnalysis():
    """
    Analysis class for Replica Exchange Simulations written with Fulton Market

    methods:
        init: input_dir
    """
    def __init__(self, input_dir:str, pdb: str, skip: int=0, resids: List[int]=None):
        """
        get Numpy arrays, determine indices of interpolations, and set state_inds
        """
        
        # Find directores
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]
        self.input_dir = input_dir
        self.stor_dir = os.path.join(input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir)
        fprint(f"Found storage directory at {self.stor_dir}")
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'), key=lambda x: int(x.split('/')[-1]))
        self.pdb = pdb
        self.top = md.load_pdb(self.pdb).topology
        if resids is not None:
            self.resids = [[self.top.residue(i).resSeq for i in range(self.top.n_residues)].index(resid) for resid in resids] # convert to mdtraj resids
        
        # Load saved variables
        self.temperatures_list = [np.round(np.load(os.path.join(storage_dir, 'temperatures.npy'), mmap_mode='r'), decimals=2) for storage_dir in self.storage_dirs]
        self.temperatures = self.temperatures_list[-1]
        fprint(f"Shapes of temperature arrays: {[(i, temp.shape) for i, temp in enumerate(self.temperatures_list)]}")
        self.state_inds = [np.load(os.path.join(storage_dir, 'states.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        self.unshaped_energies = [np.load(os.path.join(storage_dir, 'energies.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        self.unshaped_positions = [np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        self.unshaped_box_vectors = [np.load(os.path.join(storage_dir, 'box_vectors.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        if os.path.exists(all([os.path.join(storage_dir, 'spring_centers.npy') for storage_dir in self.storage_dirs])):
            self.spring_centers_list = [np.load(os.path.join(storage_dir, 'spring_centers.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
            self.spring_centers = self.spring_centers_list[-1]
            
        # Reshape lists 
        self.energies = self._reshape_list(self.unshaped_energies)
        self.reduced_potentials = [e / get_kT(temps) for (e, temps) in zip(self.energies, self.temperatures_list)]

        # Compute positions/box_vectors map 
        self._get_postions_map()
        self.skip = skip        

        # Determine if interpolation occured and resample to fill in missing states
        self._backfill()
        fprint(f'Shape of final energies determined to be: {self.energies.shape}')

        
        
    def get_state_energies(self, state_indice: int=0):
        """
        get energies of each replicate in its own state (iters, state, state) -> (iters, state)
        Optionally reduce energies based on temperatures, and concatenate the list of arrays to a single array
        """
        
        state_energies = self.energies[:,state_indice, state_indice]
        
        return state_energies
    
    
    def get_average_energy(self, plot: bool=False, figsize: tuple=(6,6)):
        """
        """
        
        # Determine equilibration
        self._determine_equilibration()

        # Get average energies
        state_energies = np.empty((self.energies.shape[0], self.energies.shape[1]))
        for state in range(self.energies.shape[1]):
            state_energies[:,state] = self.energies[:,state,state]
        self.average_energies = state_energies.mean(axis=1)
        
        # Plot if specified:
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(range(self.average_energies.shape[0]), self.average_energies, color='k')
            ax.vlines(self.t0, self.average_energies.min(), self.average_energies.max(), color='r', label='equilibration')
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_title('Mean energy')
            ax.set_ylabel('Energy (kJ/mol)')
            ax.set_xlabel('Iterations')
            fig.tight_layout()
            plt.show
            return self.average_energies[self.t0:], fig, ax
        else:
            return self.average_energies[self.t0:]
        
        
    def plot_energy_distributions(self, figsize: tuple=(8,4)):   
        
        fig, ax = plt.subplots(figsize=figsize)
        
        state_energies = np.array([self.get_state_energies(state_indice=state_no) for state_no in range(self.energies.shape[1])]).T
        
        sns.kdeplot(state_energies, ax=ax, legend=False)
        ax.set_xlabel('Energy (kJ/mol)')
        fig.tight_layout()
        plt.show()
        
        return fig, ax 
    
    
    
    def importance_resampling(self, n_samples:int=1000, equilibration_method: str='PCA'):
        """
        """           
        self.equilibration_method = equilibration_method
        
        #Ensure equilibration has been detected
        self._determine_equilibration()
        
        # Create map to match shape of weights
  
        # Get MBAR weights
        if self.equilibration_method == 'energy':
            u_kln = self.reduced_potentials[self.t0:][self.uncorrelated_indices].T
            N_k = [len(self.uncorrelated_indices) for i in range(self.reduced_potentials.shape[1])]
            self.flat_inds = np.array([[state, ind] for ind in self.uncorrelated_indices for state in range(self.reduced_potentials.shape[1])])

        else:
            self.flat_inds = np.array([[state, ind] for ind in range(self.t0, self.reduced_potentials.shape[0]) for state in range(self.reduced_potentials.shape[1])])
            u_kln = self.reduced_potentials[self.t0:].T
            N_k = [self.reduced_potentials[self.t0:].shape[0] for i in range(self.reduced_potentials.shape[1])]
        self.resampled_inds, self.weights = resample_with_MBAR(objs=[self.flat_inds], u_kln=u_kln, N_k=N_k, size=n_samples, return_inds=False, return_weights=True, specify_state=0)
        
        

    def plot_weights(self, state_no: int=0, figsize: tuple=(4,4)):
        # Reshape weights
        self.weights = self.weights.copy().reshape(self.temperatures.shape[0], self.reduced_potentials.shape[0] - self.t0, self.temperatures.shape[0])[:,:,state_no].T
        
        # Get sum of weights by state
        sum_weights = self.weights.sum(axis=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(sum_weights.shape[0]), sum_weights, color='k')
        ax.set_ylabel('MBAR Weights')
        ax.set_xlabel('Temperature (K)')
        ax.set_xticks(range(self.temperatures.shape[0])[::10], self.temperatures[::10], rotation=90)
        fig.tight_layout()
        plt.show()
        
        return fig, ax
    
    
    
    def write_resampled_traj(self, pdb_out: str, dcd_out: str, return_traj: bool=False):
        
        # Make sure resampling has already occured
        if not hasattr(self, 'resampled_inds'):
            self.importance_resampling()
            
        # Load pos, box_vec
        self._load_positions_box_vecs()
        
        # Create mdtraj obj
        traj = md.load_pdb(self.pdb)
        
        # Use the map to find the resampled configurations
        pos = np.empty((len(self.resampled_inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(self.resampled_inds), 3, 3))
        for i, (state, iter) in enumerate(self.resampled_inds):
            
            # Use map
            sim_no, sim_iter, sim_rep_ind = self.map[iter, state].astype(int)
            
            pos[i] = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])
        
        # Apply pos, box_vec to mdtraj obj
        traj.xyz = pos.copy()
        traj.unitcell_vectors = box_vec.copy()
        traj.save_dcd(dcd_out)
        
        # Correct periodic issues
        traj = md.load(dcd_out, top=self.pdb)
        traj.image_molecules()
        traj[0].save_pdb(pdb_out)
        traj.save_dcd(dcd_out)
        
        if return_traj:
            return traj
       
    

    def state_trajectory(self, state_no=0, stride: int=1):
        """    
        State_no is the thermodynamics state to retrieve
        If pdb file is provided (top_file), then an MdTraj trajectory will be returned
        If top_file is None - the numpy array of positions will be returned
        """
        if not (hasattr(self, 'positions') or hasattr(self, 'box_vectors')):
            self._load_positions_box_vecs()
            
        # Create mdtraj obj
        traj = md.load_pdb(self.pdb)
        
        # Use the map to find the resampled configurations
        inds = np.arange(0, self.reduced_potentials.shape[0], stride)
        pos = np.empty((len(inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(inds), 3, 3))
        for i, ind in enumerate(inds):
            
            # Use map
            sim_no, sim_iter, sim_rep_ind = self.map[ind, state_no].astype(int)
            
            pos[i] = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])
        
        # Apply pos, box_vec to mdtraj obj
        traj.xyz = pos.copy()
        traj.unitcell_vectors = box_vec.copy()
        traj.save_dcd('temp.dcd')
        traj[0].save_pdb('temp.pdb')
        
        # Correct periodic issues
        traj = md.load('temp.dcd', top='temp.pdb')
        traj.image_molecules()
        
        # Align 
        prot_sel = self.top.select('protein and name CA')
        traj = traj.superpose(traj, frame=0, atom_indices=prot_sel, ref_atom_indices=prot_sel)
        

        return traj
    
    
    
    def _reshape_list(self, unshaped_list: List):
        """
        Return a list of reshaped arrays. 
        
        Parameters:
        ------------
            unshaped_list (List):
                List of numpy arrays that need the 1st axis reshaped 
        """
        reshaped_array = []
        for i in range(len(self.storage_dirs)):
            reshaped_array.append(self._reshape_array(unshaped_arr=unshaped_list[i], state_arr=self.state_inds[i]))
        return reshaped_array      

    
    
    def _reshape_array(self, unshaped_arr: np.array, state_arr: np.array):
        """
        Return a reshaped array. 

        Parameters:
        ------------
            unshaped_arr (np.array):
                Numpy arrays that need the 1st axis reshaped.

            state_arr (np.array)
                Numpy array with the state indices for each replicate
        """        
        # Reshape 1st axis
        reshaped_arr = np.empty(unshaped_arr.shape)
        for state in range(unshaped_arr.shape[1]):
            for iter_num in range(unshaped_arr.shape[0]):
                reshaped_arr[iter_num, state, :] = unshaped_arr[iter_num, np.where(state_arr[iter_num] == state)[0], :]
        
        return reshaped_arr
    
    
    
    def _get_postions_map(self):
        """
        Dimensions of map:  Dimensions are (sim_no, iteration, state) >>> corresponding[sim_no, state, iter]. EXAMPLE: self.map[1,0] could return [0, 1, 2] which means the correct positions for the 1st interation at the 0th state of the reshaped energies matrix can be cound at the the 2nd iteration of the 1st replicate of the 0th simulation.
        """
        
        
        # Make maps
        self.map = []
        for sim_no, sim_state_inds in enumerate(self.state_inds):
    
            # Build map for simulation
            sim_map = np.empty((self.energies[sim_no].shape[0], self.energies[sim_no].shape[1], 3), dtype=int)
    
            # Iterate through simulation iters, states to build map
            for sim_iter in range(sim_map.shape[0]):
                for sim_state in range(sim_map.shape[1]):
                    sim_map[sim_iter, sim_state, :] = np.array([sim_no, sim_iter, np.where(sim_state_inds[sim_iter] == sim_state)[0][0]], dtype=int)
    
            # Add sim_map
            self.map.append(sim_map.astype(int))


    def _determine_interpolation_inds(self):
        """
        determine the indices (with respect to the last simulation) which are missing from other simulations
        """
        # Set interpolation attribute
        if hasattr(self, 'spring_centers_list'):
            interpolation_list = [centers[:,0] for centers in self.spring_centers_list]
            final_set = self.spring_centers[:,0]
        else:
            interpolation_list = self.temperatures_list
            final_set = self.temperatures
            
        # Iterate through temperatures
        self.interpolation_inds = []
        for i, set_i in enumerate(interpolation_list):
            missing_sim_inds = []
            for i, s in enumerate(final_set):
                if s not in set_i:
                    missing_sim_inds.append(i)
        
            # Assert that interpolation made sense
            assert len(missing_sim_inds) + len(set_i) == len(final_set), f'{len(missing_sim_inds)}, {len(set_i)}, {len(final_set)}'
            self.interpolation_inds.append(missing_sim_inds)



    def _backfill(self):
        """
        
        """
        # Determine interpolation inds
        self._determine_interpolation_inds()
        fprint(f'Detected interpolations at: {self.interpolation_inds}')

        # Determine which simulations to resample from
        filled_sims = [True if not self.interpolation_inds[i] else False for i in range(len(self.interpolation_inds))]
        filled_sim_inds = [i for i in range(len(filled_sims)) if filled_sims[i] == True]

        #Make an interpolation map
        interpolation_map = [np.arange(self.temperatures.shape[0]) for i in range(len(self.temperatures))] # in shape (state, state) >>> state_ind
        for i, interpolation_ind_set in enumerate(self.interpolation_inds):
            for ind in interpolation_ind_set:
                interpolation_map[i] = interpolation_map[i][interpolation_map[i] != ind]

        # print(interpolation_map)

        # Iterate throught simulations to backfill energies
        backfilled_energies = []
        backfilled_potentials = []
        backfilled_map = []
        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            
            #Create an array for this simulations energies, in the final simulation's shape on axis 1, 2
            sim_energies = np.zeros((self.energies[sim_no].shape[0], self.temperatures.shape[0], self.temperatures.shape[0]))
            sim_reduced_potentials = np.zeros((self.reduced_potentials[sim_no].shape[0], self.temperatures.shape[0], self.temperatures.shape[0]))
            sim_map = np.zeros((self.map[sim_no].shape[0], self.temperatures.shape[0], 3))

            #Fill this array with the values that exist
            for i, ind in enumerate(interpolation_map[sim_no]):
                sim_energies[:, ind, interpolation_map[sim_no]] = self.energies[sim_no][:, i, :]
                sim_reduced_potentials[:, ind, interpolation_map[sim_no]] = self.reduced_potentials[sim_no][:, i, :]
                sim_map[:,ind] = self.map[sim_no][:,i]

            #Fill in rows and columns 
            for state_no in sim_interpolate_inds:

                # Get state-specific objects to resample from
                filled_reduced_potentials = np.concatenate([self.reduced_potentials[sim_no] for sim_no in filled_sim_inds])
                filled_energies = np.concatenate([self.energies[sim_no] for sim_no in filled_sim_inds])
                filled_map = np.concatenate([self.map[sim_no] for sim_no in filled_sim_inds])
                
                state_reduced_potentials = filled_reduced_potentials[:, state_no]
                state_energies = filled_energies[:, state_no]
                state_map = filled_map[:, state_no]

                N_k = np.array([state_reduced_potentials.shape[0]])

                # Resample
                res_potentials, res_energies, res_mappings, res_inds = resample_with_MBAR(objs=[state_reduced_potentials, state_energies, state_map], u_kln=np.array([state_reduced_potentials[:,state_no]]), N_k=N_k, reshape_weights=state_reduced_potentials.shape[0], return_inds=True, size=len(sim_energies))


                # Assign resampled configurations to empty rows/cols
                sim_energies[:, state_no] = res_energies.copy()
                sim_reduced_potentials[:, state_no] = res_potentials.copy()
                sim_map[:, state_no] = res_mappings.copy()

                sim_energies[:, :, state_no] = [filled_energies[resampled_ind, :, state_no] for resampled_ind in res_inds]
                sim_reduced_potentials[:, :, state_no] = [filled_reduced_potentials[resampled_ind, :, state_no] for resampled_ind in res_inds]


            backfilled_energies.append(sim_energies)
            backfilled_potentials.append(sim_reduced_potentials)
            backfilled_map.append(sim_map)

        # Concatenate
        self.energies = np.concatenate(backfilled_energies, axis=0)
        self.reduced_potentials = np.concatenate(backfilled_potentials, axis=0)
        self.map = np.concatenate(backfilled_map, axis=0)
    
    

    def _determine_equilibration(self):
        """
        Automated equilibration detection
        suggests an equilibration index (with respect to the whole simulation) by detecting equilibration for the average energies
        starting from each of the first 50 frames
        returns the likely best index of equilibration
        """

        
        # Compute average energies
        if self.equilibration_method == 'energy':
            
            self.average_energies = np.zeros((self.energies.shape[0], self.energies.shape[1]))
            for state_no in range(self.energies.shape[1]):
                self.average_energies[:,state_no] = self.get_state_energies(state_indice=state_no)
            self.average_energies = self.average_energies.mean(axis=1)

            t0, g, Neff_max = timeseries.detect_equilibration(self.average_energies) # compute indices of uncorrelated timeseries
            A_t_equil = self.average_energies[t0:]
            indices = timeseries.subsample_correlated_data(A_t_equil, g=g)
            A_n = A_t_equil[indices]
            self.t0 = t0
            self.uncorrelated_indices = indices
            
         
        elif self.equilibration_method == 'PCA':    
            # PCA
            pca, reduced_cartesian, explained_variance, n_components = self.get_PCA(explained_variance_threshold=0.9)
            
            # Iterate through PCs to detect equilibration
            equil_times = np.empty(n_components)
            for pc in range(n_components):
                equil_times[pc] = detect_PC_equil(pc, reduced_cartesian)


            # Save equilibration/uncorrelated inds to new variables
            weights = pca.explained_variance_ratio_[:n_components]
            self.t0 = np.sum(equil_times  * (weights / weights.sum())).astype(int)
          
        else:
            print('equilibration_method must be either PCA or energy')

        fprint(f'Equilibration detected at {np.round(self.t0 / 10, 3)} ns')

            
            

        

    def get_PCA(self, state_no: int=0, stride: int=10, n_components: int=2, explained_variance_threshold: float=None):
        """
        """
        # Get state trajectory
        traj = self.state_trajectory(state_no, stride)

        # Get protein or resids of interest
        if self.resids is not None:
            sele = traj.topology.select(f'resid {" ".join([str(resid) for resid in self.resids])}')
        else:
            sele = traj.topology.select('protein')
        traj = traj.atom_slice(sele)
    
        # PCA
        pca = PCA()
        reduced_cartesian = pca.fit_transform(traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3))
        explained_variance = np.array([np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(pca.n_components_)])

        if explained_variance_threshold is not None:
            n_components = int(np.where(explained_variance >= explained_variance_threshold)[0][0])

        return pca, reduced_cartesian[:,:n_components], explained_variance[:n_components], n_components

    
    
    def _load_positions_box_vecs(self):
        
        # Load
        self.positions = []
        self.box_vectors = []
        for sim_no, storage_dir in enumerate(self.storage_dirs):
            self.positions.append(np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r')[self.skip:])
            self.box_vectors.append(np.load(os.path.join(storage_dir, 'box_vectors.npy'), mmap_mode='r')[self.skip:]) 


    



        


@staticmethod
def resample_with_MBAR(objs: List, u_kln: np.array, N_k: np.array, size: int, reshape_weights: tuple=None, specify_state: int=0, return_inds: bool=False, return_weights: bool=False):

    # Get MBAR weights
    weights = compute_MBAR_weights(u_kln, N_k)

    # Reshape weights if specified
    if reshape_weights is not None:
        weights = weights.reshape(reshape_weights)
        
    # Get probabilities
    if len(weights.shape) == 1:
        probs = weights.copy()
    else:
        probs = weights[:, specify_state]


    # Resample
    resampled_inds = np.random.choice(range(len(probs)), size=size, replace=False, p=probs)
    resampled_objs = []
    for obj in objs:
        resampled_objs.append(np.array([obj[resampled_ind] for resampled_ind in resampled_inds]))

    # Return resampled objects
    return_list = []
    if len(objs) == 1:
        return_list.append(resampled_objs[0])
    elif len(objs) > 1:
        for resampled_obj in resampled_objs:
            return_list.append(resampled_obj)

    if return_inds:
        return_list.append(resampled_inds)
    if return_weights:
        return_list.append(weights)

    return return_list


@staticmethod
def compute_MBAR_weights(u_kln, N_k):
    """
    """

    mbar = MBAR(u_kln, N_k, initialize='BAR')

    return mbar.weights()


@staticmethod
# PCA
def detect_PC_equil(pc, reduced_cartesian):
    t0, _, _ = detect_equilibration(reduced_cartesian[:,pc])

    return t0*10










    

    



