#!/usr/bin/env python
import textwrap
import MDAnalysis as mda
import openmmtools as mmtools
import yank.utils
import numpy as np
import yaml
import yank
from yank.experiment import YankLoader, YankDumper
import netCDF4 as nc
import matplotlib.pyplot as plt
import sys, os, glob, shutil
from mdtraj.formats.dcd import DCDTrajectoryFile
import mdtraj as md

from ybp_utils import *


class YankWrapper():
    """Wrapper for YANK that
    * TODO: initializes configurations from a set of predefined structures
    * insert thermodynamic states as necessary
    * TODO: performs a pose prediction by clustering of ligand positions
    """
    def __init__(self, yaml_init_fn):
        """
        
        Parameters
        ----------
        yaml_fn : string
            The yaml file that is used to initialize YANK with an automated protocol
        
        """
        # Load the initial yaml. Based on this, determine yank_store directory and current yaml
        self.yaml_init_fn = os.path.abspath(yaml_init_fn)
        with open(self.yaml_init_fn, 'r') as f:
            self.yaml_init = yaml.load(f, Loader=YankLoader)
        try:
            self.yank_store = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}'
        except:
            self.yank_store = os.path.join(os.path.dirname(self.yaml_init_fn), self.yaml_init['options']['output_dir'])
        self.yaml_current_fn = os.path.join(self.yank_store,'experiments','experiments.yaml')
        assert hasattr(self, 'yaml_init_fn') and hasattr(self, 'yaml_init') 
        
        #Try to run yank, if it has been run, then this will simply pass
        yaml_builder = yank.experiment.ExperimentBuilder(script=self.yaml_init)
        yaml_builder.run_experiments()
                
        #YANK Throws a strange error, it cannot be restarted with the file names as elements of the dictionary
        # see such lines created by yank that look like (below) and should be deleted from yaml_current prior
        # /ocean/projects/bio230003p/dcooper/PosePred/data/fentanyl/fentanyl_1.inpcrd: ../../fentanyl_1.inpcrd
        # a simple fix may be to delete any lines containing '../..' for now
        with open(self.yaml_current_fn, 'r') as f: #Read and filter contents
            lines = [line for line in f.readlines() if "../.." not in line]
        with open(self.yaml_current_fn, 'w') as f: #write
            f.writelines(lines)
        with open(self.yaml_current_fn, 'r') as f:
            self.yaml_current = yaml.load(f, Loader=YankLoader)
        
        assert hasattr(self, 'yaml_current_fn') and hasattr(self, 'yaml_current')
        self.yaml_current['options']['output_dir'] = self.yank_store

        ##write progress.log file 
        if os.path.exists(prog_path):
            pass 
        else:
            with open(prog_path, 'a+') as f:
                f.write(f'LOG FILE FOR {name} {run} YANK POSE PREDICTION\nOne Simulation Ran\n')
                f.close()
        
        
    
    def n_more_iters(self, n):
        self.yaml_init['options']['default_number_of_iterations'] += n
        with open(self.yaml_current_fn, 'w') as f:
            f.write(yaml.dump(self.yaml_current, Dumper=YankDumper))

    def acceptance_rate(self, phase='complex'):
        """ Evaluate the replica exchange acceptance rate between neighbors
        """
        from openmmtools.multistate.multistatereporter import MultiStateReporter
        nc_fn = os.path.join(self.yank_store,'experiments',f'{phase}.nc')
        msr = MultiStateReporter(nc_fn,'r')
        (n_accepted_matrix, n_proposed_matrix) = msr.read_mixing_statistics()
        msr.close()
        fifth = int(n_accepted_matrix.shape[0]/5.)
        acc_rate_matrix = np.sum(n_accepted_matrix[-fifth:,:,:],0) / \
                          np.sum(n_proposed_matrix[-fifth:,:,:],0)
        acc_rate = np.array([acc_rate_matrix[i][i+1] \
                             for i in range(acc_rate_matrix.shape[0] - 1)])
        return acc_rate            
            
    def run_yank(self):
        yaml_builder = yank.experiment.ExperimentBuilder(script=self.yaml_current)
        yaml_builder.run_experiments()
        
    def get_nc_state(self, state, dcd_save_fn, phase='complex', yank_nc_fn=True):
        if yank_nc_fn == True:
            yank_nc_fn = f'{self.yank_store}/experiments/{phase}.nc'
        else:
            pass
        
        ncdf = nc.Dataset(yank_nc_fn, 'r+')
        
        positions = np.zeros((ncdf.dimensions['iteration'].size, \
                              ncdf.dimensions['atom'].size, \
                              ncdf.dimensions['spatial'].size))
        
        for i in range(ncdf.dimensions['iteration'].size):
            state_ind = ncdf.variables['states'][i, :].index(state)
            positions[i, :, :] = ncdf.variables['positions'] #nm positions
        
        with DCDTrajectoryFile(dcd_save_fn, 'w') as f:
            f.write(positions)
        
        
    def _insert_states(self, threshold=0.45, phase='complex'):
        """ Inserts a state whereever there is a bottleneck
        """
        assert phase == 'complex' or phase == 'solvent'
        
        acc_rate = self.acceptance_rate(phase)
        inds = [ind for ind in range(acc_rate.shape[0]) if acc_rate[ind] < threshold]
        print(inds, acc_rate)
        if len(inds)==0:
            return False
        
        # Insert states into protocol
        n_states_o = acc_rate.shape[0]+1
        n_states_n = n_states_o + len(inds)
        print(f'Starting with {n_states_o} states, inserting {len(inds)} states')
        
        # Expand the protocol
        alchemical_paths = self.yaml_current['protocols']['absolute-binding'][phase]['alchemical_path']
        for key in alchemical_paths.keys():
            for ind in inds[::-1]:
                mean_parameter = (alchemical_paths[key][ind] + alchemical_paths[key][ind+1])/2
                alchemical_paths[key] = alchemical_paths[key][:ind+1] + [mean_parameter] + \
                                        alchemical_paths[key][ind+1:]
        print(len(alchemical_paths[key]), n_states_n)
        # assert len(alchemical_paths[key])==n_states_n
        
        # Write new protocol to the current YAML
        import shutil
        shutil.copy(self.yaml_current_fn,
                    self.yaml_current_fn[:self.yaml_current_fn.rfind('.')] + f'_{n_states_o}.yaml')
        with open(self.yaml_current_fn, 'w') as f:
            f.write(yaml.dump(self.yaml_current, Dumper=YankDumper))
        
        # Opens the YANK netcdf file and coordinates.dcd file
        yank_nc_fn = f'{self.yank_store}/experiments/{phase}.nc'
        ncdf = nc.Dataset(yank_nc_fn, 'r+')
        if phase == 'complex':
            phase_num = 'phase1_path'
        elif phase == 'solvent':
            phase_num = 'phase2_path'
        coords_traj = md.load_dcd(f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates.dcd',\
                                 top = self.yaml_current['systems']['rec-lig'][phase_num][0])
        with open(f'{self.yank_store}/experiments/trailblaze/{phase}/states_map.json', 'r') as f:
            states_map_inds = eval(f.read())
        
        # Extract a positions from coords.dcd and box vectors from the YANK netcdf output
        
        # EXTRACTING FROM THE NETCDF DOES NOT GET WATERS which are required in coords.dcd
        positions = np.zeros((ncdf.dimensions['replica'].size, \
                          coords_traj.xyz.shape[1], \
                          ncdf.dimensions['spatial'].size))
        cell_lengths = np.zeros((ncdf.dimensions['replica'].size, \
                                 ncdf.dimensions['spatial'].size))
        for r in range(ncdf.dimensions['replica'].size):
            s = ncdf.variables['states'][-1, r] # State index of replica r in the last iteration
            #retrieve sth index of states_map from coordinates.dcd
            positions[s, :, :] = coords_traj.xyz[states_map_inds[s]]*10
            cell_lengths[s, :] = np.diag(ncdf.variables['box_vectors'][-1, r, :, :])*10

        inds_with_duplicates = sorted([ind for ind in range(ncdf.dimensions['replica'].size)] + inds)
        positions = positions[inds_with_duplicates, :, :]
        cell_lengths = cell_lengths[inds_with_duplicates, :]

        # Write new protocol to the trailblaze YAML
        yaml_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/protocol.yaml'
        yaml_o_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/protocol_{n_states_o}.yaml'
        shutil.copy(yaml_fn, yaml_o_fn)
        with open(yaml_fn, 'w') as f:
            f.write(yaml.dump(alchemical_paths, Dumper=YankDumper))
        
        # Write a new trailblaze states_map
        map_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/states_map.json'
        map_o_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/states_map_{n_states_o}.json'
        shutil.copy(map_fn, map_o_fn)
        with open(map_fn, 'w') as f:
            f.write(repr([ind for ind in range(n_states_n)]))

        # Generates a DCD file with duplicated coordinates
        DCD_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates.dcd'
        DCD_o_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates_{n_states_o}.dcd'
        shutil.copy(DCD_fn, DCD_o_fn)
        with DCDTrajectoryFile(DCD_fn, 'w') as f:
            f.write(positions, cell_lengths=cell_lengths, cell_angles=[[90., 90., 90.]]*len(inds_with_duplicates))
        
        # With this, previous netcdf can be renamed or deleted in order to allow
        #  a new one to be initialized from the new information in the setup directory

        #update progress.log 
        with open(prog_path, 'a') as f:
            f.write(f'STATE INSERTION:\n   {n_states_o} evaluated with accpetance rate: {acc_rate}\n   New number of states after insertion: {n_states_n}\n')
        return True

    def graph_states(self, phase='complex'):
        """ TODO: add _retreieve_states again
        """
        elecs, sters, rests, overlaps = self._retrieve_states(phase=phase)
        xs = np.arange(len(elecs))
        plt.clf()
        for yset in (elecs, sters, rests):
            plt.scatter(xs, yset)
        plt.scatter(xs[:-1]+0.5, overlaps)
        plt.legend(('Elec','Ster','Rest','OvrLp'),loc='upper right')
        plt.show()

    def _pose_insertion(self, list_of_pdb_fns, phase='complex'): #
        """ Replace the frames present in $YANK_STORE/experiments/trailblaze/$PHASE/coordinates.dcd
            With poses from a list of pdbs
            
            No matter the number of replicates, we would like to replace all with these poses"""
                
        assert phase == 'complex' or phase == 'solvent'
        
        yank_nc_fn = f'{self.yank_store}/experiments/{phase}.nc'
        ncdf = nc.Dataset(yank_nc_fn, 'r')
        if phase == 'complex':
            phase_num = 'phase1_path'
        elif phase == 'solvent':
            phase_num = 'phase2_path'
        coords_traj = md.load_dcd(f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates.dcd',\
                                 top = self.yaml_current['systems']['rec-lig'][phase_num][0])
        
        # Define positions array with appropriate dimensions
        poses_positions = np.zeros((ncdf.dimensions['replica'].size, \
                          coords_traj.xyz.shape[1], \
                          ncdf.dimensions['spatial'].size))
        # Cell Lengths
        cell_lengths = np.zeros((ncdf.dimensions['replica'].size, \
                                 ncdf.dimensions['spatial'].size))
        #Iterate over poses provided to build new coordinate array
        #Iterate from i until ncdf.dimensions['replica'].size, but if i>(len of pdb_list), wrap around
        for i in range(ncdf.dimensions['replica'].size): # replace every replica with one of our structures
            #Coordinate
            pdb_list_ind = i % len(list_of_pdb_fns)
            traj = md.load_pdb(list_of_pdb_fns[pdb_list_ind])
            #not_h20_atom_inds = traj.topology.select('not water')
            atom_inds = traj.topology.select('all')
            poses_positions[i, :, :] = traj.xyz[:, atom_inds, :]*10 #mdtraj in nm ##check after pose insertion, units could be issue
            #Cell
            s = ncdf.variables['states'][-1, i] # State index of replica r in the last iteration
            cell_lengths[s, :] = np.diag(ncdf.variables['box_vectors'][-1, i, :, :])*10.
        
        cell_angles = [[90., 90., 90.]]*ncdf.dimensions['replica'].size
        
        #Save this new array to coordinates.dcd
        DCD_fn   = f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates.dcd'
        DCD_o_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/coordinates_preinsert.dcd'
        shutil.copy(DCD_fn, DCD_o_fn)

        f = DCDTrajectoryFile(DCD_fn, 'w')
        f.write(poses_positions, cell_lengths=cell_lengths, cell_angles=cell_angles)
        f.close()
        
        #coordinates.dcd is no longer as large as it was before, so the indices in staes_map are not appropriate
        # Write a new trailblaze states_map
        map_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/states_map.json'
        map_o_fn = f'{self.yank_store}/experiments/trailblaze/{phase}/states_map_preinsert.json'
        shutil.copy(map_fn, map_o_fn)
        with open(map_fn, 'w') as f:
            f.write(repr([ind for ind in range(ncdf.dimensions['replica'].size)]))

        #update progress.log 
        with open(prog_path, 'a') as f:
            f.write(f'POSES INSERTED\n')
        return True

    
    # Insert states
    def state_evaluation(self):
        ##START JOB METHOD
        
        acc_rate = experiment.acceptance_rate(phase='complex')
        n_states_o = acc_rate.shape[0]+1
        while self._insert_states():
            print("states were inserted")
            #States were inserted, so the netcdf should be deleted
            #yank_nc_fn = f'{self.yank_store}/experiments/{phase}.nc'
            
            #Reset NetCDFs
            for fn in ['complex','solvent']:
                if os.path.isfile(f'{self.yank_store}/experiments/{fn}.nc'):
                    shutil.copy(f'{self.yank_store}/experiments/{fn}.nc',\
                                f'{self.yank_store}/experiments/{fn}_{n_states_o}.nc')
                    os.remove(f'{self.yank_store}/experiments/{fn}.nc')
                if os.path.isfile(f'{self.yank_store}/experiments/{fn}_checkpoint.nc'):
                    shutil.copy(f'{self.yank_store}/experiments/{fn}_checkpoint.nc',\
                                f'{self.yank_store}/experiments/{fn}_checkpoint_{n_states_o}.nc')
                    os.remove(f'{self.yank_store}/experiments/{fn}_checkpoint.nc')
            
            print(self.yaml_current)
            with open(prog_path, 'a') as f:
                f.write(f'CHECKING STATES\n')
            self.run_yank()

        with open(prog_path, 'a') as f:
            f.write(f'NUMBER OF STATES IS SUFFICIENT\n')        
        print(f'\nNUMBER OF STATES IS SUFFICIENT\n')

    ##insert poses
    def insert_poses_now(self, list_of_pdb_fns):
        acc_rate = experiment.acceptance_rate(phase='complex')
        n_states_o = acc_rate.shape[0]+1
        if self._pose_insertion(list_of_pdb_fns):
            print("poses were inserted")
            #Reset NetCDFs
            for fn in ['complex','solvent']:
                if os.path.isfile(f'{self.yank_store}/experiments/{fn}.nc'):
                    shutil.copy(f'{self.yank_store}/experiments/{fn}.nc',\
                                f'{self.yank_store}/experiments/{fn}_{n_states_o}.nc')
                    os.remove(f'{self.yank_store}/experiments/{fn}.nc')
                if os.path.isfile(f'{self.yank_store}/experiments/{fn}_checkpoint.nc'):
                    shutil.copy(f'{self.yank_store}/experiments/{fn}_checkpoint.nc',\
                                f'{self.yank_store}/experiments/{fn}_checkpoint_{n_states_o}.nc')
                    os.remove(f'{self.yank_store}/experiments/{fn}_checkpoint.nc')

            self.run_yank() 