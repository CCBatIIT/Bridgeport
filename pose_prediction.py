#!/usr/bin/env python
import textwrap
#import MDAnalysis as mda
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


#INPUT FILES
#DAVID CHANGE ME
name= sys.argv[1] #become an arg
data_dir = f'/ocean/projects/bio230003p/dcooper/yank/{name}/'
run = sys.argv[2]
job_type = sys.argv[3]


#MAYBE CHANGE ME
if os.path.exists(data_dir + f'{name}_crys.pdb'):
    crystal_fn = data_dir + f'{name}_crys.pdb'
ligand_resname = 'resname UNL'
yank_output_dir = data_dir + f'{name}_{run}'
yaml_file = data_dir + f'{name}_{run}.yaml' 
prog_path = os.path.join(data_dir,f'progress_{run}.log')



#DON"T NEED TO CHANGE ME
complex_fns = data_dir + f'{name}_1.prmtop', data_dir + f'{name}_1.inpcrd'
solvent_fns = data_dir + f'solvent_{name}.prmtop', data_dir + f'solvent_{name}.inpcrd'
try:
    pose_fns = [crystal_fn] + sorted(glob.glob(data_dir + f'{name}_*.pdb'))
    print('crystal pose added to poses_fns as ind 0')
except:
    pose_fns = sorted(glob.glob(data_dir + f'{name}_*.pdb'))




#DAVID DONT CHANGE MUCH BELOW THIS LINE ___________

#write a yaml file
def write_the_yaml(complex_fns, solvent_fns, ligand_string, out_dir):
    yaml_contents = f"""---
experiments:
  protocol: absolute-binding
  restraint:
    type: FlatBottom
    restrained_receptor_atoms: (resname TYR and resid 261) or (resname ASP and resid 49) or (resname HIS and resid 232) or (resname TYR and resid 83)
    restrained_ligand_atoms: all
    spring_constant: 10.0*kilocalories_per_mole/(angstrom**2)
    well_radius: 8.0*angstroms
  system: rec-lig
options:
  default_nsteps_per_iteration: 500
  default_number_of_iterations: 50
  default_timestep: 1.0*femtosecond
  minimize: no
  number_of_equilibration_iterations: 0
  output_dir: {out_dir}
  platform: fastest
  pressure: 1.0*atmosphere
  resume_simulation: yes
  temperature: 300*kelvin
  verbose: yes
protocols:
  absolute-binding:
    complex:
      alchemical_path: auto
      trailblazer_options:
        bidirectional_redistribution: yes
        constrain_receptor: false
        distance_tolerance: 0.05
        n_equilibration_iterations: 0
        n_samples_per_state: 100
        reversed_direction: yes
        thermodynamic_distance: 1
    solvent:
      alchemical_path: auto
      trailblazer_options:
        bidirectional_redistribution: yes
        constrain_receptor: false
        distance_tolerance: 0.05
        n_equilibration_iterations: 0
        n_samples_per_state: 100
        reversed_direction: yes
        thermodynamic_distance: 1
solvents:
  PME:
    nonbonded_cutoff: 8.0*angstroms
    nonbonded_method: PME
systems:
  rec-lig:
    ligand_dsl: {ligand_string}
    phase1_path:
    - {complex_fns[0]}
    - {complex_fns[1]}
    phase2_path:
    - {solvent_fns[0]}
    - {solvent_fns[1]}
    solvent: PME"""
    return yaml_contents


#if not os.path.isfile(yaml_file):
with open(yaml_file, 'w') as f:
    f.write(write_the_yaml(complex_fns, solvent_fns, ligand_resname, yank_output_dir))


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
    
 

        

##For starting job, run here 
if job_type == 'blaze':
    print(f"\n\n\n\n\n\{yaml_file}\n\n\n\n\n\n")
    experiment = YankWrapper(yaml_file)

elif job_type == 'start':
    experiment = YankWrapper(yaml_file)#this will be skipped if this has already been ran 
    n_states = len(experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'])
    experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'] = [1.0 for i in range(n_states)]
    restraints = experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints']
    with open(prog_path, 'a') as f:
        f.write(f'Restraints now set to: \n {restraints}\n')
    experiment.state_evaluation()
    experiment.insert_poses_now(pose_fns)
    experiment.state_evaluation()
    print('SYSTEM READY FOR PRODUCTION RUN')

elif job_type == 'restart1':
    current_yaml_path = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}/experiments/experiments.yaml'
    experiment = YankWrapper(current_yaml_path)
    experiment.state_evaluation()
    experiment.insert_poses_now(pose_fns)
    n_states = len(experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'])
    experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'] = [1.0 for i in range(n_states)]
    restraints = experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints']
    with open(prog_path, 'a') as f:
        f.write(f'Restraints now set to: \n {restraints}\n')
    experiment.state_evaluation()

elif job_type == 'restart2':
    current_yaml_path = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}/experiments/experiments.yaml'
    experiment = YankWrapper(current_yaml_path)
    n_states = len(experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'])
    experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'] = [1.0 for i in range(n_states)]
    restraints = experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints']
    with open(prog_path, 'a') as f:
        f.write(f'Restraints now set to: \n {restraints}\n')
    experiment.state_evaluation()
        

elif job_type == 'production':
    #initialize experiment
    current_yaml_path = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}/experiments/experiments.yaml'
    experiment = YankWrapper(current_yaml_path)
    acc_rate = experiment.acceptance_rate(phase='complex')
    print(f"ACCEPTANCE LOWEST: " + str(np.min(acc_rate)))
    n_states_o = acc_rate.shape[0]+1

    for fn in ['complex','solvent']:
        if os.path.isfile(f'{experiment.yank_store}/experiments/{fn}.nc'):
            shutil.copy(f'{experiment.yank_store}/experiments/{fn}.nc',\
                        f'{experiment.yank_store}/experiments/{fn}_{n_states_o}.nc')
            os.remove(f'{experiment.yank_store}/experiments/{fn}.nc')
        if os.path.isfile(f'{experiment.yank_store}/experiments/{fn}_checkpoint.nc'):
            shutil.copy(f'{experiment.yank_store}/experiments/{fn}_checkpoint.nc',\
                        f'{experiment.yank_store}/experiments/{fn}_checkpoint_{n_states_o}.nc')
            os.remove(f'{experiment.yank_store}/experiments/{fn}_checkpoint.nc')

    experiment.yaml_current['options']['number_of_equilibration_iterations'] = 0
    experiment.yaml_current['options']['default_number_of_iterations'] = 50
    experiment.yaml_current['options']['default_timestep'] = '4.0*femtosecond'
    experiment.yaml_current['options']['default_nsteps_per_iteration'] = 500
    experiment.yaml_current['options']['hydrogen_mass'] = '2.0 * amu'
    
    n_states = len(experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'])
    experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'] = [1.0 for i in range(n_states)]


    num_of_iters = experiment.yaml_current['options']['default_number_of_iterations']
    print(f'number of iterations set to {num_of_iters}')
    
    experiment.run_yank()

elif job_type == 'production_restart':
    current_yaml_path = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}/experiments/experiments.yaml'
    experiment = YankWrapper(current_yaml_path)
    acc_rate = experiment.acceptance_rate(phase='complex')
    experiment.state_evaluation()


    #get the number of iterations that have already been ran 
    # ncdf_path = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}/experiments/complex.nc'
    # ncdf = nc.Dataset(ncdf_path)
    iters_ran = int(experiment.yaml_current['options']['default_number_of_iterations'])

    # set target iters with arg
    target_iters = int(sys.argv[4])
    
    #report target iters and iters ran and min acc rate
    print(f'target: {target_iters} \niters_ran: {iters_ran}')
    print(f'min acc rate: {np.min(acc_rate)}')

    while (iters_ran < target_iters):
        new_experiment = YankWrapper(current_yaml_path)
        n_states = len(experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'])
        new_experiment.yaml_current['protocols']['absolute-binding']['complex']['alchemical_path']['lambda_restraints'] = [1.0 for i in range(n_states)]
        new_experiment.yaml_current['options']['default_timestep'] = '4.0*femtosecond'
        new_experiment.yaml_current['options']['default_nsteps_per_iteration'] = 500
        new_experiment.yaml_current['options']['hydrogen_mass'] = '2.0 * amu'
        new_experiment.yaml_current['options']['default_number_of_iterations'] = iters_ran + 50
        new_experiment.yaml_current['options']['output_dir'] = f'/ocean/projects/bio230003p/dcooper/yank/{name}/{name}_{run}'
        new_experiment.run_yank()
        acc_rate = new_experiment.acceptance_rate(phase='complex')
        if(np.min(acc_rate) < 0.45):
            new_experiment.yaml_current['options']['default_number_of_iterations'] = 50
            new_experiment.state_evaluation()
            iters_ran = 50
        else:
            with open(prog_path, 'a') as f:
                f.write(f'PRODUCTION iteration: {iters_ran}\n')
                f.write(f'PRODUCTION lowest acceptance rate of {np.min(acc_rate)}\n')
            print(f" NUMBER OF ITERATIONS COMPLETED IS {iters_ran+50}")
            iters_ran += 50
    

    print('PRODUCTION RUN COMPLETE')

    acc_rate = experiment.acceptance_rate(phase='complex')
    with open(prog_path, 'a') as f:
        f.write(f'PRODUCTION lowest acceptance rate of {np.min(acc_rate)}\n')
        f.write(f'PRODUCTION accpetance rates: {acc_rate}\n')


