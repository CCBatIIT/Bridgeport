import yank
from yank.reports import notebook
from yank.analyze import ExperimentAnalyzer
import openmm as mm
import matplotlib.pyplot as plt
import seaborn
import MDAnalysis as mda
import sys, os
import mdtraj as md
import os, pickle



# Input options 
analogue = sys.argv[1]
run = sys.argv[2]
sel_str_align = 'protein and name CA'
sel_str_store = 'not resname HOH'
stride = 1
print(f"ANALYSIS FOR {analogue} RUN: {run}")


# Input files
working_dir = f'./yank/{analogue}'
store_dir = f'{working_dir}/{analogue}_{run}/experiments'
analysis_store_dir = f'{working_dir}/analysis_{run}'
delete_analysis_store_dir = sys.argv[3]
print("OPTION")
print(delete_analysis_store_dir)
if delete_analysis_store_dir == 'True' or delete_analysis_store_dir == 'true' or delete_analysis_store_dir == 'yes':
  os.system(f'rm -r {analysis_store_dir}')
if not os.path.exists(analysis_store_dir):
   os.mkdir(analysis_store_dir)
complex_nc = f'{store_dir}/complex.nc'
complex_prmtop = f'{working_dir}/{analogue}_1.prmtop'
complex_inpcrd = f'{working_dir}/{analogue}_1.inpcrd'
#output
bound_dcd_fn = f'{analysis_store_dir}/bound_{run}.dcd' 
print(bound_dcd_fn)

# Optional Settings
decorrelation_threshold = 0.1
mixing_cutoff = 0.05
mixing_warning_threshold = 0.45
phase_stacked_replica_plots = False

# MDA universe set up
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align

if not os.path.exists(f'{analysis_store_dir}/bound.pdb'):
  print("MAKING bound.pdb")
  ref = mda.Universe(complex_prmtop, complex_inpcrd)
  complex = mda.Universe(complex_prmtop, complex_inpcrd)
  sel_protein = complex.select_atoms("protein")
  sel_store = complex.select_atoms(sel_str_store)
  sel_store.write(f'{analysis_store_dir}/{analogue}_complex_no_water.pdb')
  complex_no_water = mda.Universe(f'{analysis_store_dir}/{analogue}_complex_no_water.pdb')
  sel_store_no_water = complex_no_water.select_atoms('all')

  print(f"Number of atoms: {sel_protein.n_atoms} in protein, " + \
        f"{sel_store.n_atoms} selected")

  # Write template pdb of just protein and ligand
  remarks = []
  remarks.append('REMARK    <-- AMBER ATOM TYPES ')
  for n in range(0,len(sel_store.types),20):
    remarks.append('REMARK    ' +  ' '.join([f'{t:2s}' for t in sel_store.types[n:n+20]]))
  remarks.append('REMARK    AMBER ATOM TYPES -->')
  remarks = '\n' + '\n'.join(remarks)
  sel_store.write(f'{analysis_store_dir}/bound.pdb', remarks=remarks)

  print("\n\nbound.pdb written \n\n")

# Extract from .nc
import netCDF4 as netcdf
from MDAnalysis import coordinates
from MDAnalysis.coordinates import DCD

if not os.path.exists(bound_dcd_fn):
  print("\n\nWRITING TRAJECTORIES FROM .nc FILES")
  # Get equilibration iterations 
  # Make YANK report
  report = notebook.HealthReportData(store_dir)
  report.report_version()
  report.get_general_simulation_data()
  print('\n\n##################  EQUILIBRATION DATA  ##################\n\n')
  equil_report = report.get_equilibration_data()
  print(report.get_equilibration_data())
  equil_iters = equil_report["complex"]["count_total_equilibration_samples"]
  print(f"\nNUMBER OF EQUILIBRATION ITERATIONS: {equil_iters}\n")

  #Load nc file from YANK
  nc = netcdf.Dataset(complex_nc)
  n_iterations = nc.variables['positions'].shape[0]
  n_states = nc.variables['states'].shape[1]
  n_atoms = nc.variables['positions'].shape[2]
  print(f'{n_iterations} iterations (will skip first), {n_states} states, {n_atoms} atoms')

  # Determine replica indices that belong to the bound state, which is state 0
  replica_ind = \
  [list(nc.variables['states'][iteration,:]).index(0) \
      for iteration in range(0,n_iterations,stride)]
  print('replica indicies of bound state: ' + str(replica_ind))

  # Align and store snapshots, skipping first frame
  print('These are RMSDs before and after alignment:')
  writer = mda.coordinates.DCD.DCDWriter(bound_dcd_fn, sel_store.n_atoms)
  for frame in range(equil_iters+1,len(replica_ind)-2):
      coords = nc.variables['positions'][frame*stride,replica_ind[frame],:,:]*10.0
      complex_no_water.load_new(coords, format=MemoryReader)
      print(align.alignto(complex_no_water, ref, select=sel_str_align))
      writer.write(sel_store_no_water)
print("\n\nTRAJECTORIES WRITTEN\n\n")

# Ligand selection
poses_dcd = f'{analysis_store_dir}/poses_{run}.dcd'
sel_str_ligand = 'resname UNL and not name H*'

# A class to perform RMSD calculations
import numpy as np
from munkres import Munkres
from MDAnalysis.analysis.base import AnalysisBase

class HungarianRMSD(AnalysisBase):
  """Hungarian symmetry-corrected root mean square deviation
  
  HungarianRMSD(atomgroup, ref_conf)
  
  Arguments
  ---------
  atomgroup : AtomGroup
    AtomGroup
  ref_conf : numpy.array (Nx3)
    refrence configuration or None to use AtomGroup configuration

  See http://dock.compbio.ucsf.edu/DOCK_6/dock6_manual.htm#LigandRMSD
  """
  def __init__(self, atomgroup, ref_conf=None, atom_types=None, **kwargs):
    """
    sel is an AtomGroup object MDAnalysis.core.groups
    ref_conf is the default reference configuration, an Nx3 numpy array
    atom_types is a list of strings
    """
    super(HungarianRMSD, self).__init__(atomgroup.universe.trajectory, **kwargs)
    
    self._ag = atomgroup
    if ref_conf is not None:
      self._ref_conf = ref_conf
    else:
      self._ref_conf = np.copy(self._ag.positions)
      
    if atom_types is not None:
      self._atom_types = atom_types
    else:
      self._atom_types = self._ag.types
        
    self.atom_sets_to_compare = []
    atom_indices = np.array(range(len(self._atom_types)))
    for t in set(self._atom_types):
      indices_t = (self._atom_types == t)
      self.atom_sets_to_compare.append((sum(indices_t), atom_indices[indices_t]))    
    
    self.munkres = Munkres()
    
  def set_ref_conf(self, ref_conf):
    """
    Sets a new reference configuration
    """
    self._ref_conf = ref_conf

  def _prepare(self):
    self.rmsds = []

  def _single_frame(self):
    ssd = 0.
    conf = self._ag.positions
    for (nelements, atom_set) in self.atom_sets_to_compare:
      if nelements == 1:
        j = atom_set[0]
        ssd += np.sum(np.square(conf[j, :] - self._ref_conf[j, :]))
      else:
        cost_matrix = np.array([[\
          np.sum(np.square(conf[atom_set[j],:]-self._ref_conf[atom_set[k],:])) \
            for j in range(nelements)] \
              for k in range(nelements)])
        path = self.munkres.compute(cost_matrix)
        ssd += np.sum([np.sum(np.square(\
          conf[atom_set[j],:]-self._ref_conf[atom_set[k],:])) for (j,k) in path])
    self.rmsds.append(np.sqrt(ssd / self._ag.n_atoms))


# # Combine three trajectories into one 
# print("\n\n COMBINING TRAJECTORIES \n\n")
# trajs = [md.load(bound_dcd_fn, top=f'{analysis_store_dir}/bound.pdb')]
# combined = md.join(trajs=trajs)
# combined.save(f'{analysis_store_dir}/bound_combined.dcd')
# print("\n\n TRAJECTORIES COMBINED \n\n")

# Load the complex
print("\n\n LOADING BOUND COMPLEX \n\n")
bound = mda.Universe(f'{analysis_store_dir}/bound.pdb', bound_dcd_fn)
sel_ligand = bound.select_atoms(sel_str_ligand)

# Read atom types 
bound_pdb_reader = mda.coordinates.PDB.PDBReader(f'{analysis_store_dir}/bound.pdb')
AMBER_atom_types = []
for line in bound_pdb_reader.remarks[1:-1]:
    AMBER_atom_types += line.split()
ligand_AMBER_atom_types = np.array(AMBER_atom_types)[sel_ligand.indices]
print("\n\n BOUND COMPLEX  LOADED\n\n")


delete_pkl = sys.argv[3]
if delete_pkl == 'True' or delete_pkl == 'true' or delete_pkl == 'yes':
   os.system(f'rm {analysis_store_dir}/rmsds.pkl')
   print("\n\n CURRENT .pkl REMOVED MAKING NEW ONE")

if not os.path.isfile(f'{analysis_store_dir}/rmsds_{run}.pkl'):
    # Calculate the RMSD matrix
    print("CALCULATING RMSDS")
    calcHungarianRMSD = HungarianRMSD(sel_ligand, atom_types = ligand_AMBER_atom_types)
    rmsds = []
    bound.trajectory.rewind()
    for frame in range(bound.trajectory.n_frames):
        calcHungarianRMSD.set_ref_conf(bound.trajectory[frame].positions[sel_ligand.indices,:])
        calcHungarianRMSD.start = frame + 1
        calcHungarianRMSD.run()
        print(f'Calculated {len(calcHungarianRMSD.rmsds)} rmsds relative to frame {frame}')
        rmsds += calcHungarianRMSD.rmsds[frame + 1:]
        bound.trajectory.next()
    rmsds = np.array(rmsds)    
    F = open(f'{analysis_store_dir}/rmsds_{run}.pkl','wb')
    pickle.dump(rmsds, F)
    F.close()
    print(f"\n\n rmsds_{run}.pkl MADE \n\n")

else:
    F = open(f'{analysis_store_dir}/rmsds_{run}.pkl','rb')
    rmsds = pickle.load(F)
    F.close()
    print("\n\n rmsds.pkl READ \n\n")


# Pairwise RMDS plot
print("\n\nCLUSTERING\n\n")

from scipy.spatial.distance import squareform
rmsds_sq = squareform(rmsds)

import scipy.cluster
Z = scipy.cluster.hierarchy.linkage(rmsds, method='complete')
assignments = np.array(\
  scipy.cluster.hierarchy.fcluster(Z, 2, criterion='distance'))

# Reindexes the assignments in order of appearance
new_index = 0
mapping_to_new_index = {}
for assignment in assignments:
  if not assignment in mapping_to_new_index.keys():
    mapping_to_new_index[assignment] = new_index
    new_index += 1
assignments = [mapping_to_new_index[a] for a in assignments]
print("THESE ARE THE ASSIGNMENTS")
print(assignments)


# Create array of cluster sizes and order 
no_clusters = np.max(assignments)+1
cluster_counts = [0 for i in range(no_clusters)]
for i in range(len(assignments)):
    assignment = assignments[i]
    cluster_counts[assignment] += 1
cluster_counts.sort(reverse=True)

print("CLUSTER COUNTS:")
print(cluster_counts)

# load combined trajectory
complex_no_water = mda.Universe(f'{analysis_store_dir}/{analogue}_complex_no_water.pdb', bound_dcd_fn)
sel_store_no_water = complex_no_water.select_atoms('all')

# Write traj files for each assignment >= 20 frames
from MDAnalysis.coordinates.DCD import DCDWriter

for cluster in range(len(cluster_counts)):
    print(f"cluster: {cluster} with size: {cluster_counts[cluster]}")
    if(cluster_counts[cluster] >= 20):
        writer = DCDWriter(f'{analysis_store_dir}/replicate_{run}_cluster_{cluster+1}.dcd',sel_store_no_water.n_atoms )
        iter_counter=0
        for iteration in assignments:
            complex_no_water.trajectory[iter_counter]
            if iteration == cluster:
                writer.write(sel_store_no_water)
            iter_counter += 1

# Create a list of how many frames are in a cluster and the representative frame from each cluster
counts_and_medoids = []
for n in range(max(assignments) + 1):
  inds = [i for i in range(len(assignments)) if assignments[i] == n]
  rmsds_n = rmsds_sq[inds][:, inds]
  counts_and_medoids.append((len(inds), inds[np.argmin(np.mean(rmsds_n, 0))]))
counts_and_medoids.sort(reverse=True)
print(counts_and_medoids)
with open(f'{analysis_store_dir}/replicate_{run}_cluster_sizes.txt', 'w') as f:
  f.write(f'CLUSTER COUNTS AND MEDIOIDS OF {analogue}\n')
  for cluster in counts_and_medoids:
    f.write(str(cluster))


# Writes the representative frames to a new dcd file in decreasing order of population
from MDAnalysis.coordinates.memory import MemoryReader
writer = mda.Writer(poses_dcd, complex_no_water.atoms.n_atoms)
complex_out = mda.Universe(f'{analysis_store_dir}/bound.pdb')

for (count, medoid) in counts_and_medoids:
  if count<20:
    continue
  positions = bound.trajectory[medoid].positions
  complex_out.load_new(positions, format=MemoryReader)
  writer.write(complex_out)




