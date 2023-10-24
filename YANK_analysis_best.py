import yank
from yank.reports import notebook
from yank.analyze import ExperimentAnalyzer
import openmm as mm
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.coordinates.DCD import DCDWriter
from MDAnalysis.analysis.align import alignto
import netCDF4 as ncdf
import sys, os
import mdtraj as md
import os, pickle
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Arguments 
drug = sys.argv[1]
centroid_no = sys.argv[2]
reset = sys.argv[3]
print(f"ANALYSIS FOR {drug} centroid {centroid_no}, reset? = {reset}")


"""
INPUT
""" 
# Directories
working_dir = f'../../yank/{drug}/centroid_{centroid_no}/'
exp_dirs = [f'{working_dir}/{drug}_{i}/experiments/' for i in range(1,4)]
analysis_dir = f'{working_dir}/analysis'

# If reset, delete previous analysis
if reset:
  os.system(f'rm -r {analysis_dir}')
if not os.path.exists(analysis_dir):
  os.mkdir(analysis_dir)

# Files
complex_pdb = f'{working_dir}/{drug}_{centroid_no}_1.pdb'
ref_pdb = f'./CRYO/{drug}.pdb'
complex_ncs = [f'{exp_dirs[i]}/complex.nc' for i in range(len(exp_dirs))]

# # Selection String for MDA
# if drug == '8ef5':
#   cryo_sele_str = 'protein and resid 73:86 88:193 195:349 and name CA'
#   bound_sele_str = 'protein and resid 71:347 and name CA'

# Optional Settings
decorrelation_threshold = 0.1
mixing_cutoff = 0.05
mixing_warning_threshold = 0.45
phase_stacked_replica_plots = False

"""
MDAnalysis set up
"""

# Create complex w/out water pdb (YANK does not save HOH coords), and lig only pdb
no_hoh_pdb = f'{analysis_dir}/complex_no_water.pdb'
lig_pdb = f'{analysis_dir}/lig_only.pdb'
complex_u = mda.Universe(complex_pdb)
no_hoh_sele = complex_u.select_atoms('not resname HOH')
no_hoh_sele.write(no_hoh_pdb)
lig_sele = complex_u.select_atoms('resname UNL')
lig_sele.write(lig_pdb)

# Reload no_hoh_pdb as new mda.Universe
no_hoh_u = mda.Universe(no_hoh_pdb)
no_hoh_sele = no_hoh_u.select_atoms('all')

# Rewrite no_hoh_pdb with Amber atom types
bound_pdb = f'{analysis_dir}/bound.pdb'
remarks = []
remarks.append('REMARK    <-- AMBER ATOM TYPES ')
for n in range(0,len(no_hoh_sele.types),20):
    remarks.append('REMARK    ' +  ' '.join([f'{t:2s}' for t in no_hoh_sele.types[n:n+20]]))
remarks.append('REMARK    AMBER ATOM TYPES -->')
remarks = '\n' + '\n'.join(remarks)
no_hoh_sele.write(bound_pdb, remarks=remarks)

# Reload lig_pdb as new mda.Universe
lig_u = mda.Universe(lig_pdb)
lig_sele = lig_u.select_atoms('all')

# Find necessary information to correct ncdf reading. (NA/CL get replaced with ligand atom posititions for some reason)
na_first_ind = np.where(no_hoh_sele.atoms.names == 'NA')[0][0]
n_lig_atoms = lig_sele.n_atoms


"""
Extract from ncdf
"""
# Iterate through replicates
for i in range(len(exp_dirs)):
   
    # Get equilibration data
    print('\n\n Calculating Equilibration...\n\n')
    report = notebook.HealthReportData(exp_dirs[i])
    report.report_version()
    report.get_general_simulation_data()
    equil_report = report.get_equilibration_data()
    equil_iters = equil_report['complex']['count_total_equilibration_samples']
    print('No. of equilibration iterations: ' + str(equil_iters))

    # Extract information from ncdf
    nc = ncdf.Dataset(complex_ncs[i])
    n_iters, n_states, n_atoms, _ = nc.variables['positions'].shape
    print(f'{n_iters} found for {n_states} states with {n_atoms} atoms')

    # Determine replica indices of bound state (0)
    replica_ind = \
    [list(nc.variables['states'][iteration,:]).index(0) \
        for iteration in range(0,n_iters,1)]
    print('replica indicies of bound state: ' + str(replica_ind))

    # Align and store ligand coordinates as .dcd
    prot_resids = no_hoh_u.select_atoms('protein').residues.resids
    prot_resids_str = ' '.join(str(resid) for resid in prot_resids)
    bound_sele_str = f'protein and name CA and resid {prot_resids_str}'
    ref_resids_str = ' '.join(str(resid+2) for resid in prot_resids)
    cryo_sele_str = f'protein and name CA and resid {ref_resids_str}'

    lig_dcd = f'{analysis_dir}/lig_{i+1}.dcd'
    writer = DCDWriter(lig_dcd, lig_sele.n_atoms)
    ref_u = mda.Universe(ref_pdb)
    for frame in range(equil_iters+1, len(replica_ind)-2):
       
       # Load coords into no_hoh_u
       coords = nc.variables['positions'][frame, replica_ind[frame], :, :]*10.0
       no_hoh_u.load_new(coords, format=MemoryReader)

       #Align to reference
       alignto(no_hoh_u, ref_u, select={'mobile': bound_sele_str, 'reference': cryo_sele_str})

       # Get actual lig coords
       lig_coords = coords[na_first_ind:na_first_ind+n_lig_atoms]
       lig_u.load_new(lig_coords, format=MemoryReader)

       # Write out ligand coordinates
       writer.write(lig_sele)

    print(f'{lig_dcd} written')

"""
Combine ligand trajectories into one
"""
super_traj_dcd = f'{analysis_dir}/lig_super_traj.dcd'
lig_dcds = [f'{analysis_dir}/lig_{i}.dcd' for i in range(1,4)]
trajs = []
for dcd in lig_dcds:
   trajs.append(md.load(dcd, top=lig_pdb))
super_traj = md.join(trajs)
super_traj.save_dcd(super_traj_dcd)


# # A class to perform RMSD calculations
# import numpy as np
# from munkres import Munkres
# from MDAnalysis.analysis.base import AnalysisBase

# class HungarianRMSD(AnalysisBase):
#   """Hungarian symmetry-corrected root mean square deviation
  
#   HungarianRMSD(atomgroup, ref_conf)
  
#   Arguments
#   ---------
#   atomgroup : AtomGroup
#     AtomGroup
#   ref_conf : numpy.array (Nx3)
#     refrence configuration or None to use AtomGroup configuration

#   See http://dock.compbio.ucsf.edu/DOCK_6/dock6_manual.htm#LigandRMSD
#   """
#   def __init__(self, atomgroup, ref_conf=None, atom_types=None, **kwargs):
#     """
#     sel is an AtomGroup object MDAnalysis.core.groups
#     ref_conf is the default reference configuration, an Nx3 numpy array
#     atom_types is a list of strings
#     """
#     super(HungarianRMSD, self).__init__(atomgroup.universe.trajectory, **kwargs)
    
#     self._ag = atomgroup
#     if ref_conf is not None:
#       self._ref_conf = ref_conf
#     else:
#       self._ref_conf = np.copy(self._ag.positions)
      
#     if atom_types is not None:
#       self._atom_types = atom_types
#     else:
#       self._atom_types = self._ag.types
        
#     self.atom_sets_to_compare = []
#     atom_indices = np.array(range(len(self._atom_types)))
#     for t in set(self._atom_types):
#       indices_t = (self._atom_types == t)
#       self.atom_sets_to_compare.append((sum(indices_t), atom_indices[indices_t]))    
    
#     self.munkres = Munkres()
    
#   def set_ref_conf(self, ref_conf):
#     """
#     Sets a new reference configuration
#     """
#     self._ref_conf = ref_conf

#   def _prepare(self):
#     self.rmsds = []

#   def _single_frame(self):
#     ssd = 0.
#     conf = self._ag.positions
#     for (nelements, atom_set) in self.atom_sets_to_compare:
#       if nelements == 1:
#         j = atom_set[0]
#         ssd += np.sum(np.square(conf[j, :] - self._ref_conf[j, :]))
#       else:
#         cost_matrix = np.array([[\
#           np.sum(np.square(conf[atom_set[j],:]-self._ref_conf[atom_set[k],:])) \
#             for j in range(nelements)] \
#               for k in range(nelements)])
#         path = self.munkres.compute(cost_matrix)
#         ssd += np.sum([np.sum(np.square(\
#           conf[atom_set[j],:]-self._ref_conf[atom_set[k],:])) for (j,k) in path])
#     self.rmsds.append(np.sqrt(ssd / self._ag.n_atoms))

# """
# Calculate RMSDs to CRYO
# """
# # Load with MDA
# u = mda.Universe(lig_pdb, super_traj_dcd)
# sel_ligand = u.select_atoms('all')

# # Read atom types 
# bound_pdb_reader = mda.coordinates.PDB.PDBReader(bound_pdb)
# AMBER_atom_types = []
# for line in bound_pdb_reader.remarks[1:-1]:
#     AMBER_atom_types += line.split()
# ligand_AMBER_atom_types = np.array(AMBER_atom_types)[sel_ligand.indices]


# # Calculate the RMSD matrix using the Hungarian RMSD class
# if not os.path.isfile(f'{analysis_dir}/rmsds.pkl'):
#     print("CALCULATING RMSDS")
#     calcHungarianRMSD = HungarianRMSD(sel_ligand, atom_types = ligand_AMBER_atom_types)
#     rmsds = []
#     u.trajectory.rewind()
#     for frame in range(u.trajectory.n_frames):
#         calcHungarianRMSD.set_ref_conf(u.trajectory[frame].positions[sel_ligand.indices,:])
#         calcHungarianRMSD.start = frame + 1
#         calcHungarianRMSD.run()
#         print(f'Calculated {len(calcHungarianRMSD.rmsds)} rmsds relative to frame {frame}')
#         rmsds += calcHungarianRMSD.rmsds[frame + 1:]
#         u.trajectory.next()
#     rmsds = np.array(rmsds)    
#     F = open(f'{analysis_dir}/rmsds.pkl','wb')
#     pickle.dump(rmsds, F)
#     F.close()
#     print(f"\n\n rmsds.pkl MADE \n\n")

# else:
#     F = open(f'{analysis_dir}/rmsds.pkl','rb')
#     rmsds = pickle.load(F)
#     F.close()
#     print("\n\n rmsds.pkl READ \n\n")

# """
# Clustering
# """
# print('\n\nClustering ligand RMSDs...\n\n')

# # Linkage
# rmsds_sq = squareform(rmsds)
# Z = linkage(rmsds, method='complete')

# # Cluster
# assignments = np.array(fcluster(Z, 3, criterion='distance'))

# # Reindexes the assignments in order of appearance
# new_index = 0
# mapping_to_new_index = {}
# for assignment in assignments:
#     if not assignment in mapping_to_new_index.keys():
#         mapping_to_new_index[assignment] = new_index
#         new_index += 1
# assignments = [mapping_to_new_index[a] for a in assignments]

# # Create array of cluster sizes and order 
# no_clusters = np.max(assignments)+1
# cluster_counts = [0 for i in range(no_clusters)]
# for i in range(len(assignments)):
#     assignment = assignments[i]
#     cluster_counts[assignment] += 1
# cluster_counts.sort(reverse=True)
# print("CLUSTER COUNTS: " + cluster_counts)

# # Create a list of how many frames are in a cluster and the representative frame from each cluster
# counts_and_medoids = []
# for n in range(max(assignments) + 1):
#     inds = [i for i in range(len(assignments)) if assignments[i] == n]
#     rmsds_n = rmsds_sq[inds][:, inds]
#     counts_and_medoids.append((len(inds), inds[np.argmin(np.mean(rmsds_n, 0))]))
# counts_and_medoids.sort(reverse=True)
# print(counts_and_medoids)
# with open(f'{analysis_dir}/cluster_sizes.txt', 'w') as f:
#     f.write(f'CLUSTER COUNTS AND MEDIOIDS OF {drug}\n')
#     for cluster in counts_and_medoids:
#         f.write(str(cluster))

# # Writes the representative frames to a new dcd file in decreasing order of population
# poses_dcd = f'{analysis_dir}/poses.dcd'
# writer = mda.Writer(poses_dcd, lig_sele.n_atoms)
# complex_out = mda.Universe(lig_pdb)

# for (count, medoid) in counts_and_medoids:
#   if count<20:
#     continue
#   positions = u.trajectory[medoid].positions
#   complex_out.load_new(positions, format=MemoryReader)
#   writer.write(complex_out)
