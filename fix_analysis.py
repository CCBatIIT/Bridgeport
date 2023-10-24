import netCDF4 as ncdf
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

nc_path = '../../yank/8ef5/centroid_2/8ef5_1/experiments/complex_31.nc'
nc = ncdf.Dataset(nc_path)


pdb_path = '../../yank/8ef5/centroid_2/8ef5_2_1.pdb'
u = mda.Universe(pdb_path)
sele = u.select_atoms('not resname HOH')
sele.write('./bound.pdb')

u = mda.Universe('./bound.pdb')
sele = u.select_atoms('all')


n_frames = nc.variables['positions'].shape[0]
for frame in range(3):
	coords = nc.variables['positions'][frame,0,:,:]
	u.load_new(coords, format=MemoryReader)
	lig_sele = u.select_atoms('resname UNL')
	print(lig_sele.positions)
	lig_sele.write(f'./test_{frame}.pdb')

