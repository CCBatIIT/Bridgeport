import os
import argparse
from Docking import Docking
from typing import List

# List of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

# Arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--receptor_path', type=str, help='path to .pdbqt file of receptor', required=True)
parser.add_argument('-l', '--ligand_path', type=str, help='path to .pdbqt file of ligand', required=True)
parser.add_argument('-ol', '--ligand_out_dir', type=str, help='path to directory to store docked poses', default='./')
parser.add_argument('--config_dir', type=str, help='path to directory to store Autodock vina configuration files', default='./')
parser.add_argument('--box_center', type=list_of_ints, help='list of box-center positions: x-coord,y-coord,z-coord', default=[0, 0, 0])
parser.add_argument('--box_dim', type=list_of_ints, help='list of box dimensions (Angstrom): x-size,y-size,z-size', default=[15, 15, 15])
parser.add_argument('--n_poses', type=int, help='number of docked poses', default=20)
parser.add_argument('--exhaustiveness', type=int, help='level of exhaustiveness', default=8)
parser.add_argument('--min_rmsd', type=float, help='to save a new docked pose, the new pose must surpass the minimum rmsd of new pose compared to previous poses', default=1.0)
parser.add_argument('--compare', nargs=3, help='Requires 3 arguments delimited by a space. path to .pdb file with both receptor and ligand used for reference, chainid of protein to align to docking structure, and MDAnalysis selection string to parse the ligand from ref_pdb file. EX: --compare /path/to/ref.pdb R "selection string".')

args = vars(parser.parse_args())

rec_path = args['receptor_path']
lig_path = args['ligand_path']
ligand_out_dir = args['ligand_out_dir']
config_dir = args['config_dir']
box_center = args['box_center']
box_dim = args['box_dim']
n_poses = args['n_poses']
exhaustiveness = args['exhaustiveness']
min_rmsd = args['min_rmsd']
compare_str = args['compare']

# Assert paths exist
if os.path.exists(rec_path):
    print('')
else:
    raise FileNotFoundError("Must provide path to RECEPTOR.pdbqt")
if os.path.exists(lig_path):
    print('')
else:
    raise FileNotFoundError("Must provide path to LIGAND.pdbqt")

# Run Docking
docking = Docking(receptor_path=rec_path,
                  ligand_path=lig_path,
                  config_dir=config_dir)

docking.set_box(box_center=box_center,
                box_dim=box_dim)

docking.dock(lig_out_dir=ligand_out_dir,
             n_poses=n_poses,
             exhaustiveness=exhaustiveness,
             min_rmsd=min_rmsd)

# Run compare analysis, if specified
if compare_str != None:
    try:
        ref_pdb, ref_chainid, ref_lig_sele_str = compare_str
        print(ref_pdb, ref_chainid, ref_lig_sele_str)
    except:
        raise AssertionError('--compare Requires 3 arguments delimited by a space. path to .pdb file with both receptor and ligand used for reference, chainid of protein to align to docking structure, and MDAnalysis selection string to parse the ligand from ref_pdb file. EX: --compare /path/to/ref.pdb R "selection string".')

    docking.compare(ref_pdb=ref_pdb,
                    ref_chainid=ref_chainid,
                    ref_lig_sele_str=ref_lig_sele_str)

    

