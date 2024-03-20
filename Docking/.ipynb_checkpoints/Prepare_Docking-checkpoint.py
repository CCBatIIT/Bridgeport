import os
import argparse

"""
Prepares receptor and ligands for docking with conversion to .pdbqt format via mgltools

Conda envs:
-----------
    Only mgltools 1.5.7 installed in environment

"""

# Arguments 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--receptor_dir')
parser.add_argument('-l', '--ligand_dir')
parser.add_argument('-or', '--out_receptor_dir')
parser.add_argument('-ol', '--out_ligand_dir')
args = vars(parser.parse_args())


in_receptor_dir = args['receptor_dir']
in_ligand_dir = args['ligand_dir']
out_receptor_dir = args['out_receptor_dir']
out_ligand_dir = args['out_ligand_dir']


# Assertions
if in_receptor_dir != None:
    assert out_receptor_dir != None, 'If -r --receptor_dir argument is provided, -or --out_receptor_dir must be provided.'
    if not os.path.exists(out_receptor_dir):
        os.mkdir(out_receptor_dir)

if in_ligand_dir != None:
    assert out_ligand_dir != None, 'If -l --ligand_dir argument is provided, -or --out_ligand_dir must be provided.'
    if not os.path.exists(out_ligand_dir):
        os.mkdir(out_ligand_dir)


# Prepare proteins 
if in_receptor_dir != None:
    for file in os.listdir(in_receptor_dir):
        os.system('prepare_receptor4.py -r ' + in_receptor_dir + '/' + file  + ' -o ' + out_receptor_dir + '/' + file.split(".")[0] + '.pdbqt -U nphs_lps')

# Prepare ligands
if in_ligand_dir != None:
    os.chdir(in_ligand_dir)
    for file in os.listdir(os.getcwd()):

        os.system('prepare_ligand4.py -l ' + in_ligand_dir  + file + ' -o ' + out_ligand_dir + '/' + file.split('.')[0] + '.pdbqt -U nphs_lps')