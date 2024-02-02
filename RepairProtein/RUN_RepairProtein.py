"""
Use the RepairProtein class to repair any given .pdb that is in the directory. 
This script is contingent on all the target proteins having the same template sequence. 

USAGE:
------
>python RUN_RepairProtein.py {OPTIONS}

Required Arguments:
    -i --input_dir: directory where .pdb files of target proteins are found
    -o --output_dir: directory where repaired .pdb files will be added
    -f --fasta: path to .fasta file, which will serve as the template sequence to repair the target proteins

Optional Arguments:
    --tails: if called, will add tail residues to N and C termini.

"""

# Imports
import os, sys, argparse
from RepairProtein import RepairProtein

# Arguments
parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input_dir', action='store', required=True, help='directory where .pdb files of target proteins are found', default=None)
parser.add_argument('-o', '--output_dir', action='store', required=True, help='directory where repaired .pdb files will be added', default=None)
parser.add_argument('-f', '--fasta', action='store', required=True, help='path to .fasta file, which will serve as the template sequence to repair the target proteins', default=None)
parser.add_argument('--tails', action='store_true', help='if called, will add tail residues to N and C termini.', default=False)

args = parser.parse_args()
print('!!!'+str(args.tails))
# Input files 
prot_dir = args.input_dir
fasta_fn = args.fasta

# Make intermediate directory 
working_dir = os.getcwd() + '/'
int_dir = working_dir + 'modeller_intermediates'
if not os.path.exists(int_dir):
    os.mkdir(int_dir)

# Make output directory
prot_out_dir = args.output_dir
if not os.path.exists(prot_out_dir):
    os.mkdir(prot_out_dir)

# Iterate through input files
for pdb in os.listdir(prot_dir):
    rp = RepairProtein(pdb_fn=prot_dir + '/' + pdb,
                    fasta_fn=fasta_fn,
                    working_dir=int_dir)
    rp.run(pdb_out_fn=prot_out_dir + '/' + pdb, tails=args.tails)


