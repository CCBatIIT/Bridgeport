"""
Equilibrate usage

python EQUIL_MEMBRANE.py input_directory name output_directory

	input_directory: absolute path to directory where input OpenMM files are
	name: identifier for OpenMM.files without the extension 
	output_directory: absolute path to directory where subdirectory with equilibration steps will be saved

"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import os, sys
sys.path.append('MotorRow')
from MotorRow import MotorRow

# Input files
input_dir = sys.argv[1]
input_name = sys.argv[2]
input_xml = os.path.join(input_dir, input_name + '.xml')
input_pdb = os.path.join(input_dir, input_name + '.pdb')

# Output files
output_dir = sys.argv[3]
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
output_dir = os.path.join(sys.argv[3], input_name)

# Run equilibration
final_xml, final_pdb = MotorRow(input_xml, input_pdb, output_dir).main(input_pdb)

# Print final files
print('Final pdb saved to', final_pdb)
print('Final xml saved to', final_xml)
