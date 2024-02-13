from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import os, sys
from MotorRow import MotorRow

#Usage python simulate.py xmlfile pdbfile directory_name
prob_state_xml, prod_pdb = MotorRow(sys.argv[1], sys.argv[2], sys.argv[3]).main(sys.argv[2])
