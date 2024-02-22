"""
Simple python script to run Bridgeport job

Positional arguments
   [1] String path to input.json file. 
"""

#Imports
from Bridgeport import Bridgeport
import os, sys
from datetime import datetime

# Assert that provided .json file is valid
json_path = sys.argv[1]
if not os.path.exists(json_path):
	raise FileNotFoundError(f"Cannot find the input .json file at given location {json_path}.")

# Run Bridgeport
start_time = datetime.now()
BP = Bridgeport(input_json=json_path)
BP.run()
end_time = datetime.now()
print('Time to run:', end_time - start_time)
