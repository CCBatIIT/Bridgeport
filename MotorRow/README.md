# MotorRow - A sub-module to Bridgeport for the equilibration of prepared membrane protein systems.
## Overview
This module executes a five stage equilibration protocol specific to membrane proteins.  These are:
1. Minimization (Typical OpenMM Minimization)​
2. 250 ps of NVT with restraints​
2.1. Restraint 1 – All Protein Heavy Atoms​
2.2. Restraint 2 – Z Coordinate of Membrane Heavy Atoms​
3. 250 ps of NVT without restraints​
4. 500 ps of NPT with Monte Carlo Membrane Barostat​
5. 500 ps of NPT with Monte Carlo Barostat

Following these five steps, the system that was contructed with Bridgeport is ready for production simulation.

## MotorRow Usage 
Bridgeport can be easily run in 2 lines of code:

```python
input_pdb, input_xml = 'built.pdb', 'built.xml' # The outputs of Bridgeport
MR = MotorRow(input_xml, input_pdb, os.path.getcwd())
production_xml, production_pdb = MR.main(input_pdb)
```

### Input files
An XML/PDB Pair which were generated with the main usage case of Bridgeport\
Note that this XML is an XML serialized OpenMM System, not a force field file

### Output files
An XML/PDB Pair which can now be used in a production simulation.\
Note that this XML is an XML serialized OpenMM System, not a force field file