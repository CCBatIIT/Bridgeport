# ProteinPreparer Module 
ProteinPreparer automates several crucial preprocessing steps required to ready a protein structure for molecular simulation. This includes correcting common issues in crystal structures, ensuring proper protonation states, and setting up the environment for the simulation, which can either be an explicit membrane and solvent system or a solvent-only system.

## Methods
- **__init__**: Initializes the ProteinPreparer object with the path to the PDB file, working directory, pH level, environment type, and ion strength for the simulation preparation process.

- **main**: Coordinates the workflow for preparing the protein, including protonation and environment creation. Returns the path to the prepared protein structure file.
- **_protonate_with_pdb2pqr**: Uses `pdb2pqr30` to protonate the protein at the specified pH, also handling the addition of missing atoms. This step does not add missing residues.
- **_protonate_with_PDBFixer**: An alternative protonation method using PDBFixer. This method is called if pdb2pqr30 is not used or if additional protonation adjustments are needed after pdb2pqr30 processing.
- **_run_PDBFixer**: Creates the specified simulation environment around the protein. Depending on the mode ('MEM' or 'SOL'), it adds either a membrane and solvent or just solvent around the protein. This method also takes care of additional missing atoms, not residues, and adjusts the protonation states if necessary.

## Usage

Output files will be found in specified working_dir with suffixes '_H.pdb' to indicate a protonated protein and '_env.pdb' to indicate a solvated and/or membrane protein. 

Valid options for environment building are:
- **'MEM'**: insert protein into membrane and solvate
- **'SOL'**: solvate protein 

```python
from ProteinPreparer import ProteinPreparer

#Initialization
pp = ProteinPreparer(pdb_path='/path/to/file.pdb', working_dir='path/to/working_dir', pH=7.0, env='MEM', ion_strength=0.15)
                                 
# Run
pp.main()
```

# bp_utils Module 

This module provides utilities the Bridgeport module.

## Methods

- **analogue_alignment**: Creates an aligned analogue of a known ligand structure based on SMILES notation. It uses RDKit for ligand generation, MDAnalysis for structural alignment based on the maximum common substructure, and returns the RMSD of the aligned structures.
- **return_max_common_substructure**: Identifies the maximum common substructure between two molecules, aiding in the alignment process by focusing on structurally conserved regions.
- **change_resname**: Allows for the modification of residue names within a PDB file, facilitating the customization of ligand identifiers in preparation for simulation or further analysis.
- **describe_system**: Provides a summary of an OpenMM system, including box vectors, forces, and the total number of particles, offering insights into the setup of molecular dynamics simulations.
- **describe_state**: Outputs the energy and maximum force of a given OpenMM state, aiding in the evaluation of system stability and the identification of potential issues in simulation setups.
