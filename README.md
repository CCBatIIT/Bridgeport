# Bridgeport - A python package for automated preparation of protein simulations.
## Overview
The Bridgeport module serves as a comprehensive tool for preparing crystal structures for simulation using OpenMM. It integrates several steps necessary for the preparation of both protein and ligand molecules, ensuring they are ready for molecular dynamics simulation. The process includes alignment to a reference structure, separation of ligand and protein, repair and addition of missing residues and atoms, protonation, solvation, and the generation of a complete OpenMM system ready for simulation. This module streamlines the preparation process, combining functionalities from various tools and libraries such as MDAnalysis, mdtraj, OpenMM, PDBFixer, and RDKit, into a singular, cohesive workflow.

![alt text](https://github.com/CCBatIIT/Bridgeport/blob/main/Bridgeport_Flowchart.png)

## Environment
Construct the conda environment with the following command
```bash
conda env create -f conda_setup.yml
```

## BridgePort Usage 
Bridgeport can be easily run in 2 lines of code:

```python
BP = Bridgeport(input_json='Bridgeport_input.json')
BP.run()
```

### Input .json files
All the input files and parameters are specifid in the Bridgeprot_input.json:

#### First the working directory must be specified:
- **"working_dir"**: specifies where all output and intermediate directories will be created.

#### "ligand"

- **"lig_resname"**: specifies the ligand resname in the input .pdb file. If the ligand is a peptide choose "false". Ligand resnames that start with and number are not easily recognized by MDAnalysis (which is used for parsing), so we recommend changing the ligand resname in that case.
- **"peptide_chain"**: If ligand is a peptide, specify the letter code that denotes the ligand, if not choose "false".
- **"peptide_fasta"**: If ligand is a peptide, specify the path to the .fasta file to repair the peptide if desired. If no repair is desired, choose "false", or remove argument.
- **"peptide_nonstandard_resids"**: If non-standard residues are present in the peptide, present the indicies of the non-standard residues in ascending order, 0-indexed. This step is only necessary if a .fasta file is provided as well. 
- **"analogue_smiles"**: String of smiles that represent the analogue to generate. Optional argument.
- **"analogue_name"**: Name to generate new files with. 
- **"known_structure"**: Path to pdb file that contains the known ligand to align analogue to. This should be the same path as "protein" "input_pdb" to start, Bridgeport will correct automatically later. 
- **"known_resname"**:  Resname of ligand to parse in known_structure, if known ligand is a small molecule.
- **"known_chainid"**: Chainid of ligand to parse in known_structure, if known ligand is a peptide.
- **"small_molecule_params"**: If analogue is a peptide, setting this option to true will choose to parameterize it as a small molecule with Smirnoff from OpenFF. 

#### "protein"
- **"input_pdb_dir"**: Path to directory where the input .pdb can be found.
- **"input_pdb"**: Name of .pdb file to use as an input structure.
- **"chains"**: One letter code of chain to parse from file. There is currently not an option to select multiple chains. 

- **"RepairProtein"** There is currently not an option to skip protein repair.
    - **"working_dir"**: Path to directory to put all the modeller intermediates.
    - **"fasta_path"**: Path to .fasta file to repair the input protein structure.
    - **"tails"**: List of indices to parse the extra tails. EX: [30, 479]
    - **"loops"**: 2-D List of indices that specify lower and upper bounds of loops to optimize during refinement. Loop optimization can take a while, but if skipped, unbonded output structures will result. 
    - **"secondary_template"**: Path to secondary .pdb to use as a reference to accurately model large portions that are missing in the input .pdb structure. 

#### "environment" 
- **"membrane"**: If membrane should be specified choose "true", and make sure that "alignment_ref" argument is the appropriate OPM structure. Default is false. 
- **"pH"**: Specify the pH. Default is 7.0.
- **"ion_strength"**: Specify the concentration of NaCl ions (in Molar). Default is 0.15 M.
- **"alignment_structure"**: Path to structure to align the final system to.

## Examples
### 7vvk (membrane, peptide ligand)
This example uses the PDBID 7vvk as the input structure and repairs both the protein and the peptide ligand. 

### 8jr9 (membrane, small-molecule ligand, secondary template)
This example uses the PDBID 8jr9 as the input structure and repairs the protein with the secondary template of the repaired 7vvk.pdb structure to properly model the extracellular domain. This example uses a small molecule ligand. 
### 5zty analogue (membrane, analogue, small-molecule ligand)
This example uses the protein and ligand from the PDBID 5zty but creates the final ligand from a smiles string and alignes maximum common substructure to ligand in 5zty.

### Leu-Enkphalin (membrane, small-peptide ligand that is an analogue of Endomorphin-1 (8f7r))
This example showcases how to use the small molecule alignment, including specifying extra atoms to match that are not recognized as the maximum common substructure from rdkit. 
