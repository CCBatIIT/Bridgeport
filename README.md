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
Ligands can be generated using an experimental structure. If the ligand is a small molecule, select the resname in the input.pdb with "lig_resname". If the ligand is a peptide, select the chainid in the input.pdb with "peptide_chainid". 

Bridgeprot is also capable of generating ligand analogues based on a known ligand-protein complex. This workflow consists of forcing the maximum common substructure (MCS) between the analogue and the known structure to have matching torsions and the MCS of the analogue is aligned to that of the known ligand. Then, then a specified number of conformations of functional groups unique to the analogue are generated that maintain a certain RMSD threshold of the MCS and any other specified atoms that are matching in the known ligand and analogue. Then, each conformation is prepared in complex with the receptor and minimized to avoid steric clashes between the receptor and analogue. The lowest potential energy conformation, after minimization, becomes the final conformation of the analogue. 

If you want to generate an analogue of a known structure: Make sure lig_resname and peptide_chain are set to false. Provide the SMILES string for you analogue with the "analogue_smiles" key. Provide the name of your analogue with "analogue_name". Provide a path to the known structure with "known_structure" that contains the experimental ligand to align analogue to with a "known_resname" or "known_chainid". 

- **"lig_resname"**: specifies the ligand resname in the input .pdb file. If the ligand is a peptide choose "false". Ligand resnames that start with and number are not easily recognized by MDAnalysis (which is used for parsing), so we recommend changing the ligand resname in that case.
- **"peptide_chain"**: If ligand is a peptide, specify the letter code that denotes the ligand, if not choose "false".
- **"peptide_fasta"**: If ligand is a peptide, specify the path to the .fasta file to repair the peptide if desired. If no repair is desired, choose "false", or remove argument.
- **"peptide_nonstandard_resids"**: If non-standard residues are present in the peptide, present the indicies of the non-standard residues in ascending order, 0-indexed. This step is only necessary if a .fasta file is provided as well. 
- **"analogue_smiles"**: String of smiles that represent the analogue to generate. Optional argument.
- **"analogue_name"**: Name to generate new files with. 
- **"known_structure"**: Path to pdb file that contains the known ligand to align analogue to. This should be the same path as "protein" "input_pdb" to start, Bridgeport will correct automatically later. 
- **"known_resname"**:  Resname of ligand to parse in known_structure, if known ligand is a small molecule.
- **"known_chainid"**: Chainid of ligand to parse in known_structure, if known ligand is a peptide.
- **"add_known_atoms"**: List of atom names that are also in analogue, but not recognized in the MCS. These atoms are used to compare the RMSD during conformer generation.
- **"add_known_resids"**: List of resids that match each atom in "add_known_atoms". This should always be specified, but is especially helpful in cases where the known ligand is a peptide and each residue may have the same atoms names (ex: CA, N, C, O, etc.)
- **"remove_analogue_atoms"**: List of atom names that are found in the MCS, but are unwanted in the MCS due to user discretion. 
- **"add_analogue_atoms"**: List of atoms names that match each atom in "add_known_atoms". For example if "add_known_atoms" is ['C21', 'C23', 'N5'] than add_analogue_atoms should be a list of the equivalent atoms in the analogue structure, such as ['C22', 'C24', 'N5']. It may be helpful to run Bridgeport without specifying any of the "add" arguments to generate the .pdb file of the ligand to better understand the atom names that will be generated from the provided "analogue_smiles". 
- **"analogue_rmsd_thres"***: RMSD threshold that analogue conformer generation cannot exceed. RMSD is calculated of MCS and other specified atoms in "add_known_atoms", "add_known_resids", and "add_analogue_atoms"
- **"small_molecule_params"**: If analogue is a peptide, setting this option to true will choose to parameterize it as a small molecule with Smirnoff from OpenFF.
- **"align_all"**: If true, use both atoms identified in MCS and atoms specified in "add_known_atoms", "add_analogue_atoms" for alignment. Default is false, which won't use additional atoms specified in "add_known_atoms", "add_analogue_atoms" for alignment, only using them for assigning matching torsions. 

#### "protein"
- **"input_pdb_dir"**: Path to directory where the input .pdb can be found.
- **"input_pdb"**: Name of .pdb file to use as an input structure.
- **"chains"**: One letter code of chain to parse from file. There is currently not an option to select multiple chains. 

- **"RepairProtein"** There is currently not an option to skip protein repair.
    - **"working_dir"**: Path to directory to put all the modeller intermediates.
    - **"fasta_path"**: Path to .fasta file to repair the input protein structure.
    - **"tails"**: List of indices to parse the extra tails. EX: [30, 479]
    - **"loops"**: 2-D List of indices that specify lower and upper bounds of loops to optimize during refinement. Loop optimization can take a while, but if skipped, unbonded output structures will result.
    - **"engineered_resids"**: List of resids that are known engineered mutations in the crystal pdb. Adding this argument may prevent sequence errors in the RepairProtein section of Bridgeport.
    - **"secondary_template"**: Path to secondary .pdb to use as a reference to accurately model large portions that are missing in the input .pdb structure. 

#### "environment" 
- **"membrane"**: If membrane should be specified choose "true", and make sure that "alignment_ref" argument is the appropriate OPM structure. Default is false. 
- **"pH"**: Specify the pH. Default is 7.0.
- **"ion_strength"**: Specify the concentration of NaCl ions (in Molar). Default is 0.15 M.
- **"alignment_structure"**: Path to structure to align the final system to.
- **"alignment_chains"**: List of chains in the alignment structure to use for alignment. EX: ["A"]

## Examples
### 7vvk (membrane, peptide ligand)
This example uses the PDBID 7vvk as the input structure and repairs both the protein and the peptide ligand. 

### 8jr9 (membrane, small-molecule ligand, secondary template)
This example uses the PDBID 8jr9 as the input structure and repairs the protein with the secondary template of the repaired 7vvk.pdb structure to properly model the extracellular domain. This example uses a small molecule ligand. 
### 5zty analogue (membrane, analogue, small-molecule ligand)
This example uses the protein and ligand from the PDBID 5zty but creates the final ligand from a smiles string and alignes maximum common substructure to ligand in 5zty.

### Leu-Enkphalin (membrane, small-peptide ligand that is an analogue of Endomorphin-1 (8f7r))
This example showcases how to use the small molecule alignment, including specifying extra atoms to match that are not recognized as the maximum common substructure from rdkit. 
