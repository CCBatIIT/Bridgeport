# Bridgeport - A python package for automated preparation of protein simulations.
## Overview
The Bridgeport module serves as a comprehensive tool for preparing crystal structures for simulation using OpenMM. It integrates several steps necessary for the preparation of both protein and ligand molecules, ensuring they are ready for molecular dynamics simulation. The process includes alignment to a reference structure, separation of ligand and protein, repair and addition of missing residues and atoms, protonation, solvation, and the generation of a complete OpenMM system ready for simulation. This module streamlines the preparation process, combining functionalities from various tools and libraries such as MDAnalysis, mdtraj, OpenMM, PDBFixer, and RDKit, into a singular, cohesive workflow.

![alt text](https://github.com/CCBatIIT/Bridgeport/blob/main/Bridgeport/Bridgeport_Flowchart.png)

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
All the input files and parameters are specifid in the Bridgeprot_input.json. json files can be build interactively using *interactive.ipynb* in the *interactive* module. FOR A VIDEO TUTORIAL VISIT (https://youtu.be/9fXNMXbbLtM)

#### First the working directory must be specified:
- **"working_dir"**: specifies where all output and intermediate directories will be created.

#### "Ligand"
Ligands can be generated using an experimental structure. If the ligand is a small molecule, select the resname in the input.pdb with *resname*. If the ligand is a peptide, select the chainid in the input.pdb with *chainid*. 

Bridgeport is also capable of generating ligand analogues based on a known ligand-protein complex. This workflow consists of forcing the maximum common substructure (MCS) between the analogue and the known structure to have matching torsions and the MCS of the analogue is aligned to that of the known ligand. Then, then a specified number of conformations of functional groups unique to the analogue are generated that maintain a certain RMSD threshold of the MCS and any other specified atoms that are matching in the known ligand and analogue. Then, each conformation is prepared in complex with the receptor and minimized to avoid steric clashes between the receptor and analogue. The lowest potential energy conformation, after minimization, becomes the final conformation of the analogue. 

If you want to generate an analogue of a known structure: Provide the SMILES string for you analogue with the *smiles* key. Provide the name of your analogue with *analogue_name*. There parameters should be in a subdictionary of *Ligand* named *Analogue*.

If no ligand is desired (e.g. apo structure), then do not include the 'Ligand' section in the input.json file.

- **"resname"**: resname of ligand found in *input_pdb*. If the ligand, is a peptide set to "false".
- **"chainid"**: If ligand is a peptide, specify the letter code that denotes the ligand, if not choose "false".
- **"sequence"**: If ligand is a peptide, and you would like to use modeller to repair the ligand, provide the sequence.
- **"smiles"**: Mandatory entry of smiles string of the ligand (either small molecule or peptide) in the input.pdb file
- **"peptide_nonstandard_resids"**: If non-standard residues are present in the peptide, present the indicies of the non-standard residues in ascending order, 0-indexed. This step is only necessary if a *sequence* is provided as well.
- **"name"**: Should match the name of your input.pdb
- **"small_molecule_params"**: If true, treat ligand like a small molecule. Default is True.
- **"sanitize"**: If true, sanitize molecule with rdkit. Default is True. Only applicable if small_molecule_params is True. 
- **"removeHs"**: If true, remove any hydrogens that may be present. Default is True. Only applicable if small_molecule_params is True.
- **"proximityBonding"**: If true, use rdkit's 'proximityBonding' method to load rdkit molecule. 
- **"pH"**: pH to protonate a peptide ligand. Default is 7.0.
- **"nstd_resids "**: List of nonstandard resids to conserve from input structure. 
- **"neutral_Cterm"**: If true, neutralize the C-terminus of a peptide ligand. Only applicable is small_molecule_params is False


- **"Analogue"**:
    - **"name"**: a name for your ligand
    - **"smiles"**: smiles str of your analogue.
    - **"add_atoms"**: List of atom inds as depicted to be added to common substructure. Ex: [[0, 1], [2, 4]] where atoms 0 and 2 in the analogue match atoms 1 and 4 in the template, respectively. Default is False, which will use automatically determined maximum common substructure.
    - **"remove_atoms"**: List of atoms inds to remove from the analogue. The corresponding atoms from the template structure will be removed automatically. Default is False, which will use automatically determined maximum common substructure.
    - **"align_all"**: If True, will use atoms in *add_atoms* for alignment. Default is False which will only use the automatically detected maximum common substructure.
    - **"rmsd_tresh"**: RMSD threshold that analogue conformation must reach during alignment to be accepted as a permittable structure. Default is 3.0 Angstrom.


#### "Protein"
- **"input_pdb_dir"**: Path to directory where the input .pdb can be found.
- **"input_pdb"**: Name of .pdb file to use as an input structure.
- **"chain"**: One letter code of chain to parse from file. There is currently not an option to select multiple chains. 

- **"RepairProtein"** There is currently not an option to skip protein repair.
    - **"working_dir"**: Path to directory to put all the modeller intermediates.
    - **"fasta_path"**: Path to .fasta file to repair the input protein structure.
    - **"tails"**: List of indices to parse the extra tails. EX: [30, 479]
    - **"loops"**: 2-D List of indices that specify lower and upper bounds of loops to optimize during refinement. Loop optimization can take a while, but if skipped, unbonded output structures will result.
    - **"engineered_resids"**: List of resids that are known engineered mutations in the crystal pdb. Adding this argument may prevent sequence errors in the RepairProtein section of Bridgeport.
    - **"secondary_template"**: Path to secondary .pdb to use as a reference to accurately model large portions that are missing in the input .pdb structure. 

#### "Environment" 
- **"membrane"**: If membrane should be specified choose "true", and make sure that "alignment_ref" argument is the appropriate OPM structure. Default is false. 
- **"pH"**: Specify the pH. Default is 7.0.
- **"ion_strength"**: Specify the concentration of NaCl ions (in Molar). Default is 0.15 M.
- **"alignment_structure"**: Path to structure to align the final system to.
- **"alignment_chains"**: List of chains in the alignment structure to use for alignment. EX: ["A"]

