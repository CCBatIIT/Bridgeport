# Docking Module
The Docking module is designed to facilitate the docking of ligands to protein structures using AutoDock Vina 1.2.3. It provides a straightforward interface for setting up and running docking simulations, as well as for comparing the resulting poses with reference structures.

## Features

- Easy setup of docking parameters and execution of docking simulations.
- Comparison of docked poses with reference structures using RMSD of the maximum common substructure.

## Conda Environments
### Environment 1 Requirements
- mgltools 1.5.7 (ONLY)

### Enviroment 2 Requirements
- AutoDock Vina 1.2.3
- MDAnalysis
- RDKit
- numpy
- openbabel

## Preparing Files for Docking
Conda Environment 1 is necessary to prepare input structures for Autodock Vina. To avoid issues with the python 2 interpretor, make sure to run the following whenever switching to a Conda Environment 1.
```bash
source mgltools_config.sh
```

### Command Line Usage
This script prepares receptors and ligands for docking by converting them into the .pdbqt format using mgltools. 

To use the script, you need to specify directories for the input receptor and ligand files as well as the output directories where the converted .pdbqt files will be saved. The conversions are powered by prepare_receptor4.py and prepare_ligand4.py from the mgltools package.

Use `-r` to specify the input directory for receptors, `-l` for the input directory for ligands, `-or` for the output directory for converted receptors, and `-ol` for the output directory for converted ligands.

```bash
python prepare_docking.py -r path/to/receptor_dir -l path/to/ligand_dir -or path/to/out_receptor_dir -ol path/to/out_ligand_dir
```

## Docking
Using Conda Environment 2...

### Command Line Usage
This script facilitates the docking of ligands to protein structures using AutoDock Vina through a command-line interface. It allows users to specify parameters for docking, including paths to the receptor and ligand files, the docking box dimensions, and options for the docking process.

#### Basic Usage
To run the docking script, provide the mandatory arguments `-r` (path to the receptor .pdbqt file) and `-l` (path to the ligand .pdbqt file):

```bash
python RUN_DOCKING.py -r path/to/receptor.pdbqt -l path/to/ligand.pdbqt
```

#### Specifying Docking Parameters
You can specify additional docking parameters such as the output directory for docked poses, the configuration directory, the center and dimensions of the docking box, the number of poses, exhaustiveness, and the minimum RMSD for saving a new pose. Here is an example command with additional parameters:

```bash
python RUN_DOCKING.py -r path/to/receptor.pdbqt -l path/to/ligand.pdbqt -ol path/to/output/dir --config_dir path/to/config/dir --box_center 10,20,30 --box_dim 20,20,20 --n_poses 10 --exhaustiveness 9 --min_rmsd 1.5
```

#### Comparison with Reference Structure
To compare the maximum common substructure of a docked poses with a reference ligand, use the --compare argument followed by the path to the reference .pdb file, the chain ID of the protein, and the MDAnalysis selection string to parse the ligand. Separate these three values with spaces:

```bash
python RUN_DOCKING.py -r path/to/receptor.pdbqt -l path/to/ligand.pdbqt --compare path/to/ref.pdb R "selection string"
```

### Python API

#### Initialization
```python
from Docking import Docking

docking = Docking(receptor_path='path/to/receptor.pdbqt', ligand_path='path/to/ligand.pdbqt')
```

#### Setting the Docking Box
```python
docking.set_box(box_center=[x, y, z], box_dim=[size_x, size_y, size_z])
```

#### Running the Docking
```python
docking.dock(lig_out_dir='path/to/output', n_poses=20, exhaustiveness=8, min_rmsd=1.0)
```

#### Comparing Docked Poses with Reference Structure
```python
rmsds = docking.compare(ref_pdb='path/to/reference.pdb', ref_chainid='A', ref_lig_sele_str='ligand_selection_string')
print(rmsds)
```



