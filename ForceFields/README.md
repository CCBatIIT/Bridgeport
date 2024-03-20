# ForceFieldHandler Module

This module provides functionality for parameterizing molecular structures for simulation using OpenFF (Open Force Field) for SDF files and OpenMM for PDB files. It automates the process of generating system, topology, and positions needed to construct an OpenMM simulation environment.

## Features

- Automatic detection and parameterization based on input file type (SDF or PDB).
- Support for utilizing custom force field files in addition to default provided force fields.
- Ability to generate custom XML force field files from provided parameters.

## Requirements

- OpenFF Toolkit
- RDKit
- OpenMM
- numpy

## Usage
To use the ForceFieldHandler class, import it from the module and initialize it with the path to your structure file. Optionally, you can specify a list of custom force field files.

### Use Default Forcefields

#### Default Forcefields
- For .pdb files
    - amber14_protein14SB
    - amber14_lipid17
    - wat_opc3
- For .sdf files
    - openff-2.1.0.offxml

```python
from ForceFieldHandler import ForceFieldHandler

# Initialize
handler = ForceFieldHandler(structure_file="path/to/your/structure.sdf")

# Parameterize
sys, top, pos = handler.main()
```

### Use Custom Forcefields
Specify the usage of a custom.xml file with argument 'force_field_files'. You can generate a custom XML force field file tailored to your parameters using the generate_custom_xml method.

```python
from ForceFieldHandler import ForceFieldHandler

# Generate custom .xml
handler = ForceFieldHandler(structure_file="path/to/you/structure.pdb")
handler.generate_custom_xml(out_xml="path/to/output.xml", name="RES")

# Initialize 
handler = ForceFieldHandler(structure_file="path/to/your/structure.pdb", force_field_files=["path/to/output.xml"], use_defaults=False)

# Parameterize
sys, top, pos = handler.main()
```

# Joiner Module

The Joiner module is designed to facilitate the addition of an OpenFF parameterized ligand system to an OpenMM environment. This allows users to seamlessly integrate small molecule systems with more complex biological systems, such as proteins, in their simulation setup.

## Features

- Combines OpenMM systems, topologies, and positions for ligand and receptor sets.
- Supports integration of OpenFF ligands into OpenMM environments.
- Automated handling of system, topology, and position merging for comprehensive simulation setups.

## Requirements

- OpenMM
- OpenFF Toolkit
- numpy

## Usage
### Initialization
To use the Joiner class, import it from the module and initialize it with the ligand and receptor sets. Each set must be a 3-tuple of System, Topology, and Positions.

```python
from joiner_module import Joiner

ligand_set = (ligand_system, ligand_topology, ligand_positions)
receptor_set = (receptor_system, receptor_topology, receptor_positions)

joiner = Joiner(ligand_set, receptor_set)
```

### Merging OpenMM Systems
Invoke the main method to combine the ligand and receptor sets into a single set, ready for simulation.

```python
combined_system, combined_topology, combined_positions = joiner.main()
```








