# RepairProtein Module

The RepairProtein module is designed for repairing and remodeling protein structures using a combination of template sequences, structural information, and homology modeling techniques. It leverages UCSF Modeller for constructing missing parts and refining the overall structure of proteins based on template sequences provided in FASTA format.

## Features

- Automated repair of missing and mutated residues in protein structures.
- Utilizes template sequences from FASTA files for accurate remodeling.
- Supports optimization of loop regions for improved structure prediction.
- Capable of preserving non-standard residues during the repair process.
- Integrates with Modeller and OpenMM for a comprehensive structure repair workflow.

## Requirements

- UCSF Modeller
- MDtraj
- OpenMM
- PDBFixer
- numpy

## Installation

Ensure you have all the required libraries and dependencies installed in your Python environment. Modeller requires a Academic License, which can be aquired at https://salilab.org/modeller/registration.html.

## Usage

### Typical Usage
Typical usage involves remodelling the input structure by filling in holes in the input structure, repairing mutations, and optimizing the structure of loops. 
```python
from repair_protein import RepairProtein

# Initialize
repair = RepairProtein(pdb_fn='path/to/protein.pdb', fasta_fn='path/to/template.fasta', working_dir='path/to/working/dir')

# Run 
repair.run(pdb_out_fn='path/to/protein_output.pdb', tails=[66, 352], loops=[[21, 34], [256, 285]], verbose=True)
```

### Repair Protein with Secondary Structure as Template
Running RepairProtein with a secondary template is advantageous in the scenario where the input structure is missing large chunks of the protein with defined secondary structure. Modeller will not competently build the secondary structure on its own, so providing another .pdb file with the approriate secondary structure helps accurate modelling. 
```python
from repair_protein import RepairProtein

# Initialize
repair = RepairProtein(pdb_fn='path/to/protein.pdb', fasta_fn='path/to/template.fasta', working_dir='path/to/working/dir')

# Run 
repair.run_with_secondary(secondary_template_pdb='path/to/secondary_template.pdb', pdb_out_fn='path/to/protein_output.pdb', tails=[66, 352], loops=[[21, 34], [256, 285]], verbose=True)
```

# File Handling  Module

This module provides utility functions to assist in the preparation of sequence and structural data for protein repair and remodeling using UCSF Modeller. It focuses on converting sequence and structure data into formats compatible with Modeller, specifically for tasks involving missing residue repairs based on template sequences from PDB files.

## Methods

- **fasta_to_pir**: Converts FASTA files to PIR format, which is required by UCSF Modeller for input sequences.
- **parse_sequence**: Extracts sequences from PIR files and returns them as string objects.
- **pdb_to_pir**: Generates PIR files from PDB structure files, facilitating the use of structural data in UCSF Modeller.

# SequenceWrapper Module 

The SequenceWrapper module is a specialized tool designed to identify discrepancies such as missing or mutated residues in a target protein sequence when compared to a template sequence. It integrates seamlessly with UCSF Modeller by preparing alignment and PIR files required for protein structure repair and homology modeling based on a comparative analysis of template and target sequences.

## Key Features

- Identifies missing and mutated residues between the target and template sequences.
- Supports the integration of secondary structure templates for enhanced accuracy in homology modeling.
- Automates the conversion of sequences from FASTA or PDB format to the PIR format required by UCSF Modeller.
- Generates alignment files (.ali) necessary for UCSF Modeller, accommodating missing residues and ensuring structural integrity in the modeling process.

## Methods

- **_find_missing_residues**: Analyzes the target sequence against the template to identify and list missing or mutated residues. This method is fundamental for pinpointing discrepancies between the template and target sequences.
- **_write_alignment_file**: Creates an alignment file in the format required by UCSF Modeller, incorporating sequence information and locations of missing residues. This file is essential for guiding UCSF Modeller in protein structure repair and homology modeling.
- **_write_alignment_file_secondary**: Similar to `_write_alignment_file`, but includes additional structural information from a secondary template structure. This method is useful for enhancing modeling accuracy with extra structural context.
- **_sequence_to_file**: Formats a sequence string for file inclusion, ensuring readability and compliance with bioinformatics tools' expectations. This utility method breaks sequences into appropriately sized lines.
- **_write_struc_section**: Compiles the structured portion of the alignment file, marking missing residues within the template sequence. This section is crucial for UCSF Modeller to understand where the model requires adjustments.
- **_write_seq_section**: Constructs the sequence section of the alignment file, detailing the complete template sequence. This section provides a comprehensive reference for the modeling process, outlining the template sequence in full.




