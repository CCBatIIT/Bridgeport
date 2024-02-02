# RepairProtein

This repo provides a quick and accurate way to model the missing residues in a .pdb file. \
To use this tool, you need to have the original .pdb you want to fix, and the .fasta file for the corresponding sequence from rcsb.org \
./RepairProtein_README.png


## Environment 
Important dependencies: 
- modeller 10.4 -> UCSF Modeller license is necessary to use this package. A free academic license is available at https://salilab.org/modeller/registration.html 

## RUN_RepairProtein.py 
Use the RepairProtein class to repair any given .pdb that is in the directory. \
This script is contingent on all the target proteins having the same template sequence. 

### USAGE: 
```
>>>python RUN_RepairProtein.py {OPTIONS}

Required Arguments:
    -i --input_dir: directory where .pdb files of target proteins are found
    -o --output_dir: directory where repaired .pdb files will be added
    -f --fasta: path to .fasta file, which will serve as the template sequence to repair the target proteins

Optional Arguments:
    --tails: if called, will add tail residues to N and C termini.
```


