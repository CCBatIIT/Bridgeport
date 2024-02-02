from Bio import SeqIO
from modeller import *
from modeller.automodel import *
import math
import numpy as np


"""
Methods that are implemented in RepairProtein class, which all assist in quickly using UCSF Modeller to fix missing residues based on the .fasta sequence from the PDB.

Methods:
--------
    fasta_to_pir: Convert .fasta to .pir type. UCSF Modeller expects input files to be in .pir type, which is unavailable on the PDB.
    parse_sequence: Return the sequence from any .pir file as a str object.
    pdb_to_pir: Use the sequence of the provided .pdb to create the .pir file for the structure. 
"""

def fasta_to_pir(fasta_fn: str, pir_out_fn: str):
    """
    Convert .fasta file type to .pir file type.

    Parameters:
    -----------
        fasta_fn (str):
            String path to .fasta file to parse.
        
        pir_out_fn (str):
            String path to output .pir file. 

    Returns:
    --------
        String path to .pir file.
    """
    # Parse file
    SeqIO_obj = SeqIO.parse(open(fasta_fn), 'fasta')

    # Convert to .pir
    SeqIO.write(SeqIO_obj, pir_out_fn, 'pir')

def parse_sequence(pir_fn: str):
    """ 
    Return str object of sequence from .pir file type.

    Parameters:
    -----------
        pir_fn (str):
            String path to .pir file to parse.

    Returns:
    --------
        String object with sequence parsed from .pir file. 
    """
    # Parse file
    return ''.join([line[:-1] for line in open(pir_fn, 'r').readlines() if line != '\n'][2:])[:-1]

def pdb_to_pir(pdb_name: str, pdb_dir: str):
    """
    Get the sequence of a .pdb file to .pir file.

    Parameters:
    -----------
        pdb_name (str):
            String NAME of .pdb file to parse. 

        pdb_dir (str):
            String path to directory contaning pdb_name.pdb

    Returns:
    --------
        String path to .pir file.
    """
    # Set up Modeller environment
    e = Environ()
    m = Model(e, file=pdb_dir + '/' + pdb_name + '.pdb')

    # Write with alignment object
    aln = Alignment(e)
    aln.append_model(m, align_codes=pdb_name)
    pir_fn = pdb_dir + '/' + pdb_name + '.pir'
    aln.write(file=pir_fn)
