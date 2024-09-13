import math
import numpy as np
from datetime import datetime


class SequenceWrapper():
    """
    This class is able to identify the missing holes in a target sequence, when compared to a template sequence. 

    Attributes:
    -----------
        temp_seq (str): String sequence that is the template. e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.

        tar_seq (str): String sequence that is should match the template. e.g. 'CDGHLMNOTVXY'

        missing_residues (np.array): 2-D Array of indices of missing sequence entries and what the missing entry is. e.g. [[0, 'A'], [1, 'B'], [4, 'E'], ....]

        
    """

    def __init__(self, template_seq: str, target_seq: str, secondary_seq: str=None):
        """ 
        Initialize the SequenceWrapper class

        Parameters:
        -----------
            template_sequence (str):
                String sequence that is the template. e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.
            
            target_sequence (str):
                String sequence that is should match the target. e.g. 'CDGHLMNOTVXY'

            secondary_template_sequence (str):
                String sequence that should match the secondary template structure. Default is None. 
        """

        # Assert
        assert len(template_seq) >= len(target_seq), f"Length of template sequence {len(template_seq)} should not be shorter than target sequence {len(target_seq)}."

        # Initialize objects:
        self.temp_seq = template_seq
        self.tar_seq = target_seq
        self.secondary_seq = secondary_seq
        self.missing_residues = []

    def find_missing_residues(self, verbose: bool=False):
        """
        Find the missing residues in the target sequence.

        Parameters:
        -----------
            verbose (bool):
                If True, will print missing residues and indices of missing residues to console. Default is True.
        """ 
        # Find the missing residues
        if self.secondary_seq == None:
            self.missing_residues, self.mutated_residues, self.term_residues = _find_missing_residues(tar_seq=self.tar_seq, temp_seq=self.temp_seq, verbose=verbose)
        else:
            self.missing_residues, self.mutated_residues, self.term_residues = _find_missing_residues(tar_seq=self.tar_seq, temp_seq=self.temp_seq, verbose=verbose)
            self.missing_residues_secondary, _, _ = _find_missing_residues(tar_seq=self.secondary_seq, temp_seq=self.temp_seq, verbose=verbose)            

    def write_alignment_file(self, ali_fn: str, reference_pir_fn: str, secondary_pir_fn: str=None):
        """
        Write an alignment file for UCSF Modeller. This method puts the sequence information in the format as specified by an example from https://salilab.org/modeller/wiki/Missing_residues

        Parameters:
        -----------
            ali_fn (str):
                String path to write .ali file.

            reference_pir_fn (str):
                String path to get structural information from. This .pir file should be created from the .pdb that is in need of mending.

            secondary_pir_fn (str):
                String path to get structural information from for secondary template. This .pir file should be created from the .pdb that is in need of mending. Default is None.

        """
        # Write the alignment file
        if hasattr(self, "missing_residues_secondary"):
            _write_alignment_file_secondary(temp_seq=self.temp_seq, 
                                            missing_residues=self.missing_residues, 
                                            missing_residues_secondary=self.missing_residues_secondary, 
                                            ali_fn=ali_fn, 
                                            reference_pir_fn=reference_pir_fn,
                                            secondary_pir_fn=secondary_pir_fn)
        else:
            _write_alignment_file(temp_seq=self.temp_seq, missing_residues=self.missing_residues, ali_fn=ali_fn, reference_pir_fn=reference_pir_fn)
        

"""
These methods provides essential functionality for handling and manipulating protein sequences, particularly in the context of preparing input for UCSF Modeller. It enables the identification of missing residues in a target sequence when compared to a template sequence and facilitates the creation of PIR and alignment files required by Modeller for protein structure repair and homology modeling.


Methods:
--------
    _find_missing_residues(temp_seq, tar_seq, verbose): 
        Identifies missing and mutated residues in the target sequence compared to a template sequence.
    
    _write_alignment_file(temp_seq, missing_residues, ali_fn, reference_pir_fn): 
        Creates an alignment file (.ali) for UCSF Modeller based on the template sequence and identified missing residues.
   
    _write_alignment_file_secondary(temp_seq, missing_residues, missing_residues_secondary, ali_fn, reference_pir_fn, secondary_pir_fn): 
        Similar to _write_alignment_file but includes handling for a secondary template structure.
   
    _sequence_to_file(seq): 
        Helper function that formats a sequence string for inclusion in a file, breaking it into lines of appropriate length.
    
    _write_struc_section(writer_obj, struc_seq, ref_lines): 
        Writes the structure section of the alignment file.
    
    _write_seq_section(writer_obj, temp_seq, ref_lines): 
        Writes the sequence section of the alignment file.
"""


def _find_missing_residues(temp_seq: str, tar_seq: str, verbose: bool=False):
        """
        Find the missing residues in the target sequence.

        Parameters:
        -----------
            temp_seq (str):
                String of the sequence that will be used as the template. e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.

            tar_sequence (str):
                String of the sequence that will be used as the target. e.g. 'CDGHLMNOTVXY'

            verbose (bool):
                If True, will print missing residues and indices of missing residues to console. Default is True.

        Returns:
        --------
            missing_residues (np.array): 2-D Array of indices of missing sequence entries and what the missing entry is. e.g. [[0, 'A'], [1, 'B'], [4, 'E'], ....]
        """
        missing_residues = []
        mutated_residues = []
        term_residues = [0,0]

        # Find missing residues at start of target sequence
        if verbose:
            print('targ:', tar_seq)
            print('temp:', temp_seq, '\n')
        tar_start_ind = temp_seq.index(tar_seq[:3])
        missing_start = temp_seq[:tar_start_ind]
        for ind, res in enumerate(missing_start):
            missing_residues.append([ind, res])
        tar_seq = missing_start + tar_seq
        term_residues[0] = len(missing_start)
        if verbose:    
            print('targ:', tar_seq)
            print('temp:', temp_seq, '\n')
            print('strt:', missing_start) 
        # Iterate through remaining residues
        counter = 0
        while temp_seq[:len(tar_seq)] != tar_seq:
            for i in range(len(tar_seq)):
                # Append if not matching
                temp_res = temp_seq[i]
                tar_res = tar_seq[i]
                if tar_res != temp_res:
                    if verbose:          
                        print(i)
                        print(len(temp_seq), len(tar_seq))
                        print('targ:', tar_seq)
                        print('temp:', temp_seq)
                        print(temp_res, tar_res)

                    # Check for point mutation
                    if tar_seq[i+1:i+5] == temp_seq[i+1:i+5]:
                        if verbose:
                            print(temp_seq[i-10:i], temp_seq[i:i+10], '\n')
                            print(tar_seq[i-10:i], tar_seq[i:i+10])
                            print('Point' , i, i-tar_start_ind, '\n\n\n')
                        tar_seq = tar_seq[:i] + temp_res + tar_seq[i+1:]
                        mutated_residues.append([i - tar_start_ind, temp_res])

                    # Check for deletion
                    else:
                        if verbose:
                            print(tar_seq[i-10:i], tar_seq[i:i+10])
                            print('Del', '\n')
                        tar_seq = tar_seq[:i] + temp_res + tar_seq[i:]
                        missing_residues.append([i, temp_res])

                    break
             
            # Raise error if too many iterations
            if counter == 10000:
                raise RuntimeError("Unable to match template and target sequences")
            # elif tar_seq[:len(temp_seq)] == temp_seq:
            #     tar_seq = tar_seq[:len(temp_seq)]
            #     break
            else:
                counter+=1
        # Add remaining missing residues
        # print('!!!tar_seq ', tar_seq)
        # print('!!!temp_seq', temp_seq, '\n')   
        upper_ind = len(tar_seq)
        term_residues[1] = upper_ind
        for i, res in enumerate(temp_seq[upper_ind:]):
            ind = i + upper_ind
            missing_residues.append([ind, res])
            tar_seq += res

        # Check for match
        if not temp_seq == tar_seq:
            raise Exception(f"Sequences do not match. Attempts to match target sequence resulted in:n\\tTemplate: {temp_seq}\n\tTarget: {tar_seq}")
        elif verbose == True:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Missing Residues:', missing_residues, flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Mutated Residues:', mutated_residues, flush=True)

        return [np.array(missing_residues), np.array(mutated_residues), term_residues]

def _write_alignment_file(temp_seq: str, missing_residues: np.array, ali_fn: str, reference_pir_fn: str):
    """
    Write an alignment file for UCSF Modeller. This method puts the sequence information in the format as specified by an example from https://salilab.org/modeller/wiki/Missing_residues

    Parameters:
    -----------
        temp_seq (str):
            String of template sequence. 

        missing_residues (np.array):
            Array with inidices of missing residues. 
    
        ali_fn (str):
            String path to write .ali file.

        reference_pir_fn (str):
            String path to get structural information from. This .pir file should be created from the .pdb that is in need of mending.
    """

    # Make sequence lists
    struc_seq = ''
    for i, temp_res in enumerate(temp_seq):

        # Append structure list
        if len(missing_residues) > 1:
            if str(i) in missing_residues[:,0]:
                struc_seq += '-'
            else:
                struc_seq += temp_res
                    
    # Read lines from reference
    ref_lines = [line for line in open(reference_pir_fn, 'r').readlines() if line != '\n']

    # Open file obj
    w = open(ali_fn, 'w')

    # Write stucture portion
    w = _write_struc_section(w, struc_seq, ref_lines)

    # Write sequence portion
    w = _write_seq_section(w, temp_seq, ref_lines)
    w.close()

def _write_alignment_file_secondary(temp_seq: str, 
                                             missing_residues: np.array, 
                                             missing_residues_secondary: np.array, 
                                             ali_fn: str, 
                                             reference_pir_fn: str,
                                             secondary_pir_fn: str):
    """
    Write an alignment file for UCSF Modeller. This method puts the sequence information in the format as specified by an example from https://salilab.org/modeller/wiki/Missing_residues

    Parameters:
    -----------
        temp_seq (str):
            String of template sequence. 

        missing_residues (np.array):
            Array with inidices of missing residues. 

        missing_residues_secondary (np.array):
            Array with inidices of missing residues in secondary.         
    
        ali_fn (str):
            String path to write .ali file.

        reference_pir_fn (str):
            String path to get structural information from. This .pir file should be created from the .pdb that is in need of mending.

        secondary_pir_fn (str):
            String path to get structural information from for secondary structure. This .pir file should be created from the .pdb that is in need of mending.
    """

    # Make sequence lists
    struc_seq = ''
    for i, temp_res in enumerate(temp_seq):

        # Append structure list
       if str(i) in missing_residues[:,0]:
            struc_seq += '-'
       else:
            struc_seq += temp_res

    # Make sequence lists for secondary
    secondary_seq = ''
    for i, temp_res in enumerate(temp_seq):

        # Append structure list
        if len(missing_residues_secondary) > 0 and str(i) in missing_residues_secondary[:,0]:
            secondary_seq += '-'
        else:
            secondary_seq += temp_res
                    
    # Read lines from reference
    ref_lines = [line for line in open(reference_pir_fn, 'r').readlines() if line != '\n']
    secondary_ref_lines = [line for line in open(secondary_pir_fn, 'r').readlines() if line != '\n']

    # Open file obj
    w = open(ali_fn, 'w')

    # Write stucture portion
    w = _write_struc_section(w, struc_seq, ref_lines)
    w = _write_struc_section(w, secondary_seq, secondary_ref_lines)

    # Write sequence portion
    w = _write_seq_section(w, temp_seq, ref_lines)
    w.close()


def _sequence_to_file(seq):
    d = math.floor(len(seq) / 75)
    if d == 0:
        d = 1
    seq_75 = []
    for i in range(d):
        seq_75.append(seq[i*75:i*75+75])
    seq_75.append(seq[i*75+75:])

    return seq_75

def _write_struc_section(writer_obj, struc_seq, ref_lines):
    for line in ref_lines[:2]:
        writer_obj.write(line)

    struc_lines = _sequence_to_file(struc_seq)
    for line in struc_lines[:-1]:
        writer_obj.write(line + '\n')
    
    writer_obj.write(struc_lines[-1]+'*\n')

    return writer_obj

def _write_seq_section(writer_obj, temp_seq, ref_lines):
    writer_obj.write(ref_lines[0][:-1] + '_fill\n')
    writer_obj.write('sequence:::::::::\n')
    
    seq_lines = _sequence_to_file(temp_seq)
    for line in seq_lines[:-1]:
        writer_obj.write(line + '\n')
    
    writer_obj.write(seq_lines[-1]+'*\n')

    return writer_obj
