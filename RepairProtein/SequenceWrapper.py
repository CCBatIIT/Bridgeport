import math
import numpy as np


class SequenceWrapper():
    """
    This class is able to identify the missing holes in a target sequence, when compared to a template sequence. 

    Attributes:
    -----------
        temp_seq (str): String sequence that is the template. e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.

        tar_seq (str): String sequence that is should match the template. e.g. 'CDGHLMNOTVXY'

        missing_residues (np.array): 2-D Array of indices of missing sequence entries and what the missing entry is. e.g. [[0, 'A'], [1, 'B'], [4, 'E'], ....]

        
    """

    def __init__(self, template_seq: str, target_seq: str):
        """ 
        Initialize the SequenceWrapper class

        Parameters:
        -----------
            template_sequence (str):
                String sequence that is the template. e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.
            
            target_sequence (str):
                String sequence that is should match the template. e.g. 'CDGHLMNOTVXY'
        """

        # Assert
        assert len(template_seq) >= len(target_seq), f"Length of template sequence {len(template_seq)} should not be shorter than target sequence {len(target_seq)}."

        # Initialize objects:
        self.temp_seq = template_seq
        self.tar_seq = target_seq
        self.missing_residues = []

    def find_missing_residues(self, verbose: bool=True, tails: bool=False):
        """
        Find the missing residues in the target sequence.

        Parameters:
        -----------
            verbose (bool):
                If True, will print missing residues and indices of missing residues to console. Default is True.
            
            tails (bool):
                If True, add missing residues to N and C termini.
        """ 
        # Find the missing residues
        self.missing_residues, self.mutated_residues, self.term_residues = _find_missing_residues(tar_seq=self.tar_seq, temp_seq=self.temp_seq, verbose=verbose)

    def write_alignment_file(self, ali_fn: str, reference_pir_fn: str):
        """
        Write an alignment file for UCSF Modeller. This method puts the sequence information in the format as specified by an example from https://salilab.org/modeller/wiki/Missing_residues

        Parameters:
        -----------
            ali_fn (str):
                String path to write .ali file.

            reference_pir_fn (str):
                String path to get structural information from. This .pir file should be created from the .pdb that is in need of mending.
        """
        # Write the alignment file
        _write_alignment_file(temp_seq=self.temp_seq, missing_residues=self.missing_residues, ali_fn=ali_fn, reference_pir_fn=reference_pir_fn)
        

"""Sequence Methods. These methods are used by the SequenceWrapper class to effectively map the indices and identity of missing residues of a protein sequence

Methods:
--------
    _find_missing_residues: Iterates through sequences to find missing residues and locations
    _write_alignment_file: Write the input for UCSF Modeller. 

"""


def _find_missing_residues(temp_seq: str, tar_seq: str, verbose: bool=True):
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
        tar_start_ind = temp_seq.index(tar_seq[:3])
        missing_start = temp_seq[:tar_start_ind]
        for ind, res in enumerate(missing_start):
            missing_residues.append([ind, res])
        
        tar_seq = missing_start + tar_seq
        # print('targ:', tar_seq)
        # print('temp:', temp_seq, '\n')

        # Iterate through remaining residues
        counter = 0
        while temp_seq[:len(tar_seq)] != tar_seq:
            for i in range(len(tar_seq)):
                # Append if not matching
                temp_res = temp_seq[i]
                tar_res = tar_seq[i]
                if tar_res != temp_res:
                    # print(i)
                    # print(tar_seq[i], tar_res, temp_seq[i], temp_res)
                    # print('targ:', tar_seq)
                    # print('temp:', temp_seq)

                    # Check for point mutation
                    if tar_seq[i+1:i+5] == temp_seq[i+1:i+5]:
                        # print('     ', tar_seq[:i], temp_res, tar_seq[i+1:])
                        # print('Point', '\n')
                        tar_seq = tar_seq[:i] + temp_res + tar_seq[i+1:]
                        mutated_residues.append([i - tar_start_ind, temp_res])

                    # Check for deletion
                    else:
                        # print('     ', tar_seq[:i], temp_res, tar_seq[i+1:])
                        # print('Del', '\n')
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
        elif verbose:
            print('Missing Residues:', missing_residues)
            print('Mutated Residues:', mutated_residues)

        return [np.array(missing_residues), np.array(mutated_residues), term_residues]

def _write_alignment_file(temp_seq: str, missing_residues: np.array, ali_fn: str, reference_pir_fn: str):
    """
    Write an alignment file for UCSF Modeller. This method puts the sequence information in the format as specified by an example from https://salilab.org/modeller/wiki/Missing_residues

    Parameters:
    -----------
        ali_fn (str):
            String path to write .ali file.

        reference_pir_fn (str):
            String path to get structural information from. This .pir file should be created from the .pdb that is in need of mending.
    """

    def _sequence_to_file(seq):
        d = math.floor(len(seq) / 75)
        seq_75 = []
        for i in range(d):
            seq_75.append(seq[i*75:i*75+75])
        seq_75.append(seq[i*75+75:])

        return seq_75

    # Make sequence lists
    struc_seq = ''
    for i, temp_res in enumerate(temp_seq):

        # Append structure list
        if str(i) in missing_residues[:,0]:
            struc_seq += '-'
        else:
            struc_seq += temp_res
                    
    # Read lines from reference
    ref_lines = [line for line in open(reference_pir_fn, 'r').readlines() if line != '\n']

    # Open file obj
    w = open(ali_fn, 'w')

    # Write stucture portion
    for line in ref_lines[:2]:
        w.write(line)

    struc_lines = _sequence_to_file(struc_seq)
    for line in struc_lines[:-1]:
        w.write(line + '\n')
    
    w.write(struc_lines[-1]+'*\n')

    # Write sequence protein
    w.write(ref_lines[0][:-1] + '_fill\n')
    w.write('sequence:::::::::\n')
    
    seq_lines = _sequence_to_file(temp_seq)
    for line in seq_lines[:-1]:
        w.write(line + '\n')
    
    w.write(seq_lines[-1]+'*\n')
    w.close()