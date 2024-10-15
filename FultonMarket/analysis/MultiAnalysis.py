# Imports
import os, sys, math, glob
from datetime import datetime
import netCDF4 as nc
import numpy as np
from pymbar import timeseries, MBAR
import scipy.constants as cons
import mdtraj as md
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FultonMarketAnalysis import FultonMarketAnalysis


fprint = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'


class MultiAnalysis():
    """
    """
    
    def __init__(self, input_dir: str or List, input_pdb_dir: str or List):
        """
        Initialize object
        
        Parameters:
        -----------
            input_dir (str or List[str]):
                String path to directory where all the subdirectories of FultonMarket output are located or list of string paths to all the subdirectories. 
            
            input_pdb_dir (str or List[str]):
                String path to directory where all the pdb files are located or list of string paths to all the pdb files
        """
    
        # Set attributes
        if type(input_dir) == str:
            self.input_dirs = os.listdir(input_dir)
        else:
            self.input_dirs = input_dir
        assert check_paths_exist(self.input_dirs)
        fprint(f'Found input directories: {self.input_dirs}')
            
        self.drugs = [dir.split('/')[-1] for dir in self.input_dirs]
        
        if type(input_pdb_dir) == str:
            self.input_pdbs = [os.path.join(input_pdb_dir, drug + '.pdb') for drug in self.drugs]
        else:
            self.input_pdbs = input_pdb_dir
        assert check_paths_exist(self.input_pdbs)
        fprint(f'Found input pdbs: {self.input_pdbs}')

        
               
    def load(self, skip: int=0, resample: bool=False, n_samples: int=1000, equilibration_method: str='PCA'):
        """
        Pass-through arguments to FultonMarketAnalysis and FultonMarketAnalysis.importance_resampling
        """

        # Iterate through input dirs and pdbs to load objs
        self.analysis_objs = []
        for (input_dir, pdb)  in zip(self.input_dirs, self.input_pdbs):
            if resample:
                self.analysis_objs.append(FultonMarketAnalysis(input_dir, pdb, skip).importance_resampling(n_samples, equilibration_method))
            else:
                analysis = FultonMarketAnalysis(input_dir, pdb, skip)
                analysis.equilibration_method = equilibration_method
                analysis._determine_equilibration()
                self.analysis_objs.append(analysis)
            fprint(f'Successfully loaded {input_dir}')
                   
               
               
    def plot_energy_overlap(self, state_no: int or List[int], colors: List[str], xlim: tuple=None, legend_pos: tuple=(1,1), figsize: tuple=(10,5)):
        """
        Plot the energy distributions of all replica exchange simulations in a certain state(s).
        
        Parameters:
        -----------
            state_no (int or List[int]):
                Integer or state or list of states to plot energy distributions.
        """
        # Set state_nos
        if type(state_no) is int:
            state_no = [state_no]
        
        # Get state_energies  
        state_energies = [self._get_state_energies(s) for s in state_no]
               

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        for i, (state, energies) in enumerate(zip(state_no, state_energies)):
            for j, (drug, e) in enumerate(zip(self.drugs, energies)):
                sns.kdeplot(e, label=f'{drug}', ax=ax, color=colors[j], alpha=0.5, fill=True)

        legend_without_duplicate_labels(ax, legend_pos)
        ax.set_xlabel('Energy kJ/mol')
        if xlim is not None:
            ax.set_xlim(xlim)
        plt.show()
               
               
    
    def _get_state_energies(self, state_no):
        
        # Iterate through objs
        energies = []
        for obj in self.analysis_objs:
            energies.append(obj.get_state_energies(state_no))
               
        return energies
        
               

@staticmethod
def legend_without_duplicate_labels(ax, legend_pos):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=legend_pos)
               
               
               
@staticmethod
def check_paths_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    
    return True
