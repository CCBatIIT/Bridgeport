import textwrap, sys, os
import numpy as np
#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *

class Joiner():
    """
    A Class for adding two OpenMM system, topology, and position sets. This tool is intended for use cases such as adding an OpenFF Ligand System to an OpenMM Environment setup, allowing for comprehensive simulations involving both small molecules and complex biological systems.
    
    Parameters:
    -----------
        Ligand_Set: 
            (System, Topology, Positions) as a 3-tuple representing the ligand to be added.
            
        Receptor_Set: 
            (System, Topology, Positions) as a 3-tuple representing the receptor or environment.
    
    Attributes:
    ------------
        lig_sys, lig_top, lig_pos: Components of the Ligand set, comprising the OpenMM system, topology, and positions, respectively.
        
        rec_sys, rec_top, rec_pos: Components of the Receptor set, comprising the OpenMM system, topology, and positions, respectively.
        
    Methods:
    ---------
        init(self, Ligand_Set, Receptor_Set): 
            Initializes the Joiner instance with the provided Ligand and Receptor sets.
            
        main(self): 
            Combines the ligand and receptor sets into a single set of system, topology, and positions. Returns a 3-tuple of these combined components.
       
        _join_two_topologies(self, tops: tuple, poss: tuple): 
            Joins two OpenMM topologies and their associated positions. The first index is added to the second. Parameters include a two-tuple of topologies and a two-tuple of positions.
            
        _join_two_systems(self, sys1: System, sys2: System): 
            Joins two OpenMM systems by adding the elements of system1 to system2. This method is intended to combine a ligand system with a receptor system. Parameters include sys1 (the system to be added) and sys2 (the system to which sys1 is added).
    """
    
    def __init__(self, Ligand_Set, Receptor_Set):
        assert type(Ligand_Set) == type(Receptor_Set) and len(Ligand_Set) == len(Receptor_Set)
        assert False not in [type(Ligand_Set[i]) == type(Receptor_Set[i]) for i in range(3)]

        self.lig_sys, self.lig_top, self.lig_pos = Ligand_Set
        self.rec_sys, self.rec_top, self.rec_pos = Receptor_Set


    def main(self):
        """
        Returns
            Joined_Set: (System, Topology, Positions) as 3-tuple
        """
        comp_top, comp_positions = self._join_two_topologies((self.rec_top, self.lig_top), (self.rec_pos, self.lig_pos))
        comp_sys = self._join_two_systems(self.lig_sys, self.rec_sys)
        return comp_sys, comp_top, comp_positions
    
    def _join_two_topologies(self, tops: tuple, poss: tuple):
        """
        Joins two topologies by adding the second to the first
        INDEX 1 IS ADDED TO INDEX 0
        Parameters:
            tops: A two-tuple of openmm Topologies
            poss: A two-tuple of positions
        Returns
            2-Tuple: Topology, positions of the joined atoms
        """
        assert len(tops) == 2 and len(poss) == 2
        modeller = Modeller(tops[0], poss[0])
        modeller.add(tops[1], poss[1])
        return modeller.topology, modeller.positions

    def _join_two_systems(self, sys1: System, sys2: System):
        """
        Joins Two Openmm systems by adding the elements of system 1 to system 2
        Intended use in this notebook is join_two_systems(ligand_sys, receptor_sys)
        
        Parameters:
            sys1 - The openmm system to be added to sys2
            sys2 - The openmm system to have sys1 added to
        Returns
            System - The combined system
        """
        #Particles
        new_particle_indices = []
        for i in range(sys1.getNumParticles()):
            new_particle_indices.append(sys2.addParticle(sys1.getParticleMass(i)))
        
        #Contstraints (mostly wrt hydrogen distances)
        for i in range(sys1.getNumConstraints()):
            params = sys1.getConstraintParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2.addConstraint(*params)
        
        #NonBonded
        sys1_force_name = 'Nonbonded force'
        sys2_force_name = 'NonbondedForce'

        try:
            force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
            sys1_force = sys1.getForces()[force_ind]
        except:
            force_ind = [force.getName() for force in sys1.getForces()].index(sys2_force_name)
            sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumParticles()):
            params = sys1_force.getParticleParameters(i)
            sys2_force.addParticle(*params)
        
        for i in range(sys1_force.getNumExceptions()):
            params = sys1_force.getExceptionParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2_force.addException(*params)
    
        #Torsion
        sys1_force_name = 'PeriodicTorsionForce'
        sys2_force_name = 'PeriodicTorsionForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumTorsions()):
            params = sys1_force.getTorsionParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            params[2] = new_particle_indices[params[2]]
            params[3] = new_particle_indices[params[3]]
            sys2_force.addTorsion(*params)
    
        #Angle
        sys1_force_name = 'HarmonicAngleForce'
        sys2_force_name = 'HarmonicAngleForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumAngles()):
            params = sys1_force.getAngleParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            params[2] = new_particle_indices[params[2]]
            sys2_force.addAngle(*params)
    
        #Bond
        sys1_force_name = 'HarmonicBondForce'
        sys2_force_name = 'HarmonicBondForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumBonds()):
            params = sys1_force.getBondParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2_force.addBond(*params)
        
        return sys2