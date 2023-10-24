import os, sys
import openmm as mm
from openmm.app import *
from openmm.unit import *
from openmm.openmm import *
import numpy as np
from typing import List
import MDAnalysis as mda
import json

"""
This script adds restaints to atoms in the openmm object, and then serializes to an .xml for further simulation usage.
"""

# Directories
pdb_dir = sys.argv[1]
prmtop_dir = '../equil/para/post_equil/'
# xml_dir = '../systems/xml'
water_dir = '../water/'
ligands_dir = '../ligands'
# min_pdb_dir = f'../systems/min_pdb'

# Restrain method
def restrain_atoms(sys: mm.System, atom_inds = List[int], atom_init_pos = np.array, k: float=300.0):
     
    rest = CustomExternalForce('k*periodicdistance(x,y,z,x0,y0,z0)^2')
    rest.addGlobalParameter('k', k) # kJ/mol/nm^2
    rest.addPerParticleParameter('x0')
    rest.addPerParticleParameter('y0')
    rest.addPerParticleParameter('z0')

    for (atom_ind, init_xyz) in zip(atom_inds, atom_init_pos):
        x0, y0, z0 = init_xyz*0.1 # Convert to nm
        rest.addParticle(atom_ind, [x0, y0, z0])

    print(f'No. of Atoms Restrained: {rest.getNumParticles()}')
    sys.addForce(rest)

    return sys

# Iterate through .pdb files
for pdb_fn in os.listdir(pdb_dir):
    pdb_path = f'{pdb_dir}/{pdb_fn}'
    # min_pdb_path = f'{min_pdb_dir}/{pdb_fn}'
    drug = pdb_fn.split('_')[0]
    centroid_no = pdb_fn.split('_')[1]
    pose_no = pdb_fn.split('.')[0].split('_')[-1]
    prmtop_path = f'{prmtop_dir}/{drug}_{centroid_no}_1.prmtop'
    print(prmtop_path)
    xml_path = f'{pdb_dir}/{drug}.xml'
    
    # If drug-system xml does not exist, make xml
    if not os.path.exists(xml_path):
        print(pdb_fn)
        """
        Protein System
        """

        # Find CA atom inidices and positions with mda
        u = mda.Universe(f'{pdb_dir}/{pdb_fn}')
        prot = u.select_atoms('protein and resid 82:114 154:192 244:289 326:347 and not name H*')
        atom_inds = [ind for ind in prot.atoms.indices]
        init_pos = prot.positions

        # Load system from .pdb and forcefield
        pdb = PDBFile(pdb_path)
        prmtop = AmberPrmtopFile(prmtop_path)
        system = prmtop.createSystem(nonbondedMethod=PME,
                                    nonbondedCutoff=1.2*nanometer,
                                    constraints=HBonds,
                                    rigidWater=True,
                                    ewaldErrorTolerance=0.0005)    

        print(f'No. Atoms: {system.getNumParticles()}\nNo. Selected Atoms: {prot.n_atoms}')

        # Set restraint
        system = restrain_atoms(system, atom_inds, init_pos)
        integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds)
        simulation = Simulation(pdb.getTopology(), system, integrator)
        simulation.context.setPositions(pdb.getPositions()) 

        # Minimize structure
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        # simulation.minimizeEnergy()
        # print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

        # Save pdb
        # with open(min_pdb_path, 'w') as f:
        #     print('saving pdb')
        #     PDBFile.writeFile(simulation.topology,
        #             simulation.context.getState(getPositions=True).getPositions(),
        #             file=f,
        #             keepIds=True)
        #     f.close()

        # Save .xml
        if not os.path.exists(xml_path):
            with open(xml_path, 'w') as f:
                f.write(XmlSerializer.serialize(system))

        """
        Solvated Ligand 
        """

    solv_xml_path = f'{pdb_dir}/solvent_{drug}.xml'
    solv_pdb_path = f'{pdb_dir}/solvent_{drug}.pdb'
    solv_prmtop_path = f'{water_dir}/para/post_equil/solvent_{drug}.prmtop'
    if not os.path.exists(solv_xml_path):

        # Change JSON information for pdb2amber
        if not os.path.exists(solv_prmtop_path):
            input_json = 'input.json'
            with open(input_json, 'r') as f:
                data = json.load(f)
                f.close()

            data['fname_pdb'] = solv_pdb_path
            data['fname_prmtop'] = solv_prmtop_path
            try:
                data['fname_ff'][3] = f'/{drug}.ff.xml'
            except:
                data['fname_ff'].append(f'{ligands_dir}/xml/{drug}.ff.xml')
        
            with open(input_json, 'w') as f:
                json.dump(data, f)
                f.close()

            # Run pdb2amber
            print('\nWRITING PRMTOP\n')
            os.system(f'python pdb2amber.py -i {input_json}')

        # Load system from .pdb and forcefield
        pdb = PDBFile(solv_pdb_path)
        prmtop = AmberPrmtopFile(solv_prmtop_path)
        system = prmtop.createSystem(nonbondedMethod=PME,
                                    nonbondedCutoff=1.2*nanometer,
                                    constraints=HBonds,
                                    rigidWater=True,
                                    ewaldErrorTolerance=0.0005)   

        integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds)
        simulation = Simulation(pdb.getTopology(), system, integrator)
        simulation.context.setPositions(pdb.getPositions()) 

        # Minimize structure
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        # simulation.minimizeEnergy()
        # print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

        # # Save pdb
        # with open(solv_pdb_path, 'w') as f:
        #     print('saving pdb')
        #     PDBFile.writeFile(simulation.topology,
        #             simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
        #             file=f,
        #             keepIds=True)
        #     f.close()

        # Save .xml
        if not os.path.exists(solv_xml_path):
            with open(solv_xml_path, 'w') as f:
                f.write(XmlSerializer.serialize(system))
          
    if os.path.exists(xml_path) and os.path.exists(solv_xml_path):
        break
