#YANK BP
import textwrap, yaml, yank, sys, os, glob, shutil, openmm, rdkit
#import nglview
import openmmtools as mmtools
import yank.utils
import numpy as np
from yank.experiment import YankLoader, YankDumper
import netCDF4 as nc
import matplotlib.pyplot as plt
from mdtraj.formats.dcd import DCDTrajectoryFile
import mdtraj as md
import MDAnalysis as mda
#OPENFF
from openff.units import Quantity, unit
from openmm import unit as openmm_unit
from pdbfixer import PDBFixer
from openff.toolkit import ForceField, Molecule, Topology
from openbabel import openbabel


def seperate_crys_using_MDA(crys_pdb_fn, ligand_string, receptor_string='protein', ligand_pdb_fn='ligand.pdb', receptor_pdb_fn='receptor.pdb'):
    """Takes a crystal structure as input, splitting into receptor and ligand pdb files
    crys_pdb_fn: crystal structure to be seperated
    ligand_string: MDAnalysis selection string for ligand
    receptor_string: MDAnalysis selection string for receptor (default 'protein')
    ligand_pdb_fn: Write path for MDAnalysis ligand (default 'ligand.pdb')
    receptor_pdb_fn: Write path for MDAnalysis receptor (default 'receptor.pdb')
    """
    u = mda.Universe(crys_pdb_fn)
    all_atoms = u.select_atoms('all')
    translate_vector = -np.min(all_atoms.positions, axis=0)
    all_atoms.translate(translate_vector + 20) # padding from origin may fix the issue
    receptor = all_atoms.select_atoms(receptor_string)
    ligand = all_atoms.select_atoms(ligand_string)
    receptor.write(receptor_pdb_fn)
    ligand.write(ligand_pdb_fn)
    return receptor_pdb_fn, ligand_pdb_fn


def obabel_conversion(mol_fns, formats=['INFER','INFER']):
    """Convert a file to another format using openbabel
    mol_fns: 2-list of filenames (in, out)
    formats: 2-list of formats (in, out), default is to infer format from file extensions
    """
    #Assertion checks
    assert len(mol_fns) == 2 and len(formats) == 2
    #Infer file extensions as obabel formats
    for i, v in enumerate(formats):
        if v == 'INFER':
            #retrieve the file extension, and remove the preceding .
            formats[i] = os.path.splitext(mol_fns[i])[-1][1:]
    #Obabel conversion block
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(*formats)
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, mol_fns[0])
    mol.AddHydrogens()
    print(mol.NumAtoms())
    print(mol.NumBonds())
    print(mol.NumResidues())
    obConversion.WriteFile(mol, mol_fns[1])

    return mol_fns[1]


def protonate_with_pdb2pqr(protein_fn, protein_H_fn, at_pH=7):
    """Protonate the given structure using pdb2pqr30
    protein_fn: structure to be protonated
    protein_H_fn: filepath for protonated receptor (as pdb)
                    pqr output of pdb2pqr is inferred as protein_H_fn with a pqr extension instead of pdb
    at_pH: pH for protonation (default 7)
    """
    protein_pqr_fn = os.path.splitext(protein_H_fn)[0] + '.pqr'
    my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {protein_H_fn} --with-ph {at_pH} {protein_fn} {protein_pqr_fn}'
    print('Protanting using command line')
    print(my_cmd)
    os.system(my_cmd)
    return protein_H_fn

def insert_molecule_and_remove_clashes(topology: Topology, insert: Molecule, radius: Quantity = 1.5 * unit.angstrom,
                                       keep: list[Molecule] = [],) -> Topology:
    """
    Add a molecule to a copy of the topology, removing any clashing molecules.

    The molecule will be added to the end of the topology. A new topology is
    returned; the input topology will not be altered. All molecules that
    clash will be removed, and each removed molecule will be printed to stdout.
    Users are responsible for ensuring that no important molecules have been
    removed; the clash radius may be modified accordingly.

    Parameters
    ==========
    top
        The topology to insert a molecule into
    insert
        The molecule to insert
    radius
        Any atom within this distance of any atom in the insert is considered
        clashing.
    keep
        Keep copies of these molecules, even if they're clashing
    """
    # We'll collect the molecules for the output topology into a list
    new_top_mols = []
    # A molecule's positions in a topology are stored as its zeroth conformer
    insert_coordinates = insert.conformers[0][:, None, :]
    for molecule in topology.molecules:
        if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep):
            new_top_mols.append(molecule)
            continue
        molecule_coordinates = molecule.conformers[0][None, :, :]
        diff_matrix = molecule_coordinates - insert_coordinates

        # np.linalg.norm doesn't work on Pint quantities 😢
        working_unit = unit.nanometer
        distance_matrix = (
            np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit
        )

        if distance_matrix.min() > radius:
            # This molecule is not clashing, so add it to the topology
            new_top_mols.append(molecule)
        else:
            print(f"Removed {molecule.to_smiles()} molecule")

    # Insert the ligand at the end
    new_top_mols.append(ligand)

    # This pattern of assembling a topology from a list of molecules
    # ends up being much more efficient than adding each molecule
    # to a new topology one at a time
    new_top = Topology.from_molecules(new_top_mols)

    # Don't forget the box vectors!
    new_top.box_vectors = topology.box_vectors
    return new_top

def describe_state(state: openmm.State, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")

def make_openmm_simulation(interchange):
    """Construct an openn simulation object from an openff interachange"""
    # Construct and configure a Langevin integrator at 300 K with an appropriate friction constant and time-step
    integrator = openmm.LangevinIntegrator(300 * openmm_unit.kelvin, 1 / openmm_unit.picosecond, 0.001 * openmm_unit.picoseconds)
    # Under the hood, this creates *OpenMM* `System` and `Topology` objects, then combines them together
    simulation = interchange.to_openmm_simulation(integrator=integrator)
    # Add a reporter to record the structure every 10 steps
    dcd_reporter = openmm.app.DCDReporter(f"trajectory.dcd", 1000)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    #Evaluate and Report pre-minimized energy
    describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Original state")
    #Minimize the structure
    simulation.minimizeEnergy()
    #Evaluate and Report post-minimized energy
    describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Minimized state")
    simulation.context.setVelocitiesToTemperature(300 * openmm_unit.kelvin)
    return simulation

def run_for_walltime(simulation, num_minutes):
    simulation.runForClockTime(num_minutes * openmm_unit.minute)

def run_for_steps(simulation, nsteps):
    simulation.step(nsteps)