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
    """
    Takes a crystal structure as input, splitting into receptor and ligand pdb files.
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


def determine_restrained_residues(structure_file, n_closest, ligand_string):
    """
    Determines the 'n_closest' residues to the selection given by 'ligand_string' as measured by the distance of alpha
    carbons to the center of mass of the selection.  Assembles a selection string for the determined residues.
    EX.
    >>>determine_restrained_residues("SOME.pdb", 3, "resname lig")
    (resname PHE and resid 32) or (resname ALA and resid 47) or (resname GLY and resid 54)
    """
    u = mda.Universe(structure_file)
    protein_CAs = u.select_atoms('protein and name CA')
    uq = u.select_atoms(ligand_string)
    uq_com = uq.center_of_mass()
    #Build a list of resnames resids and distances (sort by distance)
    residues = []
    for atom in protein_CAs:
        dist = np.sqrt(np.sum((atom.position - uq_com)**2))
        residues.append([atom.resname, atom.resindex, dist])
    residues = sorted(residues, key = lambda x: x[2])
    #Craft the restraint string
    restraint_string = ''
    for res in residues[:n_closest]:
        restraint_string += f'(resname {res[0]} and resid {res[1]}) or '
    restraint_string = restraint_string[:-4]
    return restraint_string


def obabel_conversion(mol_fns, formats=['INFER','INFER'], add_Hs=True, rewrite_with_Hs=True):
    """Convert a file to another format using openbabel.  Neither add_Hs, nore rewrite_with_Hs should be True if the input has hydrogens.
    mol_fns: 2-list of filenames (in, out)
    formats: 2-list of formats (in, out), default is to infer format from file extensions
    add_Hs: Bool: default is to attempt to add hydrogens to the input
    rewrite_with_Hs: Bool: default is to rewrite the input in the same format, with hydrogens added
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
    if add_Hs:
        mol.AddHydrogens()
    print(mol.NumAtoms(), 'Atoms', mol.NumBonds(), 'Bonds', mol.NumResidues(), 'Residues')
    obConversion.WriteFile(mol, mol_fns[1])
    #Rewrite original format with Hydrogens (if necessary)
    if rewrite_with_Hs:
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(formats[0], formats[0])
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, mol_fns[0])
        mol.AddHydrogens()
        print(mol.NumAtoms(), 'Atoms', mol.NumBonds(), 'Bonds', mol.NumResidues(), 'Residues')
        path_elements = os.path.splitext(mol_fns[0])
        old_format_with_Hs = path_elements[0] + '_H' +path_elements[-1]
        obConversion.WriteFile(mol, old_format_with_Hs)
    
    return mol_fns[1]


def protonate_with_pdb2pqr(protein_fn, protein_H_fn=None, at_pH=7):
    """Protonate the given structure using pdb2pqr30
    protein_fn: structure to be protonated
    protein_H_fn: filepath for protonated receptor (as pdb)
                    pqr output of pdb2pqr is inferred as protein_H_fn with a pqr extension instead of pdb
    at_pH: pH for protonation (default 7)
    """
    if protein_H_fn is None:
        protein_H_fn = os.path.splitext(protein_fn)[0] + '_H' + os.path.splitext(protein_fn)[-1]
    protein_pqr_fn = os.path.splitext(protein_H_fn)[0] + '.pqr'
    my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {protein_H_fn} --with-ph {at_pH} {protein_fn} {protein_pqr_fn}'
    print('Protanting using command line')
    print(my_cmd)
    os.system(my_cmd)
    print('Done')
    return protein_H_fn


def PDBFixer_solvation(in_pdbfile, solvated_file=None, padding=1.5, ionicStrength=0.15):
    """
    Generates a solvated system using PDBFixer.
    in_pdbfile: structure file: structure to be placed in a solvation box
    solvated_file: string: filename to save output.  Default is the add '_solvated' between the body and extension of the input file name
    padding: float or int: minimum nanometer distance between the boundary and any atoms in the input.  Default 1.5 nm = 15 A
    ionicStrength: float (not int as thats a lot of ions :) : molar strength of ionic solution. Default 0.15 M = 150 mmol
    """
    fixer = PDBFixer(in_pdbfile)
    fixer.addSolvent(padding=padding * openmm_unit.nanometer, ionicStrength=ionicStrength * openmm_unit.molar)
    # ADD PDBFixer hydrogens and parsing crystal structures (Hydrogens with pdb2pqr30 at the moment)
    if solvated_file == None:
        solvated_file = os.path.splitext(in_pdbfile)[0] + '_solvated' + os.path.splitext(in_pdbfile)[-1]
    with open(solvated_file, "w") as f:
        openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return solvated_file


def just_the_solvent(ligand_solvated_path, ligand_resname, solvent_phase_fn='solvent_phase.pdb'):
    u = mda.Universe(ligand_solvated_path)
    solvent = u.select_atoms(f'not {ligand_resname}')
    solvent.write(solvent_phase_fn)
    return solvent_phase_fn


def write_the_yaml(complex_fns, solvent_fns, ligand_string, out_dir, restraint_string):
    """
    Write a general yaml file that can be adjusted to run a series of yank simulations.
    This file runs one simulation, but is easyily iterable to create many simulations.
    complex_fns: 2-tuple or 2-list: structure, topology pair of file names
                                    typically (.pdb, .xml) or (.prmtop, .inpcrd) or (.gro, .top)
    solvent_fns: 2-tuple or 2-list: structure, topology pair of file names
                                    typically (.pdb, .xml) or (.prmtop, .inpcrd) or (.gro, .top)    
    ligand_string: string:  mdtraj selection string to identify the ligand
                            (the thing you want to phase in and out of existance)
    out_dir: string: Yank's storage directory.  Find most data at out_dir/experiments/
    restraint_string: string: string created by determine_restrained_residues to define the
                              residues of the protein that will be restrained to the ligand
    """
    yaml_contents = f"""---
experiments:
  protocol: absolute-binding
  restraint:
    type: FlatBottom
    restrained_receptor_atoms: {restraint_string}
    restrained_ligand_atoms: all
    spring_constant: 10.0*kilocalories_per_mole/(angstrom**2)
    well_radius: 8.0*angstroms
  system: rec-lig
options:
  default_nsteps_per_iteration: 500
  default_number_of_iterations: 500
  default_timestep: 1.0*femtosecond
  minimize: no
  number_of_equilibration_iterations: 0
  output_dir: {out_dir}
  platform: fastest
  pressure: 1.0*atmosphere
  resume_simulation: yes
  temperature: 300*kelvin
  verbose: yes
samplers:
    replica-exchange:
        type: ReplicaExchangeSampler
        replica_mixing_scheme: swap-neighbors
        online_analysis_interval: null
protocols:
  absolute-binding:
    complex:
      alchemical_path: auto
      trailblazer_options:
        bidirectional_redistribution: yes
        constrain_receptor: false
        distance_tolerance: 0.05
        n_equilibration_iterations: 0
        n_samples_per_state: 100
        reversed_direction: yes
        thermodynamic_distance: 1
    solvent:
      alchemical_path: auto
      trailblazer_options:
        bidirectional_redistribution: yes
        constrain_receptor: false
        distance_tolerance: 0.05
        n_equilibration_iterations: 0
        n_samples_per_state: 100
        reversed_direction: yes
        thermodynamic_distance: 1
solvents:
  PME:
    nonbonded_cutoff: 8.0*angstroms
    nonbonded_method: PME
systems:
  rec-lig:
    ligand_dsl: {ligand_string}
    phase1_path:
    - {complex_fns[0]}
    - {complex_fns[1]}
    phase2_path:
    - {solvent_fns[0]}
    - {solvent_fns[1]}
    solvent: PME"""
    return yaml_contents


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
    new_top_mols.append(insert)

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


def make_simulation_and_writeout(interchange, phase):
    sim = make_openmm_simulation(interchange)
    with open(f'{phase}_final.xml', "w") as xml_file:
        xml_file.write(openmm.XmlSerializer.serialize(sim.system))
    
    with open(f'{phase}_final.pdb', "w") as pdb_file:  
        openmm.app.PDBFile.writeFile(sim.topology,
                                     sim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                                     file=pdb_file,
                                     keepIds=True)


def run_for_walltime(simulation, num_minutes):
    simulation.runForClockTime(num_minutes * openmm_unit.minute)


def run_for_steps(simulation, nsteps):
    simulation.step(nsteps)