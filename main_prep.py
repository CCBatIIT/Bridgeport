#YANK BP
import textwrap, sys, os, glob, shutil, openmm
#import nglview
#import openmmtools as mmtools
#import yank.utils
import numpy as np
import MDAnalysis as mda
#OPENFF
from openff.units import Quantity, unit
from openmm import unit as openmm_unit
from pdbfixer import PDBFixer
import openff
import openff.interchange as openff_ic
from openff.toolkit import ForceField, Molecule, Topology
from openbabel import openbabel
from datetime import datetime

def change_resname(pdb_file_in, pdb_file_out, resname_in, resname_out):
    """
    Changes a resname in a pdb file by changing all occurences of resname_in to resname_out
    
    """

    with open(pdb_file_in, 'r') as f:
        lines = f.readlines()
    print('Effected Lines:')
    eff_lines = [line for line in lines if resname_in in line]
    for line in eff_lines:
        print(line, "-->", line.replace(resname_in, resname_out))
    user_input = input("Confirm to make these changes [y/n] :")
    if user_input == 'y':
        lines = [line.replace(resname_in, resname_out) for line in lines]
        with open(pdb_file_out, 'w') as f:
            f.writelines(lines)
        return pdb_file_out
    else:
        print('Aborting....')
        return None

job_inputs = {'working_dir_name': 'gpcr_join_testing',
              'ligand_resname': 'resname V4O',
              'crystal_pdb_fn': '5c1m_V4O.pdb',
              'build_modes': ['MEM', 'SOL']}


class Simulation_Preparer():
    def __init__(self, job_inputs):
        #Declare Filenames
        self.job_inputs = job_inputs
        working_dir_name = job_inputs['working_dir_name']
        self.ligand_resname = job_inputs['ligand_resname']
        self.crystal_pdb_fn = job_inputs['crystal_pdb_fn']

        self.abs_work_dir = os.path.join(os.getcwd(), working_dir_name)
        if not os.path.isdir(self.abs_work_dir):
            os.mkdir(self.abs_work_dir)

    
    def seperate_crys_using_MDA(self,
                                crys_pdb_fn: str,
                                ligand_string: str,
                                receptor_string: str = 'protein',
                                ligand_pdb_fn: str = 'ligand_crys.pdb',
                                receptor_pdb_fn: str = 'receptor_crys.pdb'):
        """
        Loads the provided crystal structure, and seperates the atoms of resname LIGAND STRING into their own file
        Arguments:
            crys_pdb_fn: crystal structure to be seperated
            ligand_string: MDAnalysis selection string for ligand
            receptor_string: MDAnalysis selection string for receptor (default 'protein')
            ligand_pdb_fn: Write path for MDAnalysis ligand (default 'ligand.pdb')
            receptor_pdb_fn: Write path for MDAnalysis receptor (default 'receptor.pdb')
        Returns:
            receptor_pdb_fn: filename of the pdb file containing the receptor only
            ligand_pdb_fn: filename of the pdb file containing the atoms from the LIGAND_STRING selection
        """
        u = mda.Universe(crys_pdb_fn)
        all_atoms = u.select_atoms('all')
        receptor = all_atoms.select_atoms(receptor_string)
        ligand = all_atoms.select_atoms(ligand_string)
        receptor.write(os.path.join(self.abs_work_dir, receptor_pdb_fn))
        ligand.write(os.path.join(self.abs_work_dir, ligand_pdb_fn))
        return os.path.join(self.abs_work_dir, receptor_pdb_fn), os.path.join(self.abs_work_dir, ligand_pdb_fn)

    def obabel_conversion(self,
                          mol_fn: str,
                          formats: list,
                          out_fn: str = 'AUTO',
                          add_Hs: bool = True,
                          rewrite_with_Hs: bool = True):
        """
        Convert a file to another format using openbabel.  Neither add_Hs, nore rewrite_with_Hs should be True if the input has hydrogens.
        Arguments:
            mol_fn: in_file_name : (THis should be infile.xxx where xxx = formats[0])
            formats: 2-list of babel formats (in, out)
            out_fn: Filename for the ligand with Hydrogens (if AUTO, it will be MOL_FN with the extension swapped to format[-1])
            add_Hs: Bool: Whether adding hydrogens to the input file should be done (default True)
            rewrite_with_Hs: Bool: Additionally rewrites the protonated ligand in the original file format (default True)
        Returns:
            str: The path of the converted file
        Example:
            obabel_conversion('ligand.pdb', ['pdb', 'sdf']) will attempt to protonate (by default) and convert a pdb file to sdf (named ligand.sdf)
        """
        #Assertion checks
        assert len(formats) == 2
        #Obabel conversion block
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(*formats)
        mol = openbabel.OBMol()
        #Find INput
        if os.path.isfile(mol_fn):
            obConversion.ReadFile(mol, mol_fn)
        elif os.path.isfile(os.path.join(self.abs_work_dir, mol_fn)):
            obConversion.ReadFile(mol, os.path.join(self.abs_work_dir, mol_fn))
        else:
            raise FileNotFoundError('mol_fn was not found')
        #Add Hydrogens
        if add_Hs:
            mol.AddHydrogens()
        print(mol.NumAtoms(), 'Atoms', mol.NumBonds(), 'Bonds', mol.NumResidues(), 'Residues')
        #Output file name parsing
        if out_fn == 'AUTO':
            out_fn = os.path.splitext(mol_fn)[0] + '.' + formats[-1]
        #Actually writeout the protonated file in the second format
        obConversion.WriteFile(mol, out_fn)
        
        #Rewrite original format with Hydrogens (if necessary)
        if rewrite_with_Hs:
            #recursively use this function to convert from format 0 to format 0 again
            org_form_new_fn = os.path.splitext(mol_fn)[0] + '_H.' + formats[0]
            org_form_wHs_fn = self.obabel_conversion(mol_fn, [formats[0], formats[0]], out_fn=org_form_new_fn, add_Hs=True, rewrite_with_Hs=False)[0]
            return (out_fn, org_form_wHs_fn)
        else:
            return (out_fn, None)

    def protonate_with_pdb2pqr(self,
                               protein_fn: str,
                               protein_H_fn: str = None,
                               at_pH=7):
        """Protonates the given structure using pdb2pqr30
        Parameters:
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
        print(f'Running {my_cmd}')
        exit_status = os.system(my_cmd)
        print(f'Done with exit status {exit_status}')
        return protein_H_fn, protein_pqr_fn

    def run_PDBFixer(self,
                     in_pdbfile: str,
                     mode: str = "MEM",
                     out_file_fn: str = None,
                     padding = 1.5,
                     ionicStrength = 0.15):
        """
        Generates a solvated and membrane-added system (depending on MODE)
        MODE = 'MEM' for membrane and solvent
        MODE = 'SOL' for solvent only
        Parameters:
            in_pdbfile: the structure to be solvated
            mode: string: must be in ['MEM', 'SOL']
            solvated_file_fn: Filename to save solvated system; default is to add '_solvated' between the body and extension of the input file name
            padding: float or int: minimum nanometer distance between the boundary and any atoms in the input.  Default 1.5 nm = 15 A
            ionicStrength: float (not int) : molar strength of ionic solution. Default 0.15 M = 150 mmol
        Returns:
            solvated_file_fn: the filename of the solvated output
        """
        assert mode in ['MEM', 'SOL']
        fixer = PDBFixer(in_pdbfile)

        if mode == 'MEM':
            fixer.addMembrane('POPC', minimumPadding=padding * openmm_unit.nanometer, ionicStrength=ionicStrength * openmm_unit.molar)
        elif mode == 'SOL':
            fixer.addSolvent(padding=padding * openmm_unit.nanometer, ionicStrength=ionicStrength * openmm_unit.molar)

        fixer.addMissingHydrogens()
        
        # ADD PDBFixer hydrogens and parsing crystal structures (Hydrogens with pdb2pqr30 at the moment)
        if out_file_fn is None:
            out_file_fn = os.path.splitext(in_pdbfile)[0] + f'_{mode}' + os.path.splitext(in_pdbfile)[-1]
        
        with open(out_file_fn, "w") as f:
            openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return out_file_fn

    def insert_lipid_molecules(self,
                               topology: Topology,
                               insert: Molecule) -> Topology:
        
        # We'll collect the molecules for the output topology into a list
        new_top_mols = []
        # A molecule's positions in a topology are stored as its zeroth conformer
        insert_coordinates = insert.conformers[0][:, None, :]
        for molecule in topology.molecules:
            molecule_coordinates = molecule.conformers[0][None, :, :]
            diff_matrix = molecule_coordinates - insert_coordinates
    
            # np.linalg.norm doesn't work on Pint quantities 😢
            working_unit = unit.nanometer
            distance_matrix = (np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit)
    
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
                               
    
    def insert_molecule_and_remove_clashes(self,
                                           topology: Topology,
                                           insert: Molecule,
                                           radius: Quantity = 1.5 * unit.angstrom,
                                           keep: list[Molecule] = []) -> Topology:
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
            if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep): #keep is an empty list by default
                new_top_mols.append(molecule)
                continue
            molecule_coordinates = molecule.conformers[0][None, :, :]
            diff_matrix = molecule_coordinates - insert_coordinates
    
            # np.linalg.norm doesn't work on Pint quantities 😢
            working_unit = unit.nanometer
            distance_matrix = (np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit)
    
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
        
    def generate_topologies(self,
                            structure_file: str,
                            ligand_to_insert: Molecule = None,
                            json_save_fn: str = None):
        """
        Convert the final structure files into OpenFF topologies
        Parameters:
            structure_file: the structure file to generate topology for
            ligand_to_insert: If an openff toolkit Molecule type object is given, it will be inserted into the receptor.
            save_as_json: If a filename is provided, a json file of the generated topology will be saved
        Returns:
            top: The generated topology
        """
        #Create the topology of the complex phase
        top = Topology.from_pdb(structure_file)
        #Insert the ligand into this phase and remove clashes
        if ligand_to_insert is not None:
            top = self.insert_molecule_and_remove_clashes(top, ligand_to_insert)
        if json_save_fn is not None:
            with open(json_save_fn, "w") as f:
                print(top.to_json(), file=f)
        return top

    def top2interchange(self, top: Topology, xmls: list):
        """
        Convert an OpenFF Topology into and OpenFF Interchange (Can take a long time!)
        This is the actual step where MD parameters are applied
        Parameters:
            top: the topology to be converted
        Returns:
            interchange: The interchange object which was created
        """
        sage_ff14sb = ForceField(*xmls)
        return sage_ff14sb.create_interchange(top)

    def interchange2OpenmmSim(self,
                              interchange,
                             temp: openmm_unit):
        """
        Construct an openn simulation object from an openff interachange
        
        """
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

def join_two_topologies(tops: tuple, poss: tuple):
    """
    Joins two topologies by adding the first to the second
    Parameters:
        tops: A two-tuple of openmm Topologies
    Return
    """
    assert len(tops) == 2 and len(poss) == 2
    modeller = openmm.app.Modeller(tops[0], poss[0])
    modeller.add(tops[1], poss[1])
    return modeller.topology, modeller.positions

def describe_system(sys: openmm.openmm.System):
    box_vecs = sys.getDefaultPeriodicBoxVectors()
    print('Box Vectors')
    [print(box_vec) for box_vec in box_vecs]
    forces = sys.getForces()
    print('Forces')
    [print(force) for force in forces]
    num_particles = sys.getNumParticles()
    print(num_particles, 'Particles')

def describe_state(state: openmm.State, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")

def join_two_systems(sys1: openmm.openmm.System, sys2: openmm.openmm.System):
    """
    Joins Two Openmm systems by adding the elements of system 1 to system 2
    Intended use in this notebook is join_two_systems(ligand_sys, receptor_sys)
    
    Parameters:
        sys1 - The openmm system to be added to sys2
        sys2 - The openmm system to have sys1 added to
    Returns
        openmm.openmm.System - The combined system
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
    
    force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
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


start = datetime.now()
self = Simulation_Preparer(job_inputs)

receptor_path, ligand_path = self.seperate_crys_using_MDA(self.crystal_pdb_fn, self.ligand_resname)
print(receptor_path, os.path.isfile(receptor_path))
print(ligand_path, os.path.isfile(ligand_path))

ligand_sdf_path, ligand_protonated = self.obabel_conversion(ligand_path, ['pdb', 'sdf'], out_fn='AUTO', add_Hs=True, rewrite_with_Hs=True)
print(ligand_protonated, os.path.isfile(ligand_protonated))
print(ligand_sdf_path, os.path.isfile(ligand_sdf_path))

self.ligand = Molecule.from_file(ligand_sdf_path)
print(type(self.ligand), self.ligand.to_smiles(explicit_hydrogens=False))

sage_ff = ForceField('openff-2.1.0.offxml')
openmm_ff = openmm.app.ForceField('amber14/protein.ff14SB.xml', 'amber14/lipid17.xml', 'wat_opc3.xml')

cubic_box = unit.Quantity(30 * np.eye(3), unit.angstrom)
interchange = openff_ic.Interchange.from_smirnoff(topology=[self.ligand], force_field=sage_ff, box=cubic_box)

ligand_positions = np.array(interchange.positions) * openmm_unit.nanometer
ligand_sys = interchange.to_openmm_system()
ligand_top = interchange.to_openmm_topology()

protein_protonated, protein_pqr = self.protonate_with_pdb2pqr(receptor_path, at_pH=7)
print(protein_protonated, os.path.isfile(protein_protonated))
print(protein_pqr, os.path.isfile(protein_pqr))

self.protein_solvated = self.run_PDBFixer(protein_protonated, mode=self.job_inputs['build_modes'][0], padding=1.5, ionicStrength=0.15)
print(self.protein_solvated, os.path.isfile(self.protein_solvated))

pdb = openmm.app.PDBFile(self.protein_solvated)
receptor_top, receptor_positions = pdb.getTopology(), pdb.getPositions()
receptor_sys = openmm_ff.createSystem(receptor_top, nonbondedMethod=openmm.app.forcefield.PME)

comp_top, comp_positions = join_two_topologies((ligand_top, receptor_top), (ligand_positions, receptor_positions))

comp_sys = join_two_systems(ligand_sys, receptor_sys)

integrator = openmm.openmm.LangevinIntegrator(300 * openmm_unit.kelvin, 1/openmm_unit.picosecond, 0.001 * openmm_unit.picosecond)

simulation = openmm.app.Simulation(comp_top, receptor_sys, integrator)
simulation.context.setPositions(comp_positions)

#Evaluate and Report pre-minimized energy
describe_state(simulation.context.getState(getEnergy=True, getForces=True), "Original state")
end = datetime.now() - start
print(f'Time to build this simulation: {end}')

#Write Out (and try minimizing on a GPU)
with open('test_final_result.pdb', 'w') as f:
    openmm.app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)

with open('test_final_result.xml', 'w') as f:
    f.write(openmm.XmlSerializer.serialize(simulation.system))
