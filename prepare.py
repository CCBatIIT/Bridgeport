from ybp_utils import *

class Simulation_Preparer(object):

    def __init__ (self, crystal_pdb_fn, ligand_resname):
        self.ligand_resname = ligand_resname
        # Split Crystal into protein and ligand pdbs based on selection strings
        receptor_path, ligand_path = seperate_crys_using_MDA(crystal_pdb_fn, self.ligand_resname)
        # Convert ligand pdb above to an sdf with openbabel, addhs and make sdf
        ligand_sdf_path = obabel_conversion((ligand_path,'ligand.sdf'))
        # The previous function also rewrites the input file with protons added at the following path
        ligand_protonated = os.path.splitext(ligand_path)[0] + '_H' + os.path.splitext(ligand_path)[-1]
        #Verify existance of the protonated pdb and sdf
        print(os.path.isfile(ligand_protonated), ligand_protonated, os.path.isfile(ligand_sdf_path), ligand_sdf_path)
        # Load an  openff Molecule from a SDF file
        self.ligand = Molecule.from_file(ligand_sdf_path)
        # Print out a SMILES code for the ligand
        print(self.ligand.to_smiles(explicit_hydrogens=False))
        # Protonate the receptor (ph7), then put it in a box of water
        receptor_path = protonate_with_pdb2pqr(receptor_path)
        self.receptor_solvated_fn = PDBFixer_solvation(receptor_path)
        # Create a phase of the ligand in a box of water
        self.ligand_solvated_path = PDBFixer_solvation(ligand_protonated)

    def generate_topologies(self, save_as_jsons=True):
        #Create the topology of the complex phase
        comp_top = Topology.from_pdb(self.receptor_solvated_fn)
        #Insert the ligand into this phase and remove clashes
        self.comp_top = insert_molecule_and_remove_clashes(comp_top, self.ligand)
        # Do the same as above for the solvent phase, requires a slight workaround where the ligand is removed and added back in by openff
        solvent_phase_path = just_the_solvent(self.ligand_solvated_path, self.ligand_resname)
        solv_top = Topology.from_pdb(solvent_phase_path) #ligand cannot be here
        self.solv_top = insert_molecule_and_remove_clashes(solv_top, self.ligand) #and so is added here
        if save_as_jsons:
            with open("complex_topology.json", "w") as f:
                print(comp_top.to_json(), file=f)
            with open("solvent_topology.json", "w") as f:
                print(solv_top.to_json(), file=f)

    def generate_interchanges(self, xmls):
        sage_ff14sb = ForceField(*xmls)
        # Create interchanges of both phases (this takes a while)
        self.comp_interchange = sage_ff14sb.create_interchange(self.comp_top)
        self.solv_interchange = sage_ff14sb.create_interchange(self.solv_top)

    def openmm_writeout(self):
        make_simulation_and_writeout(self.comp_interchange, 'complex', self.ligand_resname)
        make_simulation_and_writeout(self.solv_interchange, 'solvent', self.ligand_resname)
