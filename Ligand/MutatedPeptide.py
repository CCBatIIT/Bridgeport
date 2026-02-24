import os, sys, json, pathlib
from ligand_utils import *
from typing import List
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
try:
    from Ligand.Ligand import Ligand
    from Ligand.Analogue import Analogue
except:
    from Ligand import Ligand
    from Analogue import Analogue
from ForceFields.ForceFieldHandler import ForceFieldHandler
from openmm import *
from openmm.app import *
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem
#import parmed as pmd


class MutatedPeptide(Analogue):
    def __init__(self,
                 template: Ligand, 
                 replace_resid: int,
                 replace_smiles: str,
                 replace_resname: str,
                 working_dir: str, 
                 name: str, 
                 resname: str=False, 
                 smiles: str=False,
                 chainid: str=False, 
                 sequence: str=False,
                 verbose: bool=False, 
                 visualize: bool=False):

        # Initialize
        super().__init__(template, working_dir, name, resname, smiles, chainid, sequence, verbose, visualize)
        self.mut_resid = replace_resid
        self.mut_smiles = replace_smiles
        self.mut_resname = replace_resname

        # load amino acid smiles
        self.amino_acid_smiles = {
            "ALA": "N[C@@H](C)C=O",                         # Alanine
            "ARG": "N[C@@H](CCCNC(N)=[NH2+])C=O",           # Arginine (side chain guanidinium)
            "ASN": "N[C@@H](CC(=O)N)C=O",                   # Asparagine
            "ASP": "N[C@@H](CC([O-])=O)C=O",                # Aspartic acid
            "CYS": "N[C@@H](CS)C=O",                        # Cysteine (neutral thiol)
            "GLN": "N[C@@H](CCC(=O)N)C=O",                  # Glutamine
            "GLU": "N[C@@H](CCC([O-])=O)C=O",               # Glutamic acid
            "GLY": "NCC=O",                                 # Glycine
            "HIS": "N[C@@H](CC1=CN=CN1)C=O",                # Histidine (neutral imidazole)
            "ILE": "N[C@@H](C(C)CC)C=O",                    # Isoleucine
            "LEU": "N[C@@H](CC(C)C)C=O",                    # Leucine
            "LYS": "N[C@@H](CCCC[NH3+])C=O",                # Lysine (side chain ammonium)
            "MET": "N[C@@H](CCSC)C=O",                      # Methionine
            "PHE": "N[C@@H](CC1=CC=CC=C1)C=O",              # Phenylalanine
            "PRO": "N1CCC[C@H]1C=O",                        # Proline (ring N)
            "SER": "N[C@@H](CO)C=O",                        # Serine
            "THR": "N[C@@H](C(O)C)C=O",                     # Threonine
            "TRP": "N[C@@H](CC1=CNC2=CC=CC=C12)C=O",        # Tryptophan
            "TYR": "N[C@@H](CC1=CC=C(O)C=C1)C=O",           # Tyrosine
            "VAL": "N[C@@H](C(C)C)C=O",                     # Valine
            "AIB": "CC(C)(N)C=O"                            # AIB
        }



    def run(self, remove_atoms: List=[], change_atoms: dict={}, bonds_to_add: List=[[[],[]]], external_bonds=None, cyclic:bool=False):

        # Set attributes
        self.remove_atoms = remove_atoms
        self.change_atoms = change_atoms
        self.external_bonds = external_bonds
        self.cyclic = cyclic
    
        # Build resid objects 
        self._build_resids()
        self.analogue.get_MCS(strict=True)
        self.analogue.generate_conformers(rmsd_thresh=1)
        self.analogue.prepare_ligand(chain=self.chainid, cyclic=self.cyclic)

        # Build forcefield
        self._build_forcefield()
        # raise Exception()
        self._remove_atoms_from_forcefield()
        # raise Exception()
        
        # Build topology
        self._build_mut_topology()
        self._build_peptide_topology()
        self._patch_topology(bonds_to_add)

        # Update chain
        if self.chainid != False:
            u = mda.Universe(self.pdb)
            u.atoms.chainIDs = self.chainid
            u.atoms.write(self.pdb)
        


    def _build_resids(self):

        # Extract reference
        pdb = md.load_pdb(self.template.pdb)
        sel = pdb.topology.select(f'resSeq {self.mut_resid}')
        ref_resname = pdb.topology.atom(sel[0]).residue.name
        self.ref_residue_pdb = os.path.join(self.working_dir, self.mut_resname + str(self.mut_resid) + '_reference_residue.pdb')
        pdb.atom_slice(sel)[0].save_pdb(self.ref_residue_pdb)
        
        # Build refernce
        self.reference = Ligand(working_dir=self.working_dir, name=self.mut_resname + str(self.mut_resid) + '_reference_residue', smiles=self.amino_acid_smiles[ref_resname])
        self.reference.prepare_ligand(small_molecule_params=True, proximityBonding=True, removeHs=False, visualize=False, cyclic=self.cyclic)

        # Build mutation
        self.analogue_name = self.mut_resname + str(self.mut_resid) + '_residue'
        self.analogue = Analogue(template=self.reference, working_dir=self.working_dir, name=self.analogue_name, smiles=self.mut_smiles, visualize=self.visualize)


    def _build_forcefield(self):

        # Get net charge
        net_charge = int(sum(atom.GetFormalCharge() for atom in self.analogue.mol.GetAtoms()))
        
        # Get charges with antechamber
        self.analogue.mol2 = os.path.join(self.working_dir, self.analogue_name + ".mol2")
        exit_code = os.system(f'antechamber -i {self.analogue.pdb} -fi pdb -o {self.analogue.mol2} -fo mol2 -c bcc -nc {net_charge} -at amber')
        if exit_code != 0:
            raise Exception(f'antechamber -i {self.analogue.pdb} -fi pdb -o {self.analogue.mol2} -fo mol2 -c bcc -nc {net_charge} -at amber')
        # os.system(f'antechamber -i {self.analogue.pdb} -fi pdb -o {self.analogue.mol2} -fo mol2 -nc {net_charge}') # THIS MUST BE CHANGED BACK AT ALL COSTS YOU IDIOT DO NOT COMMIT THIS 

        # Create forcefield params
        self.analogue.frcmod = os.path.join(self.working_dir, self.analogue_name + ".frcmod")
        exit_code = os.system(f'parmchk2 -i {self.analogue.mol2} -f mol2 -o {self.analogue.frcmod}')
        if exit_code != 0:
            raise Exception()

        # Generate .prmtop
        self.analogue.prmtop = os.path.join(self.working_dir, self.analogue_name + ".prmtop")
        self.analogue.inpcrd = os.path.join(self.working_dir, self.analogue_name + ".inpcrd")
        self.analogue.tleap = os.path.join(self.working_dir, self.analogue_name + ".tleap")
        tleap_command = f"""
source leaprc.gaff

LIG = loadmol2 {self.analogue.mol2}
loadamberparams {self.analogue.frcmod}

saveamberparm LIG {self.analogue.prmtop} {self.analogue.inpcrd}
quit
        """
        with open(self.analogue.tleap, 'w') as f:
            f.write(tleap_command)
            f.close()
        exit_code = os.system(f'tleap -f {self.analogue.tleap}')
        if exit_code != 0:
            raise Exception()

        # Convert to .xml
        self.analogue.xml = os.path.join(self.working_dir, self.analogue_name + '.xml')
        
        input_json = f'{pathlib.Path(__file__).parent.resolve()}/../ForceFields/write_xml_pretty_input.json'
        data = json.load(open(input_json, 'r'))
        data['fname_prmtop'] = self.analogue.prmtop
        data['fname_xml'] = self.analogue.xml
        data['ff_prefix'] = str(self.mut_resid)
        json.dump(data, open(input_json, 'w'), indent=6)
        
        exit_code = os.system(f'python {pathlib.Path(__file__).parent.resolve()}/../ForceFields/write_xml_pretty.py* -i {input_json}')
        if exit_code != 0:
            raise Exception()

    def _remove_atoms_from_forcefield(self):

        # Use method from ligand_utils
        remove_xml_atoms(self.analogue.xml, self.mut_resname, self.remove_atoms, self.change_atoms, self.external_bonds)


    def _build_mut_topology(self):

        # Build mutated residue
        self.mut = PDBFile(self.analogue.pdb)
        self.mut = Modeller(self.mut.topology, self.mut.positions)
        self.mut.delete([atom for atom in self.mut.topology.atoms() if atom.name in self.remove_atoms])


    def _build_peptide_topology(self):

        # Replace 
        self.peptide = PDBFile(self.template.pdb)
        self.peptide = Modeller(self.peptide.topology, self.peptide.positions)
        
        # Replace topology
        for chain in self.peptide.topology.chains():
            for res in chain.residues():
                if int(res.id.strip()) == self.mut_resid:
                    self.peptide.delete(res.atoms())
                    self.peptide.add(self.mut.topology, self.mut.positions)   

        self.positions = self.peptide.getPositions()

    
    def _patch_topology(self, bonds_to_add):

        # Define correct order map
        residues = []
        order_map = [] # [chain index, residue index, residue id]
        for i, chain in enumerate(self.peptide.topology.chains()):
            for residue in chain.residues():
                if residue.name == 'UNL':
                    order_map.append([chain.index, residue.index, self.mut_resid])
                else:
                    order_map.append([chain.index, residue.index, residue.id])
                residues.append(residue)

        order_map = np.array(order_map, dtype=int)
        sorted_inds = np.argsort(order_map[:,2])
        residues = [residues[i] for i in sorted_inds]

        # Create new topology
        self.top = Topology()
        new_chain = self.top.addChain([c.id for c in self.peptide.topology.chains()][0])
        atom_map = {}
        for i, residue in enumerate(residues):
            if residue.name == "UNL":
                new_residue = self.top.addResidue(name=self.mut_resname, chain=new_chain, id=str(self.mut_resid))
            else:
                new_residue = self.top.addResidue(residue.name, new_chain, id=residue.id)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Added residue', new_residue.name, new_residue.id, 'to topology', flush=True)
            for atom in residue.atoms():
                new_atom = self.top.addAtom(atom.name, atom.element, new_residue)
                atom_map[atom] = new_atom
        
        # Add existing bonds
        for bond in self.peptide.topology.bonds(): 
            self.top.addBond(atom_map[bond[0]], atom_map[bond[1]])
        
        # Add new bond(s)
        if bonds_to_add is not None:
            for bond_to_add in bonds_to_add:
                atom1_resid, atom1_name = bond_to_add[0]
                atom2_resid, atom2_name = bond_to_add[1]
                atom1, atom2 = None, None
                for atom in self.top.atoms():
                    if atom.name.strip() == atom1_name and int(atom.residue.id) == atom1_resid:
                        atom1 = atom
                    if atom.name.strip() == atom2_name and int(atom.residue.id) == atom2_resid:
                        atom2 = atom
                if atom1 is not None and atom2 is not None:
                    self.top.addBond(atom1, atom2, type='single')
                    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Added bond between', atom1.residue.id, atom1.name, 'and', atom2.residue.id, atom2.name, flush=True)
                else:
                    print(bond_to_add, atom1, atom2)
                    raise Exception()

        # Reorder positions
        reordered_positions = np.zeros((len(self.positions), 3))
        for i, atom in enumerate(self.peptide.getTopology().atoms()):
            old_ind = atom.index
            new_ind = atom_map[atom].index
            reordered_positions[new_ind] = self.positions[old_ind]._value    
        
        self.positions = np.array(reordered_positions)*10

        self.pdb = os.path.join(self.working_dir, self.name + '.pdb')
        with open(self.pdb, 'w') as f:
            PDBFile.writeFile(self.top, self.positions, f, keepIds=True)
            f.close()          
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved first topology to', self.pdb, flush=True)

        

    
    
