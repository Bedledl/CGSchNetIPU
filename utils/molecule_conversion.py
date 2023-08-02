import tempfile

import ase
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from moleculekit.molecule import Molecule
from torchmd_cg.utils.psfwriter import pdb2psf_CA

AA2INT = {
    'ALA': 1,
    'GLY': 2,
    'PHE': 3,
    'TYR': 4,
    'ASP': 5,
    'GLU': 6,
    'TRP': 7,
    'PRO': 8,
    'ASN': 9,
    'GLN': 10,
    'HIS': 11,
    'HSD': 11,
    'HSE': 11,
    'SER': 12,
    'THR': 13,
    'VAL': 14,
    'MET': 15,
    'CYS': 16,
    'NLE': 17,
    'ARG': 18,
    'LYS': 19,
    'LEU': 20,
    'ILE': 21
}

# pdf file : ATOM      1  N   TYR A   1      25.824  21.671  10.238  1.00  8.64      0    N
# contain: atom number, associated AA, xyz coordinate, ?, element number


def get_moleculekit_obj(pdb_file):
    # first read in the pdb file to get a atom level moleculekit.Molecule
    atom_level_mol = Molecule(pdb_file)

    # now create a temp psf file, that collects the atoms to amino acids
    # use the temporary file to create an atom level amino acids moleculekit.Molecule
    with tempfile.NamedTemporaryFile(suffix=".psf") as psf_tmp_file:
        pdb2psf_CA(pdb_file, psf_tmp_file.name)
        amino_level_mol = Molecule(psf_tmp_file.name)

    index = atom_level_mol.resid - 1
    n_atoms = atom_level_mol.numAtoms

    amino_level_mol.coords = np.zeros((amino_level_mol.numAtoms, 3, 1))
    amino_level_mol.masses = np.zeros(amino_level_mol.numAtoms)

    for i in range(n_atoms):
        if atom_level_mol.record[i] != "ATOM":
            continue

        amino_level_mol.coords[index[i]] += atom_level_mol.coords[i]
        atomic_number = atomic_numbers[atom_level_mol.element[i]]
        amino_level_mol.masses[index[i]] += atomic_masses[atomic_number]

    amino_indices, amino_atom_num = np.unique(index, return_counts=True)

    for amino_index, num in zip(amino_indices, amino_atom_num):
        amino_level_mol.coords[amino_index] = amino_level_mol.coords[amino_index]/num

    return amino_level_mol


def moleculekit2ase(mk_mol: Molecule):
    positions = mk_mol.coords.squeeze()
    masses = mk_mol.masses
    n_atoms = mk_mol.numAtoms

    atom_types = np.zeros(n_atoms)
    for i, aa_name in enumerate(mk_mol.resname):
        atom_types[i] = AA2INT[aa_name]

    ase_mol = ase.Atoms(
        positions=positions,
        masses=masses,
        numbers=atom_types,
    )
    return ase_mol

