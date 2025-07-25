import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap, BRICS
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import re
from pathlib import Path
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold

from preprocess_utils import remove_atom_mapping
from preprocess_utils import convert_smiles_to_mol_objects
from preprocess_utils import set_atom_map_numbers

def murcko_decompose(mol):
    murcko_indices = []
    murcko_mols = []
    bonds_to_break_murcko = []

    try:
        set_atom_map_numbers(mol)
    except Exception as e:
        print(f"Error in set_atom_map_numbers: {e}")

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            print("Scaffold is None, returning full molecule as a fragment.")
            indices = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
            murcko_indices.append(indices)
            murcko_mols.append(mol)
            return murcko_indices, murcko_mols, bonds_to_break_murcko
    except Exception as e:
        print(f"Error in MurckoScaffold.GetScaffoldForMol: {e}")

    try:
        scaffold_indices = list(atom.GetAtomMapNum() for atom in scaffold.GetAtoms())
    except Exception as e:
        print(f"Error extracting scaffold indices: {e}")

    try:
        rw_mol = Chem.RWMol(mol)
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            if (atom1.GetAtomMapNum() in scaffold_indices and atom2.GetAtomMapNum() not in scaffold_indices) or \
               (atom2.GetAtomMapNum() in scaffold_indices and atom1.GetAtomMapNum() not in scaffold_indices):
                bonds_to_break_murcko.append([atom1.GetAtomMapNum(), atom2.GetAtomMapNum()])
    except Exception as e:
        print(f"Error in bond analysis: {e}")

    try:
        for atom1_idx, atom2_idx in bonds_to_break_murcko:
            rw_mol.RemoveBond(atom1_idx, atom2_idx)
    except Exception as e:
        print(f"Error removing bonds: {e}")

    try:
        fragment_mols = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=False)
        for frag in fragment_mols:
            murcko_indices.append([atom.GetAtomMapNum() for atom in frag.GetAtoms()])
            murcko_mols.append(frag)
    except Exception as e:
        print(f"Error extracting fragments: {e}")

    return murcko_indices, murcko_mols, bonds_to_break_murcko

def process_murcko_decomposition(data):
    murcko_error = []
    murcko_indices = []
    murcko_mols = []
    bonds_to_break_murcko = []
    murcko_all_frag = set()

    for i in range(len(data)):
        try:
            # Murcko 분해 실행
            indices, mols, bonds_to_break = murcko_decompose(Chem.MolFromSmiles(data['smiles'][i]))
            murcko_indices.append(indices)
            bonds_to_break_murcko.append(bonds_to_break)

            # SMILES로 변환 후 분자 객체 생성
            smiles_mols = [Chem.MolToSmiles(mol, kekuleSmiles=False) for mol in mols]
            murcko_mols.append(convert_smiles_to_mol_objects(smiles_mols))

            # 모든 분해된 구조 추가
            convert_murcko_mols = set(convert_smiles_to_mol_objects(smiles_mols))
            murcko_all_frag = murcko_all_frag.union(convert_murcko_mols)

        except Exception as e:
            # 에러 발생 시 인덱스 저장
            murcko_error.append(i)

    return murcko_error, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko
