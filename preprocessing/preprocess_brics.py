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

from preprocess_utils import remove_atom_mapping
from preprocess_utils import convert_smiles_to_mol_objects
from preprocess_utils import set_atom_map_numbers

# BRICS
def brics_decompose(mol):
    set_atom_map_numbers(mol)
    rw_mol_brics = RWMol(mol)

    bonds_to_break_brics = list(BRICS.FindBRICSBonds(mol))
    for bond in bonds_to_break_brics:
        atom1, atom2 = bond[0]
        rw_mol_brics.RemoveBond(atom1, atom2)

    brics_fragments = Chem.GetMolFrags(rw_mol_brics, asMols=True, sanitizeFrags=False)
    brics_fragment_indices = []
    brics_mols = []
    for frag in brics_fragments:
        indices = [atom.GetAtomMapNum() for atom in frag.GetAtoms()]
        brics_fragment_indices.append(indices)
        brics_mols.append(frag)  # Directly append the fragment Mol object

    return brics_fragment_indices, brics_mols, bonds_to_break_brics

def process_brics_decomposition(data):
    brics_error_indices = []
    brics_all_frag = set()
    brics_indices = []
    brics_mols = []
    bonds_to_break_brics = []

    for i in range(len(data)):
        try:
            # BRICS 분해 실행
            indices, mols, bonds_to_break = brics_decompose(Chem.MolFromSmiles(data['smiles'][i]))
            brics_indices.append(indices)
            bonds_to_break_brics.append(bonds_to_break)

            # SMILES로 변환 후 분자 객체 생성
            mols = [Chem.MolToSmiles(mol) for mol in mols]
            brics_mols.append(convert_smiles_to_mol_objects(mols))

            # 모든 분해된 구조 추가
            convert_brics_mols = set(convert_smiles_to_mol_objects(mols))
            brics_all_frag = brics_all_frag.union(convert_brics_mols)

        except Exception as e:
            # 에러 발생 시 인덱스 저장
            brics_error_indices.append(i)

    return brics_error_indices, brics_all_frag, brics_indices, brics_mols, bonds_to_break_brics

