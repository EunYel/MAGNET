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
import pickle
from pathlib import Path

# Helper Functions
# Remove atom mapping from SMILES strings in the format [element:number]
def remove_atom_mapping(smiles):
    return re.sub(r'\:\d+\]', ']', smiles)

# Convert SMILES strings to Mol objects after removing atom mappings
def convert_smiles_to_mol_objects(smiles_list):
    mol_objects = []
    for smiles in smiles_list:
        # Remove mapping numbers
        cleaned_smiles = remove_atom_mapping(smiles)
        try:
            # Create Mol object without sanitization
            mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=False)
            if mol is not None:
                smiles_with_aromaticity = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
                mol_objects.append(smiles_with_aromaticity)
        except Exception as e:
            print(f"Failed to create Mol object from: {cleaned_smiles}, error: {e}")
    return mol_objects

def set_atom_map_numbers(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def atom_index_data(mol_objects, mol_with_index):
    atom_index_data = []
    for i in range(len(mol_objects)):
        atom_data = []
        for atom in mol_with_index[i].GetAtoms():
            atom_index = atom.GetAtomMapNum()
            atom_symbol = atom.GetSymbol()
            atom_data.append({"Atom Index": atom_index, "Atom Symbol": atom_symbol})
        atom_index_data.append(atom_data)

    return atom_index_data


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

# Main Function
def process_smiles(input_file):
    data = pd.read_csv(input_file)
    smiles_list = data['smiles']

    # BRICS 분해 과정 처리 (예시용)
    brics_error_indices, brics_all_frag, brics_indices, brics_mols, bonds_to_break_brics = process_brics_decomposition(smiles_list)

    # 저장할 데이터 구성
    data_to_save = {
        "brics_error_indices": brics_error_indices,
        "brics_all_frag": brics_all_frag,
        "brics_indices": brics_indices,
        "brics_mols": brics_mols,
        "bonds_to_break_brics": bonds_to_break_brics,
    }

    # 출력 디렉토리 생성
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 피클 파일로 저장
    with open(output_dir / "brics_data.pkl", "wb") as file:
        pickle.dump(data_to_save, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SMILES for BRICS.")
    parser.add_argument("input_file", help="Path to the input CSV file containing SMILES strings.")
    args = parser.parse_args()
    process_smiles(args.input_file)