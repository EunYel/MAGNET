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
from rdkit.Chem.Scaffolds import MurckoScaffold

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


# murcko
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
    murcko_error_indices = []
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
            murcko_error_indices.append(i)

    return murcko_error_indices, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko

# Main Function
def process_smiles(input_file):
    data = pd.read_csv(input_file)
    smiles_list = data['smiles']

    # BRICS 분해 과정 처리 (예시용)
    murcko_error_indices, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko = process_murcko_decomposition(smiles_list)

    # 저장할 데이터 구성
    data_to_save = {
        "murcko_error_indices": murcko_error_indices,
        "murcko_all_frag": murcko_all_frag,
        "murcko_indices": murcko_indices,
        "murcko_mols": murcko_mols,
        "bonds_to_break_murcko": bonds_to_break_murcko,
    }

    # 출력 디렉토리 생성
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 피클 파일로 저장
    with open(output_dir / "murcko_data.pkl", "wb") as file:
        pickle.dump(data_to_save, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SMILES for Murcko.")
    parser.add_argument("input_file", help="Path to the input CSV file containing SMILES strings.")
    args = parser.parse_args()
    process_smiles(args.input_file)