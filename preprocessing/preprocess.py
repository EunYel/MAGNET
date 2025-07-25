from preprocess_murcko import process_murcko_decomposition
from preprocess_jt import process_junction_tree_decomposition 
from preprocess_brics import process_brics_decomposition

from preprocess_utils import atom_index_data
from preprocess_utils import create_atom_fragment_matrix
from preprocess_utils import convert_smiles_to_mol_objects
from preprocess_utils import set_atom_map_numbers
from preprocess_utils import frag_dict
from preprocess_utils import mk_translation_matrix_with_index
from preprocess_utils import calculate_overlap_weights
from preprocess_utils import combine_overlap_weights
from preprocess_utils import change_to_adjacency_matrix_of_combine_overlap_weight
from preprocess_utils import Create_Final_directed_Graph
from preprocess_utils import smiles_to_vector
from preprocess_utils import create_directed_graphs_with_adjacency


import argparse
from pathlib import Path
import pandas as pd
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

import pickle
import math
import torch
from torch import nn, einsum
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Recap, BRICS, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import networkx as nx
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from itertools import chain
import copy
import umap

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw

import copy
from collections import defaultdict
import re
from collections import defaultdict
from itertools import zip_longest
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report, roc_curve
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import pairwise_distances
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches




def read_smiles_file(file_path):
    ext = Path(file_path).suffix
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".txt":
        df = pd.read_csv(file_path, header=None, names=["smiles"])
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    if "smiles" not in df.columns:
        raise ValueError("Input file must contain a 'smiles' column.")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing scripts for Murcko, BRICS, and JT decomposition.")
    parser.add_argument("input_file", help="Path to the input CSV or TXT file containing SMILES strings.")
    args = parser.parse_args()

    input_file = args.input_file
    if not Path(input_file).exists():
        print(f"Error: Input file does not exist: {input_file}")
        return

    # Read input file to dataframe
    try:
        df = read_smiles_file(input_file)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return


    df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    mol_objects = [Chem.MolFromSmiles(mol, sanitize=True) for mol in convert_smiles_to_mol_objects(df['smiles'])]
    mol_with_index = [set_atom_map_numbers(mol) for mol in copy.deepcopy(mol_objects)]
    all_atom_index_data = atom_index_data(mol_objects, mol_with_index)
    
    # Run each decomposition method
    murcko_error, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko = process_murcko_decomposition(df)
    jt_error, jt_all_frag, jt_indices, jt_mols, bonds_to_break_jt = process_junction_tree_decomposition(df)
    brics_error, brics_all_frag, brics_indices, brics_mols, bonds_to_break_brics = process_brics_decomposition(df)
    
    error_indices = list(set(murcko_error + jt_error + brics_error))
    filtered_data = df.iloc[[i for i in range(len(df)) if i not in error_indices]].reset_index(drop=True)

    jt_translation_matrix_all = []; brics_translation_matrix_all = []; murcko_translation_matrix_all = []

    for data in range(len(filtered_data)):
        jt_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], jt_indices[data], jt_mols[data]))
        brics_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], brics_indices[data], brics_mols[data]))
        murcko_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], murcko_indices[data], murcko_mols[data]))

    all_frag_dict_list = [frag_dict(brics_translation_matrix_all[data], jt_translation_matrix_all[data], murcko_translation_matrix_all[data]) for data in range(len(filtered_data))]
    brics_translation_matrix_index_all, jt_translation_matrix_index_all, murcko_translation_matrix_index_all = mk_translation_matrix_with_index(filtered_data, brics_translation_matrix_all, jt_translation_matrix_all, murcko_translation_matrix_all)
    
    all_overlap_weights = [calculate_overlap_weights(brics_translation_matrix_index_all[data], jt_translation_matrix_index_all[data], murcko_translation_matrix_index_all[data]) for data in range(len(filtered_data))]
    combine_all_overlap_weights = [combine_overlap_weights(all_overlap_weights[data]) for data in range(len(filtered_data))]
    adjacency_matrix_for_combine_all_overlap_weights = [change_to_adjacency_matrix_of_combine_overlap_weight(combine_all_overlap_weights[data]) for data in range(len(filtered_data))]
    final_graph = [Create_Final_directed_Graph(murcko_translation_matrix_index_all[data], jt_translation_matrix_index_all[data], brics_translation_matrix_index_all[data], all_overlap_weights[data]) for data in tqdm(range(len(filtered_data)), desc="Creating Final Graph")]
    
    dataset = list(brics_all_frag|murcko_all_frag|jt_all_frag)
    vectors = smiles_to_vector(dataset)
    smiles_to_vectors = {smiles: vector.tolist() for smiles, vector in zip(dataset, vectors)}
    all_graphs = create_directed_graphs_with_adjacency(filtered_data, all_frag_dict_list, final_graph, smiles_to_vectors, 
                                                   combine_all_overlap_weights, adjacency_matrix_for_combine_all_overlap_weights)
    

    # Output summary
    print("Murcko Fragments:", len(murcko_all_frag))
    print("JT Fragments:", len(jt_all_frag))
    print("BRICS Fragments:", len(brics_all_frag))
    print("All preprocessing tasks completed successfully.")
    print("all graph", len(all_graphs))
    
    # Save final output: all_graphs → Pickle
    output_dir = Path("preprocessed_MetaData")
    output_dir.mkdir(parents=True, exist_ok=True) 

    dataset_name = Path(input_file).stem
    output_filename = output_dir / f"preprocessed_{dataset_name}.pkl"

    combined_data = {
        "filtered_data": filtered_data,
        "all_graphs": all_graphs
    }

    with open(output_filename, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"✅ Saved filtered_data and all_graphs to: {output_filename}")


if __name__ == "__main__":
    main()
