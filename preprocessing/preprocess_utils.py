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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def remove_atom_mapping(smiles):
    return re.sub(r'\:\d+\]', ']', smiles)

def convert_smiles_to_mol_objects(smiles_list):
    mol_objects = []
    for smiles in smiles_list:
        cleaned_smiles = remove_atom_mapping(smiles)
        try:
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

def create_atom_fragment_matrix(atom_data, atom_indices, fragments):
    total_atoms = len(atom_data)
    translation_matrix = np.zeros((total_atoms, len(fragments)), dtype=int)

    for frag_idx, indices in enumerate(atom_indices):
        for atom_idx in indices:
            translation_matrix[atom_idx, frag_idx] = 1

    atom_labels = [f"{atom['Atom Symbol']} {atom['Atom Index']}" for atom in atom_data]
    df_translation_matrix = pd.DataFrame(translation_matrix, index=atom_labels, columns=fragments)

    return df_translation_matrix

def frag_dict(brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    # Initialize matrices list

    matrices = [
        brics_translation_matrix.copy(),  # Use copy to avoid modifying the original DataFrame
        jt_translation_matrix.copy(),
        murcko_translation_matrix.copy()
    ]

    column_counter = 1  # Starting column number

    # Final dictionary to store combined results
    frag_dict = defaultdict(list)

    for i, matrix in enumerate(matrices):
        # Get original column names
        columns = matrix.columns.tolist()
        # print(columns)
        
        # Generate new column indices
        num_columns = matrix.shape[1]
        num_index = range(column_counter, column_counter + num_columns)
        matrix.columns = num_index  # Assign new columns
        num_index = list(num_index)
        column_counter += num_columns

        # Create a dictionary to handle duplicate keys
        result_dict = defaultdict(list)
        for key, value in zip_longest(columns, num_index):
            result_dict[key].append(value)

        # Merge result_dict into frag_dict, combining values for duplicate keys
        for key, value in result_dict.items():
            frag_dict[key].extend(value)

    # Convert frag_dict back to a regular dictionary for output
    return dict(frag_dict)

def mk_translation_matrix_with_index(graph_data, brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    brics_translation_matrix_index = copy.deepcopy(brics_translation_matrix)
    jt_translation_matrix_index = copy.deepcopy(jt_translation_matrix)
    murcko_translation_matrix_index = copy.deepcopy(murcko_translation_matrix)

    for data in range(len(graph_data)):
        # 각 데이터 인덱스에 대해 다시 복사
        matrices = [
            brics_translation_matrix_index[data],
            jt_translation_matrix_index[data],
            murcko_translation_matrix_index[data]
        ]

        column_counter = 1  # Starting column number

        # Final dictionary to store combined results
        frag_dict = defaultdict(list)

        for i, matrix in enumerate(matrices):
            # Get original column names
            columns = matrix.columns.tolist()
            
            # Generate new column indices
            num_columns = matrix.shape[1]
            num_index = range(column_counter, column_counter + num_columns)
            matrix.columns = num_index  # Assign new columns
            num_index = list(num_index)
            column_counter += num_columns
    return brics_translation_matrix_index, jt_translation_matrix_index, murcko_translation_matrix_index

def calculate_overlap_weights(brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    def compute_adjacency_matrix(matrix_a, matrix_b):
        indices_a = matrix_a.columns.tolist()
        indices_b = matrix_b.columns.tolist()
        indices = indices_a + indices_b

        # Compute the overlap matrix: A * (B^T)
        overlap_matrix = np.dot(matrix_a.T, matrix_b)
        overlap_df = pd.DataFrame(overlap_matrix, index=indices_a, columns=indices_b)

        num_a = len(indices_a)
        num_b = len(indices_b)
        total_nodes = num_a + num_b

        adjacency_matrix = np.zeros((total_nodes, total_nodes))
        adjacency_df = pd.DataFrame(adjacency_matrix, index=indices, columns=indices)

        # Use numpy to populate the adjacency matrix without loops
        mask_a_b = np.ix_(range(num_a), range(num_a, total_nodes))
        mask_b_a = np.ix_(range(num_a, total_nodes), range(num_a))
        
        adjacency_df.values[mask_a_b] = np.round(overlap_df.values / num_a, 3)
        adjacency_df.values[mask_b_a] = np.round(overlap_df.values.T / num_b, 3)

        return adjacency_df

    brics_jt = compute_adjacency_matrix(brics_translation_matrix, jt_translation_matrix)
    jt_murcko = compute_adjacency_matrix(jt_translation_matrix, murcko_translation_matrix)
    brics_murcko = compute_adjacency_matrix(brics_translation_matrix, murcko_translation_matrix)

    return [brics_jt, jt_murcko, brics_murcko]

def combine_overlap_weights(all_overlap_weights_data):
    brics_jt, jt_murcko, brics_murcko = all_overlap_weights_data

    # Extract indices and combine them
    combined_indices = sorted(set(brics_jt.index).union(set(jt_murcko.index), set(brics_murcko.index)))
    
    # Initialize an empty DataFrame for the combined adjacency matrix
    total_nodes = len(combined_indices)
    combined_df = pd.DataFrame(np.zeros((total_nodes, total_nodes)), 
                               index=combined_indices, columns=combined_indices)

    # Update combined_df with values from brics_jt
    row_indices = [combined_indices.index(i) for i in brics_jt.index]
    col_indices = [combined_indices.index(i) for i in brics_jt.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = brics_jt.values

    # Update combined_df with values from jt_murcko
    row_indices = [combined_indices.index(i) for i in jt_murcko.index]
    col_indices = [combined_indices.index(i) for i in jt_murcko.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = jt_murcko.values

    # Update combined_df with values from brics_murcko
    row_indices = [combined_indices.index(i) for i in brics_murcko.index]
    col_indices = [combined_indices.index(i) for i in brics_murcko.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = brics_murcko.values

    return combined_df

def change_to_adjacency_matrix_of_combine_overlap_weight(combine_overlap_weight):
    # Create a copy of the input DataFrame
    adjacency_matrix = combine_overlap_weight.copy()
    
    # Use numpy to create a boolean matrix where values are 1 if they are non-zero
    adjacency_matrix.values[adjacency_matrix.values != 0] = 1
    
    return adjacency_matrix

def Create_Final_directed_Graph(murcko_translation_matrix, jt_translation_matrix, brics_translation_matrix, weight):
    # Mock-up DataFrame based on the image
    df1 = np.matmul(brics_translation_matrix.T, jt_translation_matrix)
    df2 = np.matmul(jt_translation_matrix.T, murcko_translation_matrix)
    df3 = np.matmul(brics_translation_matrix.T, murcko_translation_matrix)

    # Create a graph using NetworkX
    G1 = nx.DiGraph()
    # Add edges for non-zero elements in the DataFrame
    for row_index, row_name in enumerate(df1.index):
        for col_index, col_name in enumerate(df1.columns):
            value = df1.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G1.add_edge(row_name, col_name, weight=weight[0].loc[row_name, col_name])
                G1.add_edge(col_name, row_name, weight=weight[0].loc[col_name, row_name])

    G2 = nx.DiGraph()
    # Add edges for non-zero elements in the DataFrame
    for row_index, row_name in enumerate(df2.index):
        for col_index, col_name in enumerate(df2.columns):
            value = df2.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G2.add_edge(row_name, col_name, weight=weight[1].loc[row_name, col_name])
                G2.add_edge(col_name, row_name, weight=weight[1].loc[col_name, row_name])

    G3 = nx.DiGraph()
    # Add edges for non-zero elements in the DataFrame
    for row_index, row_name in enumerate(df3.index):
        for col_index, col_name in enumerate(df3.columns):
            value = df3.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G3.add_edge(row_name, col_name, weight=weight[2].loc[row_name, col_name])
                G3.add_edge(col_name, row_name, weight=weight[2].loc[col_name, row_name])

    combined_graph = nx.compose_all([G1, G2, G3])
    return combined_graph

def Create_Final_undirected_Graph(murcko_translation_matrix, jt_translation_matrix, brics_translation_matrix):
    df1 = np.matmul(murcko_translation_matrix.T, jt_translation_matrix)
    df2 = np.matmul(murcko_translation_matrix.T, brics_translation_matrix)
    df3 = np.matmul(brics_translation_matrix.T, jt_translation_matrix)

    G1 = nx.Graph()
    for row_index, row_name in enumerate(df1.index):
        for col_index, col_name in enumerate(df1.columns):
            value = df1.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G1.add_edge(row_name, col_name)


    G2 = nx.Graph()
    # Add edges for non-zero elements in the DataFrame
    for row_index, row_name in enumerate(df2.index):
        for col_index, col_name in enumerate(df2.columns):
            value = df2.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G2.add_edge(row_name, col_name)

    G3 = nx.Graph()
    # Add edges for non-zero elements in the DataFrame
    for row_index, row_name in enumerate(df3.index):
        for col_index, col_name in enumerate(df3.columns):
            value = df3.iloc[row_index, col_index]
            if value != 0:  # Check for non-zero entries
                G3.add_edge(row_name, col_name)

    combined_graph = nx.compose_all([G1, G2, G3])
    return combined_graph

from functools import lru_cache
@lru_cache(maxsize=1)
def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

# SMILES 데이터셋을 벡터로 변환하는 함수
def smiles_to_vector(smiles_list, batch_size=64, output_dim=None):
    """
    SMILES 리스트를 입력받아 벡터 임베딩을 반환.
    - smiles_list: SMILES 문자열 리스트
    - batch_size: 배치 크기
    - output_dim: 출력 임베딩 차원 (None이면 기본 hidden_size 사용)
    """
    tokenizer, model = get_model_and_tokenizer("seyonec/ChemBERTa_zinc250k_v2_40k")
    model = model.to(device)
    vectors = []
    num_batches = len(smiles_list) // batch_size + (1 if len(smiles_list) % batch_size != 0 else 0)

    # 추가 선형 계층 (선택적으로 차원을 줄임)
    projection_layer = None
    if output_dim is not None:
        hidden_size = model.config.hidden_size
        projection_layer = torch.nn.Linear(hidden_size, output_dim).to(device)

    # tqdm으로 진행률 표시
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Creating smiles_to_vector"):
        # 배치 데이터 준비
        batch = smiles_list[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # 모델 출력 얻기 (hidden_states 사용)
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 마지막 히든 레이어 (batch_size, sequence_length, hidden_size)

            # [CLS] 토큰의 벡터 추출
            cls_embeddings = hidden_states[:, 0, :]  # (batch_size, hidden_size)
            if projection_layer is not None:
                cls_embeddings = projection_layer(cls_embeddings)  # (batch_size, output_dim)

            vectors.append(cls_embeddings.cpu())  # CPU로 이동하여 저장

    return torch.cat(vectors, dim=0)  # 최종 벡터 결합

# 특정 값을 찾는 함수
def find_key_by_value(dictionary, target_value):
    return [key for key, values in dictionary.items() if target_value in values]

def create_directed_graphs_with_adjacency(graph_data, frag_dict_list, final_graph, smiles_to_vectors, 
                                 combine_all_overlap_weights, adjacency_matrix_for_combine_all_overlap_weights):
    graphs = []

    for data in tqdm(range(len(graph_data)), desc="Creating Graphs with Adjacency"):
        # 노드 특징
        key_list = list(chain.from_iterable(find_key_by_value(frag_dict_list[data], i) for i in list(np.sort(final_graph[data].nodes))))
        # key_list = frag_dict_list[data]
        node_features = torch.tensor([smiles_to_vectors[key] for key in key_list])
        
        # combine_all_overlap_weights는 edge에 그대로 사용
        edge_attr = torch.tensor(combine_all_overlap_weights[data].values, dtype=torch.float)

        # adjacency_matrix_for_combine_all_overlap_weights는 adjacency에 사용
        adjacency = torch.tensor(adjacency_matrix_for_combine_all_overlap_weights[data].values, dtype=torch.float)

        # Create the graph data object
        graph = Data(
            x=node_features,
            edge=edge_attr, 
            adjacency=adjacency,
            smiles=graph_data['smiles'][data]
        )
        graphs.append(graph)

    return graphs


