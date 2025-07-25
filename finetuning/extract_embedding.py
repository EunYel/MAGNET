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

import argparse
from pathlib import Path

import sys
sys.path.append("/data1/project/yeeun/MAGNET_final/pretraining/")
from graph_transformer import GraphTransformerModel

sys.path.append("/data1/project/yeeun/MAGNET_final/preprocessing/")
from preprocess_utils import smiles_to_vector


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    """
    DataLoader에서 사용되는 collate 함수.
    edge와 adjacency가 동일한 크기인 [num_nodes, num_nodes] 형식인 경우에 맞게 데이터를 패딩합니다.
    """
    # 첫 번째 뷰만 사용 (view_1만 유지)
    view_1_nodes = [item.x for item in batch]
    view_1_edges = [item.edge for item in batch]
    view_1_adjacency = [item.adjacency for item in batch]

    def pad_graphs(nodes_batch, edges_batch, adjacency_batch):
        # 노드 및 edge/adjacency의 최대 크기를 따로 확인
        max_node_size = max(node.size(0) for node in nodes_batch)
        max_edge_size = max(edge.size(0) for edge in edges_batch)
        max_adj_size = max(adj.size(0) for adj in adjacency_batch)

        # 모든 padding은 이들 중 가장 큰 걸로 맞춘다
        max_size = max(max_node_size, max_edge_size, max_adj_size)
        feature_dim = nodes_batch[0].size(1)

        # 노드 패딩
        padded_nodes = torch.zeros(len(nodes_batch), max_size, feature_dim)
        for i, node in enumerate(nodes_batch):
            if node.size(0) == 0: print(f"Warning: Graph {i} has no nodes!")
            padded_nodes[i, :node.size(0), :] = node

        # 엣지 정보 패딩
        padded_edges = torch.zeros(len(edges_batch), max_size, max_size, 1)  # Ensure 4D
        for i, edge in enumerate(edges_batch):
            if edge.dim() == 2:  # If edges are 2D, add a singleton dimension
                edge = edge.unsqueeze(-1)  # (num_nodes, num_nodes) -> (num_nodes, num_nodes, 1)
            edge_size = edge.size(0)
            padded_edges[i, :edge_size, :edge_size, :] = edge

        # 인접 행렬 패딩
        padded_adjacency = torch.zeros(len(adjacency_batch), max_size, max_size)
        for i, adjacency in enumerate(adjacency_batch):
            adj_size = adjacency.size(0)
            padded_adjacency[i, :adj_size, :adj_size] = adjacency

        return padded_nodes, padded_edges, padded_adjacency

    # 첫 번째 뷰 패딩만 반환
    view_1_padded = pad_graphs(view_1_nodes, view_1_edges, view_1_adjacency)

    return view_1_padded  # ✅ 뷰 하나만 반환

import torch

def preprocess_batch(batch):
    """
    배치 데이터를 모델 입력 형식에 맞게 변환합니다.
    `collate_fn()`에서 반환한 데이터를 그대로 사용합니다.
    """
    # `batch`에서 노드, 엣지, 인접 행렬 정보 가져오기
    node_features, edge_matrices, adjacency_matrices = batch

    # GPU로 이동 및 데이터 타입 변환
    node_features = node_features.float().to(device)  # [batch_size, max_nodes, feature_dim]
    edge_matrices = edge_matrices.to(device)  # [batch_size, max_nodes, max_nodes, 1]
    adjacency_matrices = adjacency_matrices.to(device)  # [batch_size, max_nodes, max_nodes]

    return node_features, edge_matrices, adjacency_matrices

def extract_graph_embedding(model, loader, pooling="mean"):
    """
    모델과 데이터 로더를 사용해 그래프 임베딩을 생성합니다.
    
    Args:
        model (nn.Module): 학습된 Graph Transformer 모델
        loader (DataLoader): 데이터 로더
        pooling (str): "mean", "max", "sum" 중 선택
        
    Returns:
        graph_embeddings (torch.Tensor): 그래프 임베딩 (batch_size, embedding_dim)
    """
    model.eval()
    graph_embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            # 배치 데이터 가져오기
            node_features, edge_matrices, adjacency_matrices = preprocess_batch(batch)
            
            # 노드 임베딩 생성
            node_embeddings = model(node_features, edge_matrices, adjacency_matrices)
            
            # 그래프 임베딩 생성
            if pooling == "mean":
                # graph_embedding = node_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
                mask = adjacency_matrices.sum(dim=-1) > 0  # [batch_size, max_nodes]
                mask = mask.float().unsqueeze(-1)  # [batch_size, max_nodes, 1]
                
                # 🟢 NaN을 0으로 대체
                node_embeddings = torch.nan_to_num(node_embeddings, nan=0.0)
                
                masked_embeddings = node_embeddings * mask
                denom = mask.sum(dim=1).clamp(min=1)  # 실제 유효 노드 개수
                graph_embedding = masked_embeddings.sum(dim=1) / denom
            elif pooling == "max":
                graph_embeddinag, _ = node_embeddings.max(dim=1)  # [batch_size, embedding_dim]
            elif pooling == "sum":
                graph_embedding = node_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            graph_embeddings.append(graph_embedding)
    
    # 배치별 그래프 임베딩을 하나의 텐서로 병합
    graph_embeddings = torch.cat(graph_embeddings, dim=0)
    return graph_embeddings

def generate_combined_embeddings(df, graph_embeddings, device="cuda"):
    """
    각 데이터셋에 대해 ChemBERTa 임베딩과 그래프 임베딩을 concat하는 함수.

    Parameters:
    - df: SMILES 열을 포함하는 pandas DataFrame
    - graph_embeddings: 기존 Graph 기반 임베딩 (torch.Tensor)
    - device: 사용할 디바이스 ("cuda" or "cpu")

    Returns:
    - combined_embeddings: torch.Tensor, shape = (N, graph_dim + chemberta_dim)
    """
    # SMILES 리스트
    smiles_list = list(df["smiles"])
    
    # ChemBERTa 임베딩
    chemberta_embeddings = smiles_to_vector(smiles_list)  # (N, 768)
    chemberta_embeddings = chemberta_embeddings.to(device)
    
    # 기존 그래프 임베딩도 디바이스 일치
    graph_embeddings = graph_embeddings.to(device)

    # 임베딩 결합
    combined_embeddings = torch.cat([graph_embeddings, chemberta_embeddings], dim=1)

    return combined_embeddings


def main():
    
    parser = argparse.ArgumentParser(description="Run finetuning")
    parser.add_argument("input_file", help="Path to the input pickle file (contains 'all_graphs' and 'filtered_data').")
    parser.add_argument("--pt_file", required=True, help="Path to the trained model weights (.pt file).")
    args = parser.parse_args()

    input_file = args.input_file
    pt_file = args.pt_file
    
    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return

    # 🔽 피클 파일 로딩
    with open(input_file, "rb") as f:
        data = pickle.load(f)
        all_graphs = data["all_graphs"]
        filtered_data = data["filtered_data"]
        

    # 저장된 모델 불러오기
    GraphTransformer_model = GraphTransformerModel(node_dim=768, edge_dim=1, num_blocks=4, num_heads=8, last_average=True, model_dim=256).to(device)

    # 학습된 가중치 로드
    GraphTransformer_model.load_state_dict(torch.load(pt_file, map_location=device))

    GraphTransformer_model.eval()
    print("Model loaded successfully!")

    from torch_geometric.loader import DataLoader  

    # 데이터 로더를 torch_geometric.loader.DataLoader로 변경 
    data_loader = torch.utils.data.DataLoader(all_graphs, batch_size=32, shuffle=True, collate_fn=collate_fn) 
    embeddings = extract_graph_embedding(GraphTransformer_model, data_loader, pooling="mean")
    print(f"dataset Graph embeddings shape: {embeddings.shape}")  # [num_graphs, embedding_dim]
    
    final_embeddings = generate_combined_embeddings(filtered_data, embeddings)
    
    
    output_dir = Path("finetuning_embedding")
    output_dir.mkdir(parents=True, exist_ok=True) 

    dataset_name = Path(input_file).stem
    output_filename = output_dir / f"embedding_{dataset_name}.pkl"

    combined_data = {
        "final_embeddings": final_embeddings,
    }

    print(len(final_embeddings))
    
    with open(output_filename, "wb") as f:
        pickle.dump(combined_data, f)
    
    

if __name__ == "__main__":
    main()

    
