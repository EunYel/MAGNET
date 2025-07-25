from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

from contrastive_loss import MaskedGraphDataset
from contrastive_loss import ContrastiveLoss

from graph_transformer import GraphTransformerModel
import matplotlib.pyplot as plt

def collate_fn(batch):
    """
    DataLoader에서 사용되는 collate 함수.
    edge와 adjacency가 동일한 크기인 [num_nodes, num_nodes] 형식인 경우에 맞게 데이터를 패딩합니다.
    """
    # 각 데이터 객체에서 두 개의 마스크 뷰 추출
    view_1_nodes = [item[0].x for item in batch]
    view_1_edges = [item[0].edge for item in batch]
    view_1_adjacency = [item[0].adjacency for item in batch]

    view_2_nodes = [item[1].x for item in batch]
    view_2_edges = [item[1].edge for item in batch]
    view_2_adjacency = [item[1].adjacency for item in batch]

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

    # 첫 번째 뷰 패딩
    view_1_padded = pad_graphs(view_1_nodes, view_1_edges, view_1_adjacency)
    # 두 번째 뷰 패딩
    view_2_padded = pad_graphs(view_2_nodes, view_2_edges, view_2_adjacency)

    return view_1_padded, view_2_padded



import argparse
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run pretraining")
    parser.add_argument("input_file", help="Path to the input pickle file containing preprocessed graph objects.")
    args = parser.parse_args()

    input_file = args.input_file
    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        return

    # 🔽 피클 파일 로딩
    with open(input_file, "rb") as f:
        data = pickle.load(f)
        all_graphs = data["all_graphs"]

    print(f"✅ Loaded {len(all_graphs)} graphs from {input_file}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    model = GraphTransformerModel(
        node_dim=768,
        edge_dim=1,
        num_blocks=4,  # number of graph transformer blocks
        num_heads=8,
        last_average=True,  # whether to average or concatenation at the last block
        model_dim=256  # if None, node_dim will be used as the dimension of the graph transformer block
    ).to(device)

    contrastive_loss = ContrastiveLoss(temperature=0.15)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Example usage
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(all_graphs, 
        [int(0.8 * len(all_graphs)), int(0.1 * len(all_graphs)), len(all_graphs) - int(0.8 * len(all_graphs)) - int(0.1 * len(all_graphs))])

    # DataLoader 생성
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(MaskedGraphDataset(train_dataset, mask_ratio=0.15, num_masks=2), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(MaskedGraphDataset(valid_dataset, mask_ratio=0, num_masks=2), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(MaskedGraphDataset(test_dataset, mask_ratio=0, num_masks=2), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    from tqdm import tqdm

    train_losses = []
    valid_losses = []

    for epoch in range(10): 
        model.train()
        total_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=True)
        for batch in train_loader_tqdm:
            # collate_fn에서 반환된 데이터는 두 개의 뷰로 구성된 튜플
            (view_1_nodes, view_1_edges, view_1_adjacency), (view_2_nodes, view_2_edges, view_2_adjacency) = batch

            # GPU로 데이터 전송
            view_1_nodes, view_1_edges, view_1_adjacency = (
                view_1_nodes.to(device),
                view_1_edges.to(device),
                view_1_adjacency.to(device),
            )
            view_2_nodes, view_2_edges, view_2_adjacency = (
                view_2_nodes.to(device),
                view_2_edges.to(device),
                view_2_adjacency.to(device),
            )

            # 모델 예측
            z_1 = model(view_1_nodes, view_1_edges, view_1_adjacency)
            z_2 = model(view_2_nodes, view_2_edges, view_2_adjacency)

            # Contrastive Loss 계산
            loss = contrastive_loss(z_1, z_2)
            
            # Backpropagation 및 최적화
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping 추가
            optimizer.step()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        # 평균 학습 손실 계산
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation 단계
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                (view_1_nodes, view_1_edges, view_1_adjacency), (view_2_nodes, view_2_edges, view_2_adjacency) = batch

                # GPU로 데이터 전송
                view_1_nodes, view_1_edges, view_1_adjacency = (
                    view_1_nodes.to(device),
                    view_1_edges.to(device),
                    view_1_adjacency.to(device),
                )
                view_2_nodes, view_2_edges, view_2_adjacency = (
                    view_2_nodes.to(device),
                    view_2_edges.to(device),
                    view_2_adjacency.to(device),
                )

                # 모델 예측 및 Contrastive Loss 계산
                z_1 = model(view_1_nodes, view_1_edges, view_1_adjacency)
                z_2 = model(view_2_nodes, view_2_edges, view_2_adjacency)
                loss = contrastive_loss(z_1, z_2)

                valid_loss += loss.item()

            avg_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            print(f"Epoch {epoch+1}, Valid Loss: {avg_valid_loss:.4f}")

    # torch.save(model.state_dict(), f"graph_transformer_model_temp{0.2}_mask{0.3}_epoch{50}.pt")
    # print("Model saved successfully!")

    # Testing 단계
    model.eval()
    test_loss = 0
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for batch in test_loader_tqdm:
            (view_1_nodes, view_1_edges, view_1_adjacency), (view_2_nodes, view_2_edges, view_2_adjacency) = batch

            # GPU로 데이터 전송
            view_1_nodes, view_1_edges, view_1_adjacency = (
                view_1_nodes.to(device),
                view_1_edges.to(device),
                view_1_adjacency.to(device),
            )
            view_2_nodes, view_2_edges, view_2_adjacency = (
                view_2_nodes.to(device),
                view_2_edges.to(device),
                view_2_adjacency.to(device),
            )

            # 모델 예측 및 Contrastive Loss 계산
            z_1 = model(view_1_nodes, view_1_edges, view_1_adjacency)
            z_2 = model(view_2_nodes, view_2_edges, view_2_adjacency)
            loss = contrastive_loss(z_1, z_2)

            test_loss += loss.item()
            test_loader_tqdm.set_postfix(loss=loss.item())

        print(f"Test Loss: {test_loss / len(test_loader):.4f}")  
        

    # 지정한 출력 디렉토리
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성

    # 저장 경로 설정
    model_save_path = output_dir / "pretrain_test_model.pt"
    loss_plot_path = output_dir / "loss_curve.png"

    # === 모델 저장 ===
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to: {model_save_path}")

    # === 학습/검증 손실 시각화 및 저장 ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"📊 Loss plot saved to: {loss_plot_path}")


    


if __name__ == "__main__":
    main()

    
