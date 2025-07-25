from torch_geometric.data import Data, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Batch
import random
import torch

# MaskedGraphDataset 클래스
class MaskedGraphDataset(Dataset):
    def __init__(self, dataset, mask_ratio=0.3, num_masks=2):
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks

    def mask_nodes(self, data):
        """Randomly mask a subset of nodes."""
        num_nodes = data.x.size(0)
        mask_size = int(self.mask_ratio * num_nodes)
        masked_indices = random.sample(range(num_nodes), mask_size)

        # Masked nodes are set to zero
        masked_data = []
        for _ in range(self.num_masks):
            masked_x = data.x.clone()
            masked_x[masked_indices] = 0  # Masked nodes are set to 0.

            # Create new Data object with the masked nodes
            masked_data.append(
                Data(
                    x=masked_x,  # Masked node features
                    edge=data.edge.clone() if data.edge is not None else None,  # 엣지 정보 그대로 복사
                    adjacency=data.adjacency.clone() if data.adjacency is not None else None,  # 가중치 인접 행렬 그대로 복사
                    batch=data.batch.clone() if hasattr(data, "batch") and data.batch is not None else None
                )
            )
        return masked_data

    def __getitem__(self, index):
        data = self.dataset[index]
        return self.mask_nodes(data)

    def __len__(self):
        return len(self.dataset)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.similarity_matrices = []  # 유사도 행렬 저장

    def forward(self, z_i, z_j):
        """Compute contrastive loss."""
        # Reduce dimension by averaging over the node dimension
        z_i = z_i.mean(dim=1)  # (batch_size, embedding_dim)
        z_j = z_j.mean(dim=1)  # (batch_size, embedding_dim)

        # Concatenate the embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, embedding_dim)
        z = F.normalize(z, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.t())  # (2 * batch_size, 2 * batch_size)
        labels = torch.cat([
            torch.arange(z_i.size(0)),
            torch.arange(z_j.size(0))
        ])

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = torch.eye(labels.shape[0], device=labels.device).bool()

        labels = labels[~mask].view(labels.shape[0], -1) #
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) #

        self.similarity_matrices.append(similarity_matrix.detach().cpu().numpy())

        # Positive and negative similarity
        positives = similarity_matrix[labels.bool()].view(labels.size(0), -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.size(0), -1)

        # Calculate logits and apply cross-entropy loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        logits = logits / self.temperature

        # return F.cross_entropy(logits, labels, reduction='mean')
        return F.cross_entropy(logits, labels)