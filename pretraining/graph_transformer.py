import math
import torch
from torch import nn, einsum

List = nn.ModuleList

def softmax(x, adjacency, dim=-1, ):
    """ This calculates softmax based on the given adjacency matrix. """
    means = torch.mean(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-means) * adjacency
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum[x_exp_sum==0] = 1.
    
    return x_exp/x_exp_sum

class GatedResidual(nn.Module):
    """ This is the implementation of Eq (5), i.e., gated residual connection between block."""
    def __init__(self, dim, only_gate=False):
        super().__init__()
        self.lin_res = nn.Linear(dim, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        self.non_lin = nn.ReLU()
        self.only_gate = only_gate

    def forward(self, x, res):
        res = self.lin_res(res)
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input) # Eq (5), this is beta in the paper
        if self.only_gate: # This is for Eq (6), a case when normalizaton and non linearity is not used.
            return x * gate + res * (1 - gate)
        return self.non_lin(self.norm(x * gate + res * (1 - gate)))
    
class GraphTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, edge_dim=1, average=False):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.average = average
        inner_dim = out_dim * num_heads

        self.lin_q = nn.Linear(in_dim, inner_dim)
        self.lin_k = nn.Linear(in_dim, inner_dim)
        self.lin_v = nn.Linear(in_dim, inner_dim)
        self.lin_e = nn.Linear(edge_dim, inner_dim) 

        if edge_dim is not None:
            self.lin_e = nn.Linear(edge_dim, inner_dim)

    def forward(self, nodes, edges, adjacency):
        b, num_nodes, _ = nodes.shape
        h = self.num_heads
        d = self.out_dim

        # print(f"Input nodes: {nodes.shape}")
        
        # Q, K, V 계산
        q = self.lin_q(nodes)
        k = self.lin_k(nodes)
        v = self.lin_v(nodes)

        # print(f"Q shape after Linear: {q.shape}")

        # Multi-head attention을 위한 reshape
        q = q.view(b, num_nodes, h, d).permute(0, 2, 1, 3).reshape(-1, num_nodes, d)
        k = k.view(b, num_nodes, h, d).permute(0, 2, 1, 3).reshape(-1, num_nodes, d)
        v = v.view(b, num_nodes, h, d).permute(0, 2, 1, 3).reshape(-1, num_nodes, d)

        # Flatten for multi-head attention computation
        # q = q.reshape(b * h, num_nodes, d)
        # k = k.reshape(b * h, num_nodes, d)
        # v = v.reshape(b * h, num_nodes, d)

        # print(f"Q shape after flatten: {q.shape}")

        # Edge features (if present)
        if edges is not None:
            e = self.lin_e(edges).view(b, num_nodes, num_nodes, h, d).permute(0, 3, 1, 2, 4).reshape(-1, num_nodes, num_nodes, d)

        k = torch.unsqueeze(k, 1)
        v = torch.unsqueeze(v, 1)

        if edges is not None:
            k = k + e
            v = v + e

        # Scaled dot-product attention
        sim = einsum('b i d, b i j d -> b i j', q, k) / math.sqrt(d)
        adj = adjacency.unsqueeze(1).expand(-1, h, -1, -1).reshape(-1, num_nodes, num_nodes)
        attn = softmax(sim, adj, dim=-1)
        out = einsum('b i j, b i j d -> b i d', attn, v)

        if not self.average:
            out = out.view(-1, h, num_nodes, self.out_dim).permute(0, 2, 1, 3).reshape(-1, num_nodes, h * self.out_dim)
        else:
            out = out.view(-1, h, num_nodes, self.out_dim).permute(0, 2, 1, 3)
            out = torch.mean(out, dim=2)
        
        return out

class GraphTransformerModel(nn.Module):
    """ This is the overall architecture of the model.    
    """
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_blocks, # number of graph transformer blocks
        num_heads = 8,
        last_average=False, # whether to average or concatenate at the last block
        model_dim=None # if None, node_dim will be used as the dimension of the graph transformer block
    ):
        super().__init__()
        self.layers = List([])

        # to project the node_dim to model_dim, if model_dim is defined
        self.proj_node_dim = None
        if not model_dim:
            model_dim = node_dim
        else:
            self.proj_node_dim = nn.Linear(node_dim, model_dim) 
        assert model_dim % num_heads == 0 
        self.lin_output = nn.Linear(model_dim, 1)

        for i in range(num_blocks):
            if not last_average or i < num_blocks - 1:
                self.layers.append(List([
                    GraphTransformer(model_dim, out_dim=int(model_dim / num_heads), edge_dim=edge_dim, num_heads=num_heads),
                    GatedResidual(model_dim)
                ]))
            else:
                self.layers.append(List([
                    GraphTransformer(model_dim, out_dim=model_dim, edge_dim=edge_dim, num_heads=num_heads, average=True),
                    GatedResidual(model_dim, only_gate=True)
                ]))

    def forward(self, nodes, edges, adjacency):
        if self.proj_node_dim:
            nodes = self.proj_node_dim(nodes)
        for trans_block in self.layers:
            trans, trans_residual = trans_block
            nodes = trans_residual(trans(nodes, edges, adjacency), nodes)
        return nodes


def adjust_graph_to_model_input(graph):
    """
    Adjusts a single graph's data format to match the model's expected input format.
    :param graph: Data object with x, adjacency, etc.
    :return: Adjusted nodes, adjacency.
    """
    # Add batch dimension
    nodes = graph.x.unsqueeze(0).float()  # Shape: (1, num_nodes, feature_dim)
    edges = graph.edge.unsqueeze(0).float()  # Shape: (1, num_nodes, num_nodes)
    adjacency = graph.adjacency.unsqueeze(0)  # Shape: (1, num_nodes, num_nodes)

    if edges.shape[-1] != 1:  # If edge_dim is not 1, reshape
        edges = edges.unsqueeze(-1)  # Shape: (1, num_nodes, num_nodes, 1)

    return nodes, edges, adjacency