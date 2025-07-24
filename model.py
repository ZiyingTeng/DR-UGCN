import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import Config
from hypergraph import Hypergraph
from encoder import UniGEncoder
from torch_geometric.utils import negative_sampling
import numpy as np

from utils import prepare_data


class GraphConvolution(nn.Module):
    """
    GCN层实现 (D^{-1/2}AD^{-1/2}XW + b)
    输入:
        - X: (num_nodes, in_features)
        - adj: (num_nodes, num_nodes) 稀疏张量
    输出:
        - (num_nodes, out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.weight.size(0), "Input features' dimension mismatch with weight matrix."
        support = torch.mm(x, self.weight)  # XW
        output = torch.sparse.mm(adj, support)  # AXW
        if self.bias is not None:
            return output + self.bias
        return output



class DRUGCN(nn.Module):
    """
    完整DR-UGCN模型
    流程:
        1. UniG-Encoder编码 (含MLP)
        2. 两层GCN
        3. 两层MLP预测重要性
    输入:
        - hg: 超图对象
        - adj: 归一化邻接矩阵
        - node_features: 可选 (默认自动生成3维特征)
    输出:
        - 节点重要性得分 (num_nodes,)
    """

    def __init__(self,
                 input_dim: int = 3,
                 encoder_mlp_dims: Tuple[int, int] = (128, 3),
                 gcn_hidden_dim: int = 128,
                 output_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()

        self.dropout = dropout

        # 1. UniG-Encoder (包含内置MLP)
        self.encoder = UniGEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_mlp_dims,
            dropout=dropout
        )

        # 2. 两层GCN
        self.gcn1 = GraphConvolution(encoder_mlp_dims[-1], gcn_hidden_dim)  # 使用隐层维度 即保证经过编码器后特征向量维度不变
        self.gcn2 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim)

        # 3. 两层MLP
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden_dim // 2, output_dim)
        )

    def forward(self,
                hg: Hypergraph,
                adj: torch.Tensor,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if node_features is None:
            node_features = UniGEncoder.get_initial_features(hg).to(adj.device)

        # 维度保持 |V| × 3
        x = self.encoder(hg, node_features)

        # 两层GCN
        x = F.relu(self.gcn1(x, adj))  # |V| × gcn_hidden_dim
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, adj))  # |V| × gcn_hidden_dim

        return self.mlp(x).squeeze(-1)  # |V| × 1 → |V|


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """对称归一化邻接矩阵 D^{-1/2}AD^{-1/2}"""
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.where(degrees > 0, degrees, torch.ones_like(degrees)) ** -0.5
    d_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = d_inv_sqrt @ adj.to_dense() @ d_inv_sqrt

    # 转回稀疏张量
    indices = adj_norm.nonzero().t()
    values = adj_norm[indices[0], indices[1]]
    return torch.sparse_coo_tensor(
        indices, values, adj.size(),
        device=adj.device
    ).coalesce()


class DRUGCNTrainer:
    def __init__(self, model, hg):
        self.model = model
        self.hg = hg
        lambdas = Config().lambdas
        self.lambdas = lambdas
        thetas = Config().thetas
        self.thetas = thetas
        self.config = Config()

    def train(self, epochs=100, lr=0.01):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        all_losses = []

        for epoch in range(epochs):
            # 动态采样传播参数
            lambda_val = np.random.uniform(*self.config.lambdas[:2])
            theta_val = np.random.uniform(*self.config.thetas[:2])


            y, adj = prepare_data(
                self.hg,
                lambda_=float(lambda_val),
                theta=float(theta_val)
            )

            # 修改调用方式：显式传递超图和邻接矩阵
            pred = self.model(self.hg, adj)  # 第一个参数必须是超图对象
            loss = F.mse_loss(pred.squeeze(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f} (λ={lambda_val:.2f}, θ={theta_val:.2f})")

        return all_losses