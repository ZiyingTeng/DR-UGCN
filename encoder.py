# 前向投影：将节点特征投影到"节点-超边联合空间"
#         Z^0=QX Q的上半为原始节点特征，下半为关联矩阵的转置
# MLP：在联合空间进行非线性特征融合
# 反向投影：将融合后的特征重新映射回节点空间，就是生成新的
# 归一化：消除超边大小对权重的影响


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from hypergraph import Hypergraph


class UniGEncoder(nn.Module):

    def __init__(self,
                 input_dim: int = 3,
                 hidden_dims: Tuple[int, int] = (128, 3),
                 dropout: float = 0.5):   # 小规模dropout可以高一点
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )

        # 投影矩阵缓存
        self.Q = None
        self.Q_norm = None

    def build_projection_matrix(self, hg: Hypergraph) -> torch.Tensor:
        """
        构建投影矩阵Q
        返回:
            Q矩阵 (|V_p| × |V|)
            V_p = V_N ∪ E_N (节点与超边的并集)
        """
        num_nodes = hg.num_vertices
        num_edges = len(hg.hyperedges)

        # 单位矩阵(节点特征)
        Q_v = torch.eye(num_nodes)

        Q_e = torch.zeros((num_edges, num_nodes), device=hg.device)
        for e_idx, e in enumerate(hg.hyperedges):
            Q_e[e_idx, list(e)] = 1

        # 拼接
        Q = torch.cat([Q_v, Q_e], dim=0)
        return Q

    def normalize_projection(self, Q: torch.Tensor) -> torch.Tensor:
        """使返回的Q_norm里每个超边权重总和为1"""
        # 节点
        num_nodes = Q.shape[1]
        Q_norm = Q.clone()

        # 超边
        edge_part = Q[num_nodes:]  # 具体数值（1）
        row_sums = edge_part.sum(dim=1, keepdim=True)
        Q_norm[num_nodes:] = edge_part / (row_sums + 1e-10)  # 归一化后具体数值

        return Q_norm

    def forward(self,
                hg: Hypergraph,
                node_features: torch.Tensor) -> torch.Tensor:
        """
        流程:
            1. 前向投影: Z^(0) = QX
            2. MLP变换: Z^(l) = MLP(Z^(0))
            3. 反向投影: Z = Q^T Z^(l)

        node_features: 原始特征 (|V| × C0)

        返回:
            节点嵌入 (|V| × Ck)
        """
        # 构建并缓存投影矩阵
        if self.Q is None:
            self.Q = self.build_projection_matrix(hg).to(node_features.device)
            self.Q_norm = self.normalize_projection(self.Q)

        # 前向投影
        Z0 = torch.mm(self.Q_norm, node_features)

        # MLP
        Zl = self.mlp(Z0)

        # 反向投影
        Z = torch.mm(self.Q_norm.t(), Zl)

        return Z

    @staticmethod
    def get_initial_features(hg: Hypergraph) -> torch.Tensor:
        """
        生成初始特征矩阵 (|V| × 3)
        """
        from HDC import HyperDegreeCentrality
        from RWIEC import RWIEC
        from RWHC import RWHC

        hdc = HyperDegreeCentrality.compute(hg)
        rwiec = RWIEC.compute(hg)
        rwhc = RWHC.compute(hg)

        features = torch.tensor([
            [hdc[v], rwiec[v], rwhc[v]]
            for v in sorted(hdc.keys())
        ], dtype=torch.float32)

        # 行归一化 (MinMax)
        features = (features - features.min(0)[0]) / \
                   (features.max(0)[0] - features.min(0)[0] + 1e-10)

        return features

