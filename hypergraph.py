import numpy as np
from scipy import sparse
from typing import List, Dict, Union
import torch
from math import log2
import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")


class Hypergraph:
    def __init__(self,
                 vertices: Union[List, int] = None,
                 hyperedges: List[List] = None,
                 device: str = 'cpu'):
        """初始化超图"""
        self.device = device

        # 生成节点列表
        if isinstance(vertices, int):
            self.vertices = list(range(vertices))
            self.num_vertices = vertices
        elif isinstance(vertices, list) and all(isinstance(v, int) for v in vertices):
            self.vertices = list(vertices)
            self.num_vertices = len(self.vertices)
        else:
            raise ValueError("vertices应是一个整数或整数列表")

        # 生成超边列表
        if hyperedges is None:
            self.hyperedges = []
        elif isinstance(hyperedges, list) and all(
                isinstance(e, list) and all(isinstance(n, int) for n in e) for e in hyperedges):
            self.hyperedges = hyperedges
        else:
            raise ValueError("hyperedges应是一个超边列表")
        self.num_hyperedges = len(self.hyperedges)

        # 关联矩阵（行为节点，列为超边）
        self.incidence_matrix = self._build_incidence_matrix()
        self.incidence_matrix_torch = self._build_incidence_matrix_torch()

        # 缓存数据结构
        self._adjacency_matrix = None  # SciPy稀疏矩阵
        self._adjacency_matrix_torch = None  # PyTorch稀疏张量
        self._vertex_degrees = None
        self._vertex_degrees_torch = None
        self._hyperedge_sizes = None

    def add_hyperedge(self, nodes: List[int]) -> None:
        """添加一条超边"""
        if not all(node in self.vertices for node in nodes):
            raise ValueError("所有节点必须存在于当前的顶点列表中")

        self.hyperedges.append(nodes)
        self.num_hyperedges += 1
        self._invalidate_cache() # 否则关联矩阵会过期

    def remove_hyperedge(self, index: int) -> None:
        """移除指定索引的超边"""
        del self.hyperedges[index]
        self.num_hyperedges -= 1
        self._invalidate_cache()


    def get_vertex_degree_torch(self, v: int) -> torch.Tensor:
        """如果要与GCN结合"""
        if self._vertex_degrees_torch is None:
            self._vertex_degrees_torch = torch.sparse.sum(
                self.incidence_matrix_torch, dim=1).to_dense()
        return self._vertex_degrees_torch[v]

    def get_hyperedge_size(self, e_idx: int) -> int:
        """获取超边的大小（质量）"""
        if self._hyperedge_sizes is None:
            self._hyperedge_sizes = np.array(self.incidence_matrix.sum(axis=0)).flatten()
            # flatten()将二维矩阵转换为一维数组
        return self._hyperedge_sizes[e_idx] if e_idx < len(self._hyperedge_sizes) else 0

    def _invalidate_cache(self):
        """清空缓存的数据结构"""
        self._adjacency_matrix = None
        self._adjacency_matrix_torch = None
        self._vertex_degrees = None
        self._vertex_degrees_torch = None
        self._hyperedge_sizes = None
        self.incidence_matrix = self._build_incidence_matrix()
        self.incidence_matrix_torch = self._build_incidence_matrix_torch()  # 全部重置后立即重建关联矩阵

    def _build_incidence_matrix(self) -> sparse.csr_matrix:
        """构建超图关联矩阵"""
        rows, cols, data = [], [], []
        for e_idx, nodes in enumerate(self.hyperedges):
            for v in nodes:
                rows.append(v)
                cols.append(e_idx)
                data.append(1)
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_vertices, self.num_hyperedges)
        )

    def _build_incidence_matrix_torch(self) -> torch.Tensor:

        indices = []
        for e_idx, nodes in enumerate(self.hyperedges):
            indices.append(torch.stack([
                torch.tensor(nodes, device=self.device),
                torch.full((len(nodes),), e_idx, device=self.device)
            ], dim=0))
        indices = torch.cat(indices, dim=1)
        values = torch.ones(indices.shape[1], device=self.device)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.num_vertices, self.num_hyperedges)
        ).coalesce()

    def get_adjacency_matrix(self, weighted: bool = False) -> sparse.csr_matrix:
        """获取超图邻接矩阵"""
        if self._adjacency_matrix is None:
            H = self.incidence_matrix
            HHt = H.dot(H.T)
            diag = HHt.diagonal()
            self._adjacency_matrix = HHt - sparse.diags(diag)

            if weighted:
                sizes = np.array([len(e) for e in self.hyperedges])
                sizes = np.where(sizes > 1, sizes, 1)  # 避免除以0
                W = sparse.diags(1.0 / (sizes - 1))
                self._adjacency_matrix = H.dot(W).dot(H.T) - sparse.diags(
                    (H.dot(W).dot(H.T)).diagonal()
                )
        return self._adjacency_matrix

    def get_adjacency_matrix_torch(self, weighted: bool = False) -> torch.Tensor:
        """如果超图节点数很大"""
        if self._adjacency_matrix_torch is None:
            H = self.incidence_matrix_torch
            HHt = torch.sparse.mm(H, H.t())
            diag = torch.sparse.sum(HHt, dim=1).to_dense()

            # 将对角矩阵转为稀疏格式
            indices = torch.arange(len(diag), device=self.device).unsqueeze(0).repeat(2, 1)
            diag_sparse = torch.sparse_coo_tensor(
                indices,
                diag,
                size=HHt.size(),
                device=self.device
            )
            self._adjacency_matrix_torch = HHt - diag_sparse  # 稀疏-稀疏

        if weighted:
            sizes = torch.tensor([len(e) for e in self.hyperedges],
                                 device=self.device).float()
            W = torch.diag(1.0 / (sizes - 1))
            H = self.incidence_matrix_torch.to_dense()  # 加权计算需要稠密矩阵
            return H @ W @ H.t() - torch.diag_embed((H @ W @ H.t()).diag())
        return self._adjacency_matrix_torch

    @classmethod
    def from_dataset(cls, name: str):
        """从标准数据集加载超图"""
        # 实现从标准数据集（如cora, citeseer等）加载
        pass

    # @classmethod
    # def from_random(cls, n_vertices: int, n_hyperedges: int,
    #                 size_range: tuple = (2, 5), device: str = 'cpu'):
    #     """随机生成超图（用于测试）"""
    #     # 控制超边的大小 2~5个节点
    #     hyperedges = []
    #     for _ in range(n_hyperedges):
    #         size = np.random.randint(*size_range)
    #         nodes = np.random.choice(n_vertices, size, replace=False) # 不重复随机选择节点
    #         hyperedges.append(sorted(nodes))
    #     return cls(vertices=n_vertices, hyperedges=hyperedges, device=device)

    def remove_isolated_nodes(self) -> 'Hypergraph':
        """
        移除孤立节点（属于单节点超边且无其他连接的节点）
        返回: 新超图对象（原对象不变）
        """

        isolated = set()
        for v in range(self.num_vertices):

            incident_edges = [e for e in self.hyperedges if v in e]
            if len(incident_edges) == 1 and len(incident_edges[0]) == 1:
                isolated.add(v)

        # 构建新超图
        new_vertices = [v for v in self.vertices if v not in isolated]
        new_hyperedges = [
            [v for v in e if v not in isolated]
            for e in self.hyperedges
            if len(set(e) - isolated) > 1  # 移除单节点超边
        ]
        return Hypergraph(vertices=new_vertices, hyperedges=new_hyperedges)

    # @classmethod
    # def from_dataset(cls, name: str, device: str = 'cpu'):
    #     """加载标准超图数据集"""
    #     if name == "House":
    #         hyperedges = load_house_data()  # 需实现数据加载函数
    #     elif name == "Senate":
    #         hyperedges = load_senate_data()
    #     elif name == "Cora-CA":
    #         hyperedges = load_cora_data()
    #     elif name == "Citeseer":
    #         hyperedges = load_citeseer_data()
    #     else:
    #         raise ValueError(f"未知数据集: {name}")
    #
    #     # 自动计算节点总数（假设节点编号从0开始连续）
    #     all_nodes = set().union(*hyperedges)
    #     num_nodes = max(all_nodes) + 1 if all_nodes else 0
    #
    #     return cls(vertices=num_nodes, hyperedges=hyperedges, device=device)

    @classmethod
    def generate_test_hypergraph(cls,
                                 num_nodes: int = 300,
                                 num_hyperedges: int = 150,
                                 size_range: tuple = (2, 20),
                                 centrality_nodes: int = 10,
                                 hub_strength=10,
                                 device: str = 'cpu'):
        """生成格式严格规范的测试超图"""

        assert num_nodes >= 2, "至少需要2个节点"
        assert size_range[0] >= 2, "超边最小大小必须≥2"
        assert num_hyperedges >= centrality_nodes * hub_strength, "超边数量不足以保证hub_strength"

        hyperedges = []
        all_nodes = list(range(num_nodes))
        hub_nodes = np.random.choice(all_nodes, centrality_nodes, replace=False).tolist()

        # 生成普通超边
        for _ in range(num_hyperedges - centrality_nodes * hub_strength):
            size = np.random.randint(*size_range)
            nodes = sorted(np.random.choice(all_nodes, size, replace=False).tolist())
            hyperedges.append(nodes)

        # 生成关键节点超边
        for node in hub_nodes:
            available_nodes = [n for n in all_nodes if n != node]
            max_possible_size = min(len(available_nodes) + 1, size_range[1])
            for _ in range(hub_strength):
                size = np.random.randint(max(2, size_range[0]), max_possible_size)
                others = sorted(np.random.choice(available_nodes, size - 1, replace=False).tolist())
                hyperedges.append([node] + others)

        # 格式验证
        assert all(isinstance(e, list) for e in hyperedges), "超边必须是列表"
        assert all(isinstance(n, int) for e in hyperedges for n in e), "节点ID必须是整数"
        assert all(len(e) >= 2 for e in hyperedges), "超边至少包含2个节点"

        hypergraph = cls(vertices=num_nodes, hyperedges=hyperedges, device=device)

        # 确保关联矩阵存在
        if not hasattr(hypergraph, 'incidence_matrix'):
            hypergraph._build_incidence_matrix_torch()

        return hypergraph



def compute_effective_distance(hypergraph_or_edges, delta: float = 1e-10) -> np.ndarray:
    """
    计算超边间的有效距离矩阵（含不对称性设计）

    D: 有效距离矩阵（|E|×|E|），D[i,j]表示e_i到e_j的距离
    """
    hyperedges = hypergraph_or_edges.hyperedges if isinstance(hypergraph_or_edges, Hypergraph) else hypergraph_or_edges
    num_hyperedges = len(hyperedges)
    D = np.full((num_hyperedges, num_hyperedges), np.inf)# 初始距离定义为无穷大

    # 预计算所有超边的节点集合和大小（提升性能）
    edge_sets = [set(e) for e in hyperedges]
    edge_sizes = np.array([len(e) for e in hyperedges])

    for i in range(num_hyperedges):
        for j in range(num_hyperedges):
            if i == j:
                D[i, j] = 0
                continue
            intersection = edge_sets[i] & edge_sets[j]
            union = edge_sets[i] | edge_sets[j]
            if not intersection:
                continue
            jaccard = len(intersection) / len(union)           # Jaccard相似性分量（超边重叠程度）
            directional = len(intersection) / edge_sizes[i]    # 计算方向性分量
            p_ij = jaccard * directional                       # 转移概率
            D[i, j] = 1 - log2(p_ij + delta)                   # 加平滑系数delta防止log(0)
    return D


def compute_effective_distance_torch(hypergraph, delta: float = 1e-10) -> torch.Tensor:
    device = hypergraph.device
    hyperedges = [torch.tensor(e, device=device) for e in hypergraph.hyperedges]
    num_hyperedges = len(hyperedges)

    D = torch.full((num_hyperedges, num_hyperedges), float('inf'), device=device)

    edge_sizes = torch.tensor([len(e) for e in hyperedges], device=device).float()

    for i in range(num_hyperedges):
        for j in range(num_hyperedges):
            if i == j:
                D[i, j] = 0
                continue

            intersection = torch.intersect1d(hyperedges[i], hyperedges[j])
            union = torch.unique(torch.cat([hyperedges[i], hyperedges[j]]))

            if len(intersection) == 0:
                continue

            jaccard = len(intersection) / len(union)
            directional = len(intersection) / edge_sizes[i]
            p_ij = jaccard * directional
            D[i, j] = 1 - torch.log2(torch.tensor(p_ij + delta, device=device))

    return D

