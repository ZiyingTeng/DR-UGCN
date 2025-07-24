import numpy as np
from typing import Union, List, Dict
import torch
from hypergraph import Hypergraph  # 导入超图类
from hypergraph import compute_effective_distance


def compute_gravity_forces(
        data: Union[Hypergraph, List[List[int]]],  # 接受超图对象或原始超边
        vertex_degrees: Union[Dict[int, int], str] = 'auto',
        delta: float = 1e-10
) -> np.ndarray:
    """
    智能计算超边作用力（自动处理输入数据）

    参数:
        data: 可接受 Hypergraph对象 或 超边列表
        vertex_degrees: 
          - 'auto'（默认）自动从hypergraph计算
          - 或直接传递度数字典 {0:2, 1:3, ...}
        delta: 平滑系数
    """
    # 自动解析超边数据
    if isinstance(data, Hypergraph):
        hyperedges = data.hyperedges
        if vertex_degrees == 'auto':
            vertex_degrees = {v: data.get_vertex_degree_torch(v)
                              for v in range(data.num_vertices)}
    else:
        hyperedges = data
        if vertex_degrees == 'auto':
            raise ValueError("原始超边需手动提供vertex_degrees")

    # 检查度数完整性
    required_nodes = set().union(*hyperedges)
    missing_nodes = required_nodes - vertex_degrees.keys()
    if missing_nodes:
        raise ValueError(f"缺失节点度数: {missing_nodes}")

    # 计算质量
    qualities = [
        np.sqrt(len(e) ** 2 + np.mean([vertex_degrees[v] for v in e]) ** 2)
        for e in hyperedges
    ]

    # 计算有效距离
    D = compute_effective_distance(hyperedges, delta)

    # 计算作用力矩阵
    G = np.zeros((len(hyperedges), len(hyperedges)))
    for i in range(len(hyperedges)):
        for j in range(len(hyperedges)):
            if i == j or D[i, j] == np.inf:
                continue
            G[i, j] = qualities[i] * qualities[j] / (D[i, j] ** 2 + delta)

    return np.sum(G, axis=1)  # 返回各超边重要性




class RWHC:

    @staticmethod
    def compute(hg: Hypergraph, delta=1e-5, max_iter=1000, eps=1e-6) -> Dict[int, float]:
        """
        流程:
            1. 使用compute_gravity_forces初始化热力值
            2. 构建拉普拉斯矩阵
            3. 热扩散迭代
        """
        # 计算初始热力值
        G_e = compute_gravity_forces(hg, delta=delta)  # 超边重要性向量

        # 按权重分配到节点
        T = np.zeros(hg.num_vertices, dtype=np.float64)
        for v in range(hg.num_vertices):
            total_degree = 0.0
            for e_idx, e in enumerate(hg.hyperedges):
                if v in e:
                    total_degree = sum(hg.get_vertex_degree_torch(u) for u in e)
                    if total_degree > 0:
                        T[v] += G_e[e_idx] * hg.get_vertex_degree_torch(v) / total_degree


        L = RWHC._build_laplacian(hg)
        return RWHC._diffuse(T, L, max_iter, eps)

    @staticmethod
    def _build_laplacian(hg) -> np.ndarray:
        """构建随机游走拉普拉斯矩阵"""
        from RWIEC import RWIEC  # 避免循环引用
        P = RWIEC._build_P(hg)
        pi = RWIEC._solve_pi(P, 1000, 1e-6)  # 变成公式里的Phi
        return np.diag(pi) - (np.diag(pi) @ P + P.T @ np.diag(pi)) / 2  # L

    @staticmethod
    def _diffuse(T: np.ndarray, L: np.ndarray, max_iter: int, eps: float) -> Dict[int, float]:
        """热扩散方程求解 """
        for _ in range(max_iter):
            delta = -L @ T
            T_new = T + delta
            if np.linalg.norm(T_new - T) < eps:
                break
            T = T_new
        return {v: T[v] for v in range(len(T))}

