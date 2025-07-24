import numpy as np
from scipy import sparse
from typing import Dict, List
from math import log
from hypergraph import Hypergraph


class RWIEC:

    @staticmethod
    def compute(hg: Hypergraph, max_iter=1000, eps=1e-6) -> Dict[int, float]:
        """
        计算所有节点的RWIEC值
        流程:
            1. 构建转移矩阵P
            2. 计算平稳分布π
            3. 融合动态/静态熵
        """
        P = RWIEC._build_P(hg)
        pi = RWIEC._solve_pi(P, max_iter, eps)
        return RWIEC._calc_entropy(hg, P, pi)

    @staticmethod
    def _build_P(hg: Hypergraph) -> np.ndarray:
        N = hg.num_vertices
        P = np.zeros((N, N))
        degrees = [hg.get_vertex_degree_torch(v) for v in range(N)]

        for v in range(N):
            # 获取包含v的所有超边索引
            edge_indices = [i for i, e in enumerate(hg.hyperedges) if v in e]
            # v所在所有超边包含的节点数量和
            sum_edge_sizes = sum(len(hg.hyperedges[i]) for i in edge_indices)

            if sum_edge_sizes == 0:
                continue

            for e_idx in edge_indices:
                e = hg.hyperedges[e_idx]
                p_e_given_v = len(e) / sum_edge_sizes  # p(e|v)

                # u的超度 / 超边上所有节点的超度之和
                sum_degrees_in_e = sum(degrees[u] for u in e)
                for u in e:
                    if u != v and sum_degrees_in_e > 0:
                        P[v, u] += p_e_given_v * (degrees[u] / sum_degrees_in_e)  # 节点v到节点u的转移概率

        # 归一化保证每行和为1
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以零
        P /= row_sums
        return P

    @staticmethod
    def _solve_pi(P: np.ndarray, max_iter: int, eps: float) -> np.ndarray:
        """幂迭代法求解平稳分布"""
        pi = np.ones(P.shape[0]) / P.shape[0]  # 初始为均匀分布
        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < eps:
                break  # 直到平稳
            pi = pi_new
        return pi

    @staticmethod
    def _calc_entropy(hg: Hypergraph, P: np.ndarray, pi: np.ndarray) -> Dict[int, float]:
        """计算IR"""
        centrality = {}
        for v in range(hg.num_vertices):
            # 动态转移熵 (P[v]中非零元素的熵)
            dynamic = -np.sum(P[v][P[v] > 0] * np.log(P[v][P[v] > 0] + 1e-10)) / hg.num_vertices
            # 静态平稳分布熵
            static = -pi[v] * log(pi[v] + 1e-10)
            centrality[v] = -(dynamic + static)  # IR
        return centrality

