import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import expm
from typing import Dict

import HDC
from hypergraph import Hypergraph


class BetweennessCentrality:
    """超图上的Betweenness Centrality实现（基于 clique 扩展）"""

    @staticmethod
    def compute(hg: Hypergraph, normalized: bool = True) -> Dict[int, float]:
        """
        计算超图节点的介数中心性
        实现步骤:
            1. 将超图转为普通图（clique扩展）
            2. 使用Brandes算法计算BC值
        """
        adj = hg.get_adjacency_matrix()
        n = adj.shape[0]
        bc = np.zeros(n)

        # Brandes算法优化实现
        for s in range(n):
            # 最短路径计算
            S, P, sigma = [], [[] for _ in range(n)], np.zeros(n)
            sigma[s] = 1
            D = [-1] * n
            D[s] = 0
            Q = [s]

            while Q:
                v = Q.pop(0)
                S.append(v)
                for w in np.where(adj[v].toarray().flatten() > 0)[0]:
                    if D[w] < 0:
                        Q.append(w)
                        D[w] = D[v] + 1
                    if D[w] == D[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # 累加依赖值
            delta = np.zeros(n)
            while S:
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    bc[w] += delta[w]

        # 归一化处理
        if normalized and n > 2:
            bc /= ((n - 1) * (n - 2) / 2)

        return {v: float(bc[v]) for v in range(n)}


class SubgraphCentrality:
    """超图上的Subgraph Centrality实现（基于邻接矩阵指数）"""

    @staticmethod
    def compute(hg: Hypergraph) -> Dict[int, float]:
        """
        计算超图节点的子图中心性
        公式: SC(i) = [exp(A)]_ii
        """
        adj = hg.get_adjacency_matrix().astype(float)
        n = adj.shape[0]

        # 使用expm计算矩阵指数（小规模图）
        if n <= 1000:
            exp_A = expm(adj.toarray())
        else:
            # 大规模图近似计算（Lanczos方法）
            from scipy.sparse.linalg import expm_multiply
            exp_A = np.zeros((n, n))
            for i in range(n):
                v = np.zeros(n)
                v[i] = 1
                exp_A[:, i] = expm_multiply(adj, v)

        return {v: float(exp_A[v, v]) for v in range(n)}

def run_baseline_algorithm(hg, algorithm):
    """运行基准算法获取节点重要性排序"""
    if algorithm == 'HDC':
        scores = HDC.HyperDegreeCentrality.compute(hg)
    elif algorithm == 'DC':
        adj = hg.get_adjacency_matrix()
        degrees = np.array(adj.sum(axis=1)).flatten()
        scores = {v: degrees[v] for v in range(hg.num_vertices)}
    elif algorithm == 'BC':
        scores = BetweennessCentrality.compute(hg)
    elif algorithm == 'SC':
        scores = SubgraphCentrality.compute(hg)
    return scores