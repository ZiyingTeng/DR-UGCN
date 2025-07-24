import numpy as np
from scipy import sparse
from typing import Dict, Union
from hypergraph import Hypergraph


class HyperDegreeCentrality:
    @staticmethod
    def compute(hg: Hypergraph,
                normalized: bool = False,
                dtype: type = np.float32) -> Dict[int, Union[int, float]]:
        """
        计算所有节点的超度中心性

        参数:
            hg: 超图对象，必须包含incidence_matrix属性
            normalized: 是否归一化结果
            dtype: 输出数据类型

        返回:
            节点ID到中心性值的映射字典
        """
        # 参数验证和自动修复
        if not isinstance(hg, Hypergraph):
            raise TypeError(f"预期Hypergraph对象，但得到{type(hg)}")

        if not hasattr(hg, 'incidence_matrix'):
            if hasattr(hg, '_build_incidence_matrix'):
                hg._build_incidence_matrix_torch()
            else:
                raise AttributeError("超图对象缺少关联矩阵且无构建方法")

        # 获取关联矩阵并确保其为稀疏格式
        M = hg.incidence_matrix
        if not sparse.issparse(M):
            M = sparse.csr_matrix(M) if M is not None else sparse.csr_matrix((0, 0))

        try:
            degrees = M.sum(axis=1).A.flatten().astype(dtype)
        except AttributeError:
            degrees = M.sum(axis=1).astype(dtype)
            if hasattr(degrees, 'flatten'):
                degrees = degrees.flatten()

        # 构建结果字典
        num_nodes = hg.incidence_matrix.shape[0] if hasattr(hg, 'incidence_matrix') else len(degrees)
        result = {v: degrees[v] for v in range(num_nodes)}

        if normalized:
            max_degree = max(result.values()) if result else 1
            max_degree = max(max_degree, 1)  # 避免除以0
            result = {k: v / max_degree for k, v in result.items()}

        return result

    @staticmethod
    def single_node(hg: Hypergraph,
                    v: int,
                    normalized: bool = False,
                    dtype: type = np.float32) -> Union[int, float]:
        """
        计算单个节点的超度中心性

        参数:
            hg: 超图对象
            v: 节点ID
            normalized: 是否返回归一化值
            dtype: 输出数据类型

        返回:
            节点的超度中心性值
        """
        # 复用compute方法提高一致性
        all_degrees = HyperDegreeCentrality.compute(hg, normalized=normalized, dtype=dtype)
        return all_degrees.get(v, 0)