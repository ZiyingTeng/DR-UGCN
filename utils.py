import torch

import HDC
import RWHC
import RWIEC
import SIR
from hypergraph import Hypergraph
from SIR import *

def load_dataset(name: str, device='cpu'):
    """加载标准数据集（如cora, citeseer等）"""
    # 实现从文件加载超图的逻辑
    raise NotImplementedError("Function load_dataset is not yet implemented.")

def split_dataset(num_nodes, train_ratio=0.6, val_ratio=0.1):
    """划分训练/验证/测试集"""
    assert train_ratio + val_ratio < 1.0, "Train and validation ratios must sum to less than 1."

    indices = torch.randperm(num_nodes)
    train = indices[:int(train_ratio * num_nodes)]
    val = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
    test = indices[int((train_ratio + val_ratio) * num_nodes):]
    return train, val, test

def prepare_data(hg, lambda_, theta):
    """数据预处理"""
    # 输入验证
    assert isinstance(hg, Hypergraph), "hg 必须是 Hypergraph 对象"
    assert isinstance(lambda_, (int, float, list, tuple)), "lambda_ 必须是数值或列表/元组"
    assert isinstance(theta, (int, float, list, tuple)), "theta 必须是数值或列表/元组"

    # 静态特征
    features = torch.stack([
        torch.tensor(list(HDC.HyperDegreeCentrality.compute(hg).values())),
        torch.tensor(list(RWIEC.RWIEC.compute(hg).values())),
        torch.tensor(list(RWHC.RWHC.compute(hg).values()))
    ], dim=1).float()

    # 类型转换
    lambda_ = float(lambda_[0] if isinstance(lambda_, (list, tuple)) else lambda_)
    theta = float(theta[0] if isinstance(theta, (list, tuple)) else theta)

    X = HypergraphSIR(hg, device=hg.device)
    y = X.generate_labels(
        lambdas=lambda_,
        thetas=theta
    )  # 返回字典

    y_tensor = torch.tensor(
        [y[v] for v in sorted(y.keys())],  # 按节点索引排序
        device=hg.device
    ).float()

    adj = hg.get_adjacency_matrix_torch(weighted=True)

    return (
        y_tensor,  # 使用转换后的张量
        adj.to(hg.device) if adj is not None else None
    )
