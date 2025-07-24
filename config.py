import numpy as np
import torch


class Config:
    def __init__(self,seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 数据集参数
        self.datasets = ['House', 'Senate', 'Cora-CA', 'Citeseer']
        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

        # 传播模型参数扫描范围
        self.lambdas = np.linspace(0.001, 0.2,2)  # 感染强度λ
        self.thetas = [0.5,0.75]  # 非线性程度θ

        # 算法比较列表
        self.algorithms = ['DR-UGCN', 'HDC', 'SC', 'DC', 'BC']

        # 实验参数
        self.top_k_ratio = 0.05  # 选取前5%的关键节点
        self.num_simulations = 10  # 每个参数组合的模拟次数
        self.max_iter = 50  # 传播模型最大迭代次数

        self.visualize_curves = True
        self.curve_simulations = 5

        self.plot_params = {
            "line_styles": {"DR-UGCN": "--", "default": "-"},
            "markers": {"DR-UGCN": "o", "HDC": "s", "SC": "D", "DC": "^", "BC": "v"},
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        }

        self.run_nonlinear_experiment = True
