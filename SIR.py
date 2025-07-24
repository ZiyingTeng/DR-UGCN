import torch
import numpy as np

from config import Config
from hypergraph import Hypergraph
from typing import Dict, List, Set
import matplotlib.pyplot as plt
from collections import defaultdict


class HypergraphSIR:
    def __init__(self, hg, device='cpu',seed=42):

        self.device = device
        self.rng = np.random.RandomState(seed)

        if isinstance(hg.vertices,int):
            self.vertices = list(range(hg.vertices))
            self.num_vertices = hg.vertices
        else:
            self.vertices = list(hg.vertices)
            self.num_vertices = len(self.vertices)

        self.hyperedges = hg.hyperedges

        self.hyperedges_dict = {
            node: [i for i, e in enumerate(self.hyperedges) if node in e] for node in self.vertices
        }
        self.hg = hg
        self.M_max = max(len(e) for e in hg.hyperedges)

    def _get_neighbor_hyperedges(self, node: int) -> Set[int]:
        return set(self.hyperedges_dict.get(node, set()))

    def simulate(self, initial_infected: List[int],
                 gamma: float = 0.1,
                 thetas: float = None,
                 lambdas: float = None,
                 track_curve: bool = False) -> Dict[str, float]:
        """
        增强的单次模拟：
        track_curve参数控制是否返回传播曲线
        """
        infected = set(initial_infected)
        recovered = set()
        curve = []
        max_iter = Config().max_iter
        lambdas = float(Config().lambdas[0] if lambdas is None else lambdas)
        thetas = float(Config().thetas[0] if thetas is None else thetas)

        for _ in range(max_iter):
            new_infected = set()
            active_edges = set()

            for v in infected:
                active_edges.update(self._get_neighbor_hyperedges(v))

            valid_edges = [e_idx for e_idx in active_edges if e_idx < len(self.hg.hyperedges)]
            for e_idx in valid_edges:
                e_nodes = set(self.hg.hyperedges[e_idx])
                infected_in_e = infected & e_nodes
                i = len(infected_in_e)
                p_infect = min(lambdas * (i ** thetas), 1.0)

                for v in e_nodes - infected - recovered:
                    if self.rng.random() < p_infect:
                        new_infected.add(v)

            newly_recovered = {v for v in infected if np.random.random() < gamma}
            recovered.update(newly_recovered)
            infected.update(new_infected - newly_recovered)

            if track_curve:
                curve.append(len(infected) / len(self.vertices))

            if not new_infected:
                break

        results = {
            'infection_rate': len(infected) / len(self.vertices),
            'curve': curve if track_curve else [],
            'total_infected': len(infected),
            'total_recovered': len(recovered)
        }
        return results

    def generate_labels(self, hg=None, lambdas=None, thetas=None, num_samples=None, config=None):
        if num_samples is None:
            if config is None or not hasattr(config, 'num_simulations'):
                num_samples = 10
            else:
                num_samples = max(1, int(config.num_simulations))
        else:
            num_samples = max(1, int(num_samples))

        lambdas = float(lambdas[0] if isinstance(lambdas, (list, tuple)) else lambdas)
        thetas = float(thetas[0] if isinstance(thetas, (list, tuple)) else thetas)

        labels = {}
        for v in range(len(self.vertices)):
            total = 0.0
            for _ in range(num_samples):
                initial_infected = [v]
                results = self.simulate(
                    initial_infected,
                    lambdas=lambdas,
                    thetas=thetas
                )
                total += results['infection_rate']
            labels[v] = total / num_samples
        return labels

    @staticmethod
    def plot_curves(curves: List[List[float]], title: str):
        """标准化处理不同长度的传播曲线并绘制"""
        if not curves:
            raise ValueError("输入曲线列表为空")

        # 计算最大步长并统一填充曲线
        max_len = max(len(c) for c in curves)
        curves_padded = []
        for curve in curves:
            if not curve:  # 处理空曲线
                padded = [0.0] * max_len
            else:
                # 用最后值填充至最大长度（模拟稳态）
                padded = curve + [curve[-1]] * (max_len - len(curve))
            curves_padded.append(padded)

        curves_array = np.array(curves_padded)

        plt.figure(figsize=(10, 6))

        # 个体曲线（半透明）
        for i, curve in enumerate(curves_array):
            plt.plot(curve, color='blue', alpha=0.2, linewidth=1)

        # 平均曲线（加粗黑线）
        mean_curve = np.mean(curves_array, axis=0)
        plt.plot(mean_curve, 'k-', linewidth=2, label='Average')

        # 添加标准差阴影
        std_curve = np.std(curves_array, axis=0)
        plt.fill_between(
            range(max_len),
            mean_curve - std_curve,
            mean_curve + std_curve,
            color='gray',
            alpha=0.3,
            label='±1 Std Dev'
        )

        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Infection Rate', fontsize=12)
        plt.title(f'SIR Propagation Curves\n{title}', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(0, max_len - 1)
        plt.ylim(0, 1.05)  # 留出顶部空间

        return plt
