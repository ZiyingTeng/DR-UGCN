import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from baselines import run_baseline_algorithm
from config import Config
from hypergraph import Hypergraph
from model import DRUGCN, DRUGCNTrainer, normalize_adjacency
from utils import prepare_data
from SIR import HypergraphSIR
import random


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup_cuda():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_nonlinear_experiment(cfg, hg, device):
    """运行非线性参数实验"""
    print("\nRunning nonlinear parameter experiment...")
    algorithms = ['BC', 'SC', 'HDC', 'DC', 'DR-UGCN']
    non_linear_params = np.linspace(0, 1, 10)
    results = {algo: [] for algo in algorithms}

    drugcn_model = DRUGCN().to(device)
    drugcn_trainer = DRUGCNTrainer(drugcn_model, hg)

    for non_linear_param in non_linear_params:
        print(f"\nProcessing non-linear param: {non_linear_param:.1f}")

        for algorithm in algorithms[:-1]:
            scores = run_baseline_algorithm(hg, algorithm)
            max_score = max(scores.values()) if scores else 1.0
            infected = sum(np.random.rand() < (v / max_score) ** non_linear_param
                           for v in scores.values())
            results[algorithm].append(infected / hg.num_vertices)

        # 处理DR-UGCN
        drugcn_trainer.train(epochs=100, lr=0.01)
        X, _, adj = prepare_data(hg, lambdas=0.005, thetas=0.75)
        with torch.no_grad():
            pred = drugcn_model(X, adj)
        results['DR-UGCN'].append((pred > 0.5).float().mean().item())

    plt.figure(figsize=(10, 6))
    for algo, values in results.items():
        plt.plot(non_linear_params, values, marker='o', label=algo)
    plt.title('Nonlinear Parameter Experiment')
    plt.xlabel('Nonlinear Parameter')
    plt.ylabel('Infection Rate')
    plt.legend()
    plt.grid()
    plt.savefig("nonlinear_experiment.png")
    plt.close()
    cleanup_cuda()


def train_and_save_model(cfg, hg, device, dataset_name):
    """训练并保存模型"""
    if 'DR-UGCN' in cfg.algorithms:
        print("Training DR-UGCN model...")
        model = DRUGCN().to(device)
        losses = model.train()

        plt.figure()
        plt.plot(losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{dataset_name}_training_loss.png")
        plt.close()
        cleanup_cuda()


def plot_results(x, y_dict, title, xlabel, ylabel, filename):
    """通用绘图函数"""
    plt.figure(figsize=(10, 6))
    for label, y in y_dict.items():
        plt.plot(x, y, marker='o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


def main():
    set_random_seeds()
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hg = Hypergraph.generate_test_hypergraph()
    sir = HypergraphSIR(hg, device=device)
    results = defaultdict(list)

    # ===== 非线性实验 =====
    if getattr(cfg, 'run_nonlinear_experiment', False):
        run_nonlinear_experiment(cfg, hg, device)

    # ===== 主实验 =====
    for dataset_name in cfg.datasets:
        print(f"\nProcessing {dataset_name}")

        # 传播曲线可视化
        if getattr(cfg, 'visualize_curves', False):
            curves = []
            for _ in range(5):
                res = sir.simulate(
                    initial_infected=np.random.choice(hg.num_vertices, 5),
                    lambdas=0.1,
                    thetas=1.5,
                    track_curve=True
                )
                curves.append(res['curve'])
            plt = sir.plot_curves(curves, dataset_name)
            plt.savefig(f"{dataset_name}_curves.png")
            plt.close()

        # 参数扫描实验
        print("Running parameter scan...")
        drugcn_model = DRUGCN().to(device)

        for lambdas in tqdm(cfg.lambdas, desc="λ"):
            for theta in cfg.thetas:
                if lambdas >= 1 / (sir.M_max ** theta):
                    continue

                for algo in cfg.algorithms:
                    # 获取节点得分
                    if algo == 'DR-UGCN':
                        features, _, adj = prepare_data(hg, lambdas, theta)
                        adj = normalize_adjacency(adj).to(device)
                        with torch.no_grad():
                            scores = drugcn_model(features, adj).cpu().numpy()
                    else:
                        scores = np.array(list(run_baseline_algorithm(hg, algo).values()))

                    # 选择初始感染节点
                    top_k = int(hg.num_vertices * getattr(cfg, 'top_k_ratio', 0.1))
                    initial_infected = np.argsort(scores)[-top_k:]

                    # 运行多次模拟
                    infection_rates = [
                        sir.simulate(
                            initial_infected=initial_infected,
                            lambdas=lambdas,
                            thetas=theta
                        )['infection_rate']
                        for _ in range(getattr(cfg, 'num_simulations', 10))
                    ]
                    results[(algo, lambdas, theta)].extend(infection_rates)

        # 模型训练
        train_and_save_model(cfg, hg, device, dataset_name)

    print("\nExperiment completed!")
    cleanup_cuda()


if __name__ == "__main__":
    main()