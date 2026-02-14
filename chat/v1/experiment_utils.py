"""
Experimental utilities for SVM comparison studies
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
from svm_framework import *


class ExperimentRunner:
    """Run and compare multiple SVM methods"""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results: Dict[str, OptimizationResult] = {}
        self.models: Dict[str, BaseSVM] = {}

    def add_method(self, name: str, model: BaseSVM):
        """Add a method to compare"""
        self.models[name] = model

    def run_all(self, verbose: bool = True):
        """Train all methods and evaluate"""
        for name, model in self.models.items():
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Training: {name}")
                print("=" * 60)

            # Train
            result = model.fit(self.X_train, self.y_train)

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Evaluating: {name}")
                print("=" * 60)

            # Evaluate
            try:
                train_acc, train_f1 = model.evaluate(self.X_train, self.y_train)
            except ValueError:
                train_acc, train_f1 = 0, 0
            try:
                test_acc, test_f1 = model.evaluate(self.X_test, self.y_test)
            except ValueError:
                test_acc, test_f1 = 0, 0

            # Update result
            result.train_accuracy = train_acc
            result.train_f1 = train_f1
            result.test_accuracy = test_acc
            result.test_f1 = test_f1

            # Compute duality gap if we have primal and dual solutions
            if "primal" in name.lower() and hasattr(model, "result"):
                primal_obj = result.objective_history[-1]
                for dual_name, dual_result in self.results.items():
                    if "dual" in dual_name.lower() and dual_result.objective_history:
                        dual_obj = dual_result.objective_history[-1]
                        result.duality_gap = primal_obj - dual_obj

            self.results[name] = result

            if verbose:
                self._print_result(name, result)

    def _print_result(self, name: str, result: OptimizationResult):
        """Print summary for one method"""
        print(f"\n{name} Results:")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Wall time: {result.wall_time:.3f}s")
        print(f"  Train accuracy: {result.train_accuracy:.4f}")
        print(f"  Test accuracy: {result.test_accuracy:.4f}")
        print(f"  Test F1: {result.test_f1:.4f}")

        if result.num_support_vectors > 0:
            print(f"  Support vectors: {result.num_support_vectors}")

        if result.kkt_violation is not None:
            print(f"  KKT violation: {result.kkt_violation:.2e}")

        if result.constraint_violation_history:
            print(
                f"  Final constraint violation: {result.constraint_violation_history[-1]:.2e}"
            )

        if result.duality_gap is not None:
            print(f"  Duality gap: {result.duality_gap:.4f}")

    def create_comparison_table(self) -> pd.DataFrame:
        """Create pandas DataFrame comparing all methods"""
        data = []
        for name, result in self.results.items():
            row = {
                "Method": name,
                "Test Acc": f"{result.test_accuracy:.4f}",
                "Test F1": f"{result.test_f1:.4f}",
                "Iterations": result.iterations,
                "Time (s)": f"{result.wall_time:.3f}",
                "Converged": "✓" if result.converged else "✗",
                "Num SV": result.num_support_vectors
                if result.num_support_vectors > 0
                else "-",
                "KKT Viol": f"{result.kkt_violation:.2e}"
                if result.kkt_violation
                else "-",
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df

    def plot_convergence(self, figsize=(15, 10)):
        """Plot convergence comparison across all methods"""
        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Objective values
        ax = axes[0, 0]
        for name, result in self.results.items():
            if result.objective_history:
                ax.plot(result.objective_history, label=name, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Objective Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Gradient norms (log scale)
        ax = axes[0, 1]
        for name, result in self.results.items():
            if result.gradient_norm_history:
                ax.semilogy(result.gradient_norm_history, label=name, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("||∇L||")
        ax.set_title("Gradient Norm (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Constraint violations
        ax = axes[1, 0]
        for name, result in self.results.items():
            if result.constraint_violation_history:
                ax.semilogy(
                    result.constraint_violation_history, label=name, linewidth=2
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|∑αᵢyᵢ|")
        ax.set_title("Constraint Violation (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Performance comparison
        ax = axes[1, 1]
        methods = list(self.results.keys())
        test_accs = [self.results[m].test_accuracy for m in methods]
        test_f1s = [self.results[m].test_f1 for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, test_accs, width, label="Accuracy", alpha=0.8)
        ax.bar(x + width / 2, test_f1s, width, label="F1 Score", alpha=0.8)
        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.set_title("Test Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_efficiency(self, figsize=(12, 5)):
        """Plot time vs accuracy trade-offs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Time vs Accuracy
        for name, result in self.results.items():
            ax1.scatter(
                result.wall_time, result.test_accuracy, s=100, label=name, alpha=0.7
            )
        ax1.set_xlabel("Wall Time (seconds)")
        ax1.set_ylabel("Test Accuracy")
        ax1.set_title("Efficiency: Time vs Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Iterations vs Accuracy
        for name, result in self.results.items():
            ax2.scatter(
                result.iterations, result.test_accuracy, s=100, label=name, alpha=0.7
            )
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Test Accuracy")
        ax2.set_title("Efficiency: Iterations vs Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class HyperparameterSearch:
    """Grid search for hyperparameters"""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = []

    def search_projection_dual(
        self,
        C_values=[0.01, 0.1, 1.0, 10.0],
        lr_values=[1e-5, 1e-4, 1e-3, 1e-2],
        momentum_values=[0.0, 0.9],
        verbose=False,
    ):
        """Grid search for projection dual SVM"""
        print(
            f"Running grid search: {len(C_values)} × {len(lr_values)} × {len(momentum_values)} = "
            f"{len(C_values) * len(lr_values) * len(momentum_values)} combinations"
        )

        best_score = 0
        best_params = {}

        for C in C_values:
            for lr in lr_values:
                for momentum in momentum_values:
                    use_nesterov = momentum > 0

                    config = SVMConfig(C=C, learning_rate=lr, max_iter=1000, tol=1e-6)
                    model = ProjectionDualSVM(
                        config, use_nesterov=use_nesterov, momentum=momentum
                    )

                    try:
                        result = model.fit(self.X_train, self.y_train)
                        _, test_f1 = model.evaluate(self.X_test, self.y_test)

                        self.results.append(
                            {
                                "C": C,
                                "learning_rate": lr,
                                "momentum": momentum,
                                "use_nesterov": use_nesterov,
                                "test_f1": test_f1,
                                "test_acc": result.test_accuracy,
                                "iterations": result.iterations,
                                "converged": result.converged,
                            }
                        )

                        if test_f1 > best_score:
                            best_score = test_f1
                            best_params = {
                                "C": C,
                                "learning_rate": lr,
                                "momentum": momentum,
                                "use_nesterov": use_nesterov,
                            }

                        if verbose:
                            print(
                                f"C={C:.3f}, lr={lr:.1e}, momentum={momentum:.2f} → "
                                f"F1={test_f1:.4f}"
                            )

                    except Exception as e:
                        print(f"Failed: C={C}, lr={lr}, momentum={momentum}: {e}")

        print(f"\nBest parameters: {best_params}")
        print(f"Best test F1: {best_score:.4f}")

        return best_params, best_score

    def plot_search_results(self, figsize=(15, 5)):
        """Visualize grid search results"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # F1 vs C (grouped by learning rate)
        for lr in df["learning_rate"].unique():
            subset = df[df["learning_rate"] == lr]
            axes[0].plot(
                subset["C"], subset["test_f1"], marker="o", label=f"lr={lr:.1e}"
            )
        axes[0].set_xlabel("C")
        axes[0].set_ylabel("Test F1")
        axes[0].set_xscale("log")
        axes[0].set_title("Effect of C")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1 vs learning rate (grouped by C)
        for C in df["C"].unique():
            subset = df[df["C"] == C]
            axes[1].plot(
                subset["learning_rate"], subset["test_f1"], marker="o", label=f"C={C}"
            )
        axes[1].set_xlabel("Learning Rate")
        axes[1].set_ylabel("Test F1")
        axes[1].set_xscale("log")
        axes[1].set_title("Effect of Learning Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Heatmap: C vs learning rate (averaged over momentum)
        pivot = df.groupby(["C", "learning_rate"])["test_f1"].mean().unstack()
        im = axes[2].imshow(pivot.values, aspect="auto", cmap="viridis")
        axes[2].set_xticks(range(len(pivot.columns)))
        axes[2].set_yticks(range(len(pivot.index)))
        axes[2].set_xticklabels([f"{x:.1e}" for x in pivot.columns], rotation=45)
        axes[2].set_yticklabels([f"{y}" for y in pivot.index])
        axes[2].set_xlabel("Learning Rate")
        axes[2].set_ylabel("C")
        axes[2].set_title("Test F1 Heatmap")
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        return fig


def compare_on_synthetic_data():
    """Quick test on synthetic linearly separable data"""
    from sklearn.datasets import make_classification

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA TEST (Linearly Separable)")
    print("=" * 60)

    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=2.0,
        random_state=42,
    )

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Setup experiment
    runner = ExperimentRunner(X_train, y_train, X_test, y_test)

    # Add methods
    config = SVMConfig(C=1.0, learning_rate=0.01, max_iter=500, tol=1e-6)

    runner.add_method("Primal GD", PrimalSVM(config))
    runner.add_method(
        "Dual Projection (GD)", ProjectionDualSVM(config, use_nesterov=False)
    )
    runner.add_method(
        "Dual Projection (Nesterov)",
        ProjectionDualSVM(config, use_nesterov=True, momentum=0.9),
    )

    # Run
    runner.run_all(verbose=True)

    # Show comparison
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(runner.create_comparison_table().to_string(index=False))

    # Plot
    runner.plot_convergence()
    plt.suptitle("Synthetic Data: Method Comparison", fontsize=16, y=1.02)
    plt.show()

    return runner


if __name__ == "__main__":
    # Run test on synthetic data
    runner = compare_on_synthetic_data()
