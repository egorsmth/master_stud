"""
Example: Using the refactored framework with breast cancer data
This replaces your original experiment code with cleaner structure
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from svm_framework import ProjectionDualSVM, SVMConfig, PrimalSVM, PenaltyDualSVM
from experiment_utils import ExperimentRunner, HyperparameterSearch


def prepare_breast_cancer_data():
    """Load and preprocess breast cancer dataset"""
    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # Apply your preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Drop highly correlated features
    X_scaled_drop = np.delete(X_scaled, [2, 3, 12, 13, 22, 23], axis=1)

    # PCA
    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X_scaled_drop)

    print(f"Original features: {X.shape[1]}")
    print(f"After dropping: {X_scaled_drop.shape[1]}")
    print(f"After PCA: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Train/test split (80/20)
    split_idx = int(0.8 * len(X_pca))
    X_train = X_pca[split_idx:]
    y_train = y[split_idx:]
    X_test = X_pca[:split_idx]
    y_test = y[:split_idx]

    return X_train, y_train, X_test, y_test


def experiment_1_basic_comparison():
    """
    Experiment 1: Basic comparison of all methods with same hyperparameters
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Basic Method Comparison (Breast Cancer Dataset)")
    print("=" * 80)

    # Load data
    X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

    # Setup experiment
    runner = ExperimentRunner(X_train, y_train, X_test, y_test)

    # Configure all methods with SAME hyperparameters for fair comparison
    C = 0.1
    lr = 0.001
    max_iter = 2000

    config = SVMConfig(
        C=C, learning_rate=lr, max_iter=max_iter, tol=1e-6, verbose=False
    )

    # Add all methods
    runner.add_method("1. Primal GD", PrimalSVM(config))

    runner.add_method(
        "2. Dual Projection (Standard GD)",
        ProjectionDualSVM(config, use_nesterov=False),
    )

    runner.add_method(
        "3. Dual Projection (Nesterov)",
        ProjectionDualSVM(config, use_nesterov=True, momentum=0.9),
    )

    runner.add_method(
        "4. Dual Penalty (Standard GD)",
        PenaltyDualSVM(config, use_nesterov=False, penalty_weight=10),
    )

    runner.add_method(
        "5. Dual Penalty (Nesterov)",
        PenaltyDualSVM(config, use_nesterov=True, momentum=0.9, penalty_weight=10),
    )

    # Run all methods
    runner.run_all(verbose=True)

    # Show comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(runner.create_comparison_table().to_string(index=False))

    # Plot convergence
    fig = runner.plot_convergence()
    # plt.savefig('/mnt/user-data/outputs/exp1_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot efficiency
    fig = runner.plot_efficiency()
    # plt.savefig('/mnt/user-data/outputs/exp1_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()

    return runner


def experiment_2_hyperparameter_search():
    """
    Experiment 2: Find optimal hyperparameters for each method
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Hyperparameter Search")
    print("=" * 80)

    # Load data
    X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

    # Run grid search
    searcher = HyperparameterSearch(X_train, y_train, X_test, y_test)

    best_params, best_score = searcher.search_projection_dual(
        C_values=[0.001, 0.01, 0.1, 1.0],
        lr_values=[1e-5, 1e-4, 1e-3, 1e-2],
        momentum_values=[0.0, 0.9],
        verbose=True,
    )

    # Visualize results
    fig = searcher.plot_search_results()
    # plt.savefig('/mnt/user-data/outputs/exp2_hyperparam_search.png', dpi=150, bbox_inches='tight')
    plt.show()

    return best_params, best_score


def experiment_3_learning_rate_study():
    """
    Experiment 3: Detailed study of learning rate effects
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Learning Rate Sensitivity Analysis")
    print("=" * 80)

    X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    results = {"Nesterov": [], "Standard GD": []}

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr:.1e}")

        # Nesterov
        try:
            config = SVMConfig(C=0.1, learning_rate=lr, max_iter=1000, tol=1e-6)
            model_nesterov = ProjectionDualSVM(config, use_nesterov=True, momentum=0.9)
            result_nesterov = model_nesterov.fit(X_train, y_train)
            _, test_f1_nesterov = model_nesterov.evaluate(X_test, y_test)

            results["Nesterov"].append(
                {
                    "lr": lr,
                    "f1": test_f1_nesterov,
                    "iterations": result_nesterov.iterations,
                    "converged": result_nesterov.converged,
                }
            )
        except ValueError:
            results["Nesterov"].append(
                {"lr": lr, "f1": 0.0, "iterations": 1000, "converged": False}
            )

        # Standard GD
        try:
            model_std = ProjectionDualSVM(config, use_nesterov=False)
            result_std = model_std.fit(X_train, y_train)
            _, test_f1_std = model_std.evaluate(X_test, y_test)

            results["Standard GD"].append(
                {
                    "lr": lr,
                    "f1": test_f1_std,
                    "iterations": result_std.iterations,
                    "converged": result_std.converged,
                }
            )
        except ValueError:
            results["Standard GD"].append(
                {"lr": lr, "f1": 0, "iterations": 1000, "converged": False}
            )

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 vs learning rate
    axes[0].plot(
        [r["lr"] for r in results["Nesterov"]],
        [r["f1"] for r in results["Nesterov"]],
        "o-",
        label="Nesterov",
        linewidth=2,
        markersize=8,
    )
    axes[0].plot(
        [r["lr"] for r in results["Standard GD"]],
        [r["f1"] for r in results["Standard GD"]],
        "s-",
        label="Standard GD",
        linewidth=2,
        markersize=8,
    )
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Test F1 Score")
    axes[0].set_xscale("log")
    axes[0].set_title("F1 Score vs Learning Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Iterations vs learning rate
    axes[1].plot(
        [r["lr"] for r in results["Nesterov"]],
        [r["iterations"] for r in results["Nesterov"]],
        "o-",
        label="Nesterov",
        linewidth=2,
        markersize=8,
    )
    axes[1].plot(
        [r["lr"] for r in results["Standard GD"]],
        [r["iterations"] for r in results["Standard GD"]],
        "s-",
        label="Standard GD",
        linewidth=2,
        markersize=8,
    )
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Iterations to Convergence")
    axes[1].set_xscale("log")
    axes[1].set_title("Convergence Speed vs Learning Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/exp3_learning_rate_study.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def experiment_4_momentum_study():
    """
    Experiment 4: Study effect of momentum parameter
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Momentum Parameter Study")
    print("=" * 80)

    X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

    momentum_values = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    results = []

    for momentum in momentum_values:
        print(f"\nTesting momentum: {momentum:.2f}")

        config = SVMConfig(C=0.1, learning_rate=1e-4, max_iter=1000, tol=1e-6)
        model = ProjectionDualSVM(
            config, use_nesterov=(momentum > 0), momentum=momentum
        )
        result = model.fit(X_train, y_train)
        _, test_f1 = model.evaluate(X_test, y_test)

        results.append(
            {
                "momentum": momentum,
                "f1": test_f1,
                "iterations": result.iterations,
                "wall_time": result.wall_time,
                "final_grad_norm": result.gradient_norm_history[-1]
                if result.gradient_norm_history
                else None,
            }
        )

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # F1 vs momentum
    axes[0].plot(
        [r["momentum"] for r in results],
        [r["f1"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    axes[0].set_xlabel("Momentum")
    axes[0].set_ylabel("Test F1 Score")
    axes[0].set_title("F1 Score vs Momentum")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        x=0.9, color="red", linestyle="--", alpha=0.5, label="Standard choice"
    )
    axes[0].legend()

    # Iterations vs momentum
    axes[1].plot(
        [r["momentum"] for r in results],
        [r["iterations"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="coral",
    )
    axes[1].set_xlabel("Momentum")
    axes[1].set_ylabel("Iterations to Convergence")
    axes[1].set_title("Convergence Speed vs Momentum")
    axes[1].grid(True, alpha=0.3)

    # Wall time vs momentum
    axes[2].plot(
        [r["momentum"] for r in results],
        [r["wall_time"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="seagreen",
    )
    axes[2].set_xlabel("Momentum")
    axes[2].set_ylabel("Wall Time (seconds)")
    axes[2].set_title("Computation Time vs Momentum")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/exp4_momentum_study.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def experiment_5_projection_vs_penalty():
    """
    Experiment 5: Detailed comparison of projection vs penalty methods
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Projection vs Penalty Method Comparison")
    print("=" * 80)

    X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

    runner = ExperimentRunner(X_train, y_train, X_test, y_test)

    # Test different penalty weights
    penalty_weights = [0.5, 1, 5, 10, 25, 50, 100]

    # Add projection method (baseline)
    config = SVMConfig(C=0.1, learning_rate=1e-4, max_iter=2000, tol=1e-6)
    runner.add_method(
        "Projection (Nesterov)",
        ProjectionDualSVM(config, use_nesterov=True, momentum=0.9),
    )

    # Add penalty methods with different weights
    for pw in penalty_weights:
        runner.add_method(
            f"Penalty (weight={pw})",
            PenaltyDualSVM(config, use_nesterov=True, momentum=0.9, penalty_weight=pw),
        )

    runner.run_all(verbose=False)

    # Show comparison
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(runner.create_comparison_table().to_string(index=False))

    # Plot constraint violations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, result in runner.results.items():
        if result.constraint_violation_history:
            axes[0].semilogy(
                result.constraint_violation_history, label=name, linewidth=2
            )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("|∑αᵢyᵢ| (log scale)")
    axes[0].set_title("Constraint Violation Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final constraint violation vs test F1
    methods = []
    final_viols = []
    test_f1s = []

    for name, result in runner.results.items():
        methods.append(name)
        final_viols.append(
            result.constraint_violation_history[-1]
            if result.constraint_violation_history
            else 0
        )
        test_f1s.append(result.test_f1)

    axes[1].scatter(final_viols, test_f1s, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        axes[1].annotate(method, (final_viols[i], test_f1s[i]), fontsize=8, ha="right")
    axes[1].set_xlabel("Final Constraint Violation |∑αᵢyᵢ|")
    axes[1].set_ylabel("Test F1 Score")
    axes[1].set_title("Constraint Satisfaction vs Performance")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/exp5_projection_vs_penalty.png', dpi=150, bbox_inches='tight')
    plt.show()

    return runner


def run_all_experiments():
    """Run all experiments in sequence"""
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE EXPERIMENTAL SUITE")
    print("=" * 80)

    # Experiment 1: Basic comparison
    runner1 = experiment_1_basic_comparison()

    # Experiment 2: Hyperparameter search
    best_params, best_score = experiment_2_hyperparameter_search()

    # Experiment 3: Learning rate study
    lr_results = experiment_3_learning_rate_study()

    # Experiment 4: Momentum study
    momentum_results = experiment_4_momentum_study()

    # Experiment 5: Projection vs Penalty
    runner5 = experiment_5_projection_vs_penalty()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"1. Best hyperparameters: {best_params}")
    print(f"2. Best test F1 score: {best_score:.4f}")
    print("3. Check output plots for detailed analysis")


if __name__ == "__main__":
    # Run individual experiments or all at once

    # Option 1: Run one experiment
    runner = experiment_1_basic_comparison()

    # Option 2: Run specific experiments
    # experiment_3_learning_rate_study()
    # experiment_4_momentum_study()

    # Option 3: Run everything
    # run_all_experiments()
