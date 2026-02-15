from svm_framework import PenaltyDualSVM, SVMConfig, ProjectionDualSVM
from run_experiments import prepare_breast_cancer_data


# Load data
X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

# Test fine grid
print("\n" + "=" * 80)
print("Test fine grid")
print("=" * 80)

penalty_weights = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.5,
    2.0,
    3.0,
    5.0,
]

best_f1 = 0
best_weight = 0

for w in penalty_weights:
    print(f"Testing penalty weight: {w}")

    config = SVMConfig(C=0.1, learning_rate=1e-5, max_iter=1000, tol=1e-6)
    model = PenaltyDualSVM(config, use_nesterov=True, momentum=0.9, penalty_weight=w)

    model.fit(X_train, y_train)
    acc, f1 = model.evaluate(X_test, y_test)

    print(f"  F1={f1:.4f}, Acc={acc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_weight = w

print(f"\n🏆 Optimal penalty weight: {best_weight}")
print(f"   Best F1 score: {best_f1:.4f}")

# sklearn comparison
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

print("\n" + "=" * 80)
print("sklearn comparison")
print("=" * 80)
# sklearn
sklearn_model = LinearSVC(C=0.1, max_iter=10000, tol=1e-6, random_state=42)
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_pred)
sklearn_f1 = f1_score(y_test, sklearn_pred)

# Your method
config = SVMConfig(C=0.1, learning_rate=1e-5, max_iter=1000, tol=1e-6)
your_model = PenaltyDualSVM(config, penalty_weight=0.5, use_nesterov=True, momentum=0.9)
your_model.fit(X_train, y_train)
your_acc, your_f1 = your_model.evaluate(X_test, y_test)

print(f"Your implementation: F1={your_f1:.4f}")
print(f"sklearn LinearSVC:   F1={sklearn_f1:.4f}")
print(f"Difference:          {abs(your_f1 - sklearn_f1):.4f}")

if abs(your_f1 - sklearn_f1) < 0.01:
    print("✓ EXCELLENT: Matches sklearn!")


# Testing other datasets
print("\n" + "=" * 80)
print("Testing other datasets")
print("=" * 80)
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def test_dataset(X, y, name):
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    print(f"\n{name} Dataset (n={X_train.shape[0]}, d={X_train.shape[1]})")

    # Test penalty weights
    for w in [0.5, 1.0, 5.0, 10.0]:
        config = SVMConfig(C=0.1, learning_rate=1e-4, max_iter=1000)
        model = PenaltyDualSVM(config, penalty_weight=w, use_nesterov=False)

        model.fit(X_train, y_train)
        acc, f1 = model.evaluate(X_test, y_test)
        print(f"  w={w:4.1f}: F1={f1:.4f}, Acc={acc:.4f}")


# Test on wine (binary)
X, y_multi = load_wine(return_X_y=True)
y = (y_multi == 0).astype(int)
test_dataset(X, y, "Wine")

# Test on iris (binary)
X, y_multi = load_iris(return_X_y=True)
y = (y_multi == 0).astype(int)
test_dataset(X, y, "Iris")


#  Compute Lipschitz Constant
print("\n" + "=" * 80)
print(" Compute Lipschitz Constant ")
print("=" * 80)
import numpy as np

# Add bias
X_with_bias = np.c_[X_train, np.ones(X_train.shape[0])]

# Compute kernel
K = X_with_bias @ X_with_bias.T

# Get eigenvalues
eigenvalues = np.linalg.eigvalsh(K)
L = eigenvalues[-1]

print(f"Lipschitz constant L: {L:.2f}")
print(f"Theoretical safe lr: {1 / L:.2e}")
print(f"Your best lr: 1e-5")
print(f"Ratio: {1e-5 / (1 / L):.2f}×")

# Plot eigenvalue spectrum
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0].plot(eigenvalues, "o-", markersize=3)
# axes[0].axhline(y=1 / 1e-5, color="r", linestyle="--", label="1/lr")
# axes[0].set_title("Full Eigenvalue Spectrum")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

# axes[1].semilogy(eigenvalues[-50:], "o-", markersize=5)
# axes[1].axhline(y=1 / 1e-5, color="r", linestyle="--", label="1/lr")
# axes[1].set_title("Largest Eigenvalues")
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# plt.savefig("lipschitz_analysis.png", dpi=150)
# print("✓ Saved: lipschitz_analysis.png")
# plt.show()


# Tolerance study
print("\n" + "=" * 80)
print(" Tolerance study")
print("=" * 80)
tolerances = [1e-6, 1e-7, 1e-8]

for tol in tolerances:
    print(f"\nTolerance: {tol:.0e}")

    # Penalty
    config = SVMConfig(C=0.1, learning_rate=1e-5, max_iter=3000, tol=tol)
    pen_model = PenaltyDualSVM(config, penalty_weight=0.7, use_nesterov=False)
    pen_model.fit(X_train, y_train)
    _, pen_f1 = pen_model.evaluate(X_test, y_test)

    # Projection
    proj_model = ProjectionDualSVM(config, use_nesterov=False)
    proj_model.fit(X_train, y_train)
    _, proj_f1 = proj_model.evaluate(X_test, y_test)

    print(f"  Penalty:    {pen_f1:.4f}")
    print(f"  Projection: {proj_f1:.4f}")
    print(f"  Gap:        {pen_f1 - proj_f1:.4f}")
