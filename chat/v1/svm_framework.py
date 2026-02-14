"""
Refactored SVM Experimental Framework
Clean architecture for comparing optimization methods
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class SVMConfig:
    """Configuration for SVM optimization"""

    C: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4
    learning_rate: float = 0.01
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Store all metrics from optimization run"""

    # Solution
    weights: np.ndarray
    alpha: Optional[np.ndarray] = None

    # Convergence metrics
    objective_history: List[float] = None
    gradient_norm_history: List[float] = None
    constraint_violation_history: List[float] = None

    # Performance metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_f1: float = 0.0
    test_f1: float = 0.0

    # Optimization metrics
    iterations: int = 0
    wall_time: float = 0.0
    converged: bool = False

    # Dual-specific
    duality_gap: Optional[float] = None
    kkt_violation: Optional[float] = None
    num_support_vectors: int = 0


class BaseSVM(ABC):
    """Abstract base class for all SVM implementations"""

    def __init__(self, config: SVMConfig):
        self.config = config
        self.result: Optional[OptimizationResult] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Train the SVM"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute accuracy and F1 score"""
        predictions = self.predict(X)
        predictions_01 = (predictions + 1) // 2
        acc = accuracy_score(y, predictions_01)
        # print(y.shape, predictions.shape)
        # print(y[:10])
        # print((predictions[:10] + 1) // 2 )
        f1 = f1_score(y, predictions_01)
        return acc, f1

    def _check_convergence(
        self, gradient_norm: float, obj_change: float, iteration: int
    ) -> bool:
        """Check multiple convergence criteria"""
        if gradient_norm < self.config.tol:
            if self.config.verbose:
                print(
                    f"Converged: gradient norm {gradient_norm:.2e} < {self.config.tol}"
                )
            return True

        if iteration > 10 and obj_change < self.config.tol:
            if self.config.verbose:
                print(
                    f"Converged: objective change {obj_change:.2e} < {self.config.tol}"
                )
            return True

        if iteration >= self.config.max_iter:
            if self.config.verbose:
                print(f"Max iterations {self.config.max_iter} reached")
            return True

        return False


class PrimalSVM(BaseSVM):
    """Primal SVM using gradient descent on hinge loss"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """
        Minimize: L(w) = 1/2||w||² + C∑max(0, 1 - yᵢ(wᵀxᵢ))
        """
        start_time = time.time()

        # Add bias column
        X_with_bias = np.c_[X, np.ones(X.shape[0])]

        # Convert labels to {-1, +1}
        y = np.where(y == 0, -1, y)

        # Initialize
        w = np.zeros(X_with_bias.shape[1])

        # History tracking
        obj_history = []
        grad_norm_history = []

        for iteration in range(self.config.max_iter):
            # Compute gradient
            margins = y * (X_with_bias @ w)
            grad = w.copy()

            # Add gradient from violated constraints
            violated = margins < 1
            if np.any(violated):
                grad -= self.config.C * (X_with_bias[violated].T @ y[violated])

            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)

            # Update weights
            w = w - self.config.learning_rate * grad

            # Compute objective
            hinge_loss = np.sum(np.maximum(0, 1 - margins))
            obj_val = 0.5 * np.linalg.norm(w) ** 2 + self.config.C * hinge_loss
            obj_history.append(obj_val)

            # Check convergence
            obj_change = (
                abs(obj_history[-1] - obj_history[-2])
                if len(obj_history) > 1
                else float("inf")
            )
            if self._check_convergence(grad_norm, obj_change, iteration):
                converged = True
                break
        else:
            converged = False

        wall_time = time.time() - start_time

        # Store results
        self.weights = w
        self.X_train_bias = X_with_bias
        self.y_train = y

        result = OptimizationResult(
            weights=w,
            objective_history=obj_history,
            gradient_norm_history=grad_norm_history,
            iterations=iteration + 1,
            wall_time=wall_time,
            converged=converged,
        )

        self.result = result
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        # print("Primal.predict")
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        return np.sign(X_with_bias @ self.weights)


class DualSVM(BaseSVM):
    """Base class for dual SVM implementations"""

    def compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute linear kernel matrix K_ij = xᵢ·xⱼ"""
        return X @ X.T

    def compute_dual_objective(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Dual objective: L(α) = ∑αᵢ - 1/2 αᵀ(K ⊙ yyᵀ)α
        (Maximization problem)
        """
        return np.sum(alpha) - 0.5 * alpha.T @ ((K * np.outer(y, y)) @ alpha)

    def compute_dual_gradient(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Gradient: ∇L(α) = 1 - (K ⊙ yyᵀ)α
        """
        Y = np.diag(y)
        return np.ones(len(y)) - Y @ K @ Y @ alpha

    def reconstruct_primal_weights(
        self, alpha: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Reconstruct w from dual solution: w = ∑αᵢyᵢxᵢ"""
        return np.sum(alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

    def compute_kkt_violation(
        self, alpha: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Compute total KKT violation:
        1. Stationarity: w = ∑αᵢyᵢxᵢ (always satisfied by construction)
        2. Primal feasibility: always satisfied
        3. Dual feasibility: 0 ≤ αᵢ ≤ C, ∑αᵢyᵢ = 0
        4. Complementary slackness: αᵢ[yᵢ(w·xᵢ) - 1] = 0
        """
        w = self.reconstruct_primal_weights(alpha, X, y)
        margins = y * (X @ w)

        # Dual feasibility violation
        dual_feas = max(0, -alpha.min()) + max(0, alpha.max() - self.config.C)
        constraint_viol = abs(np.dot(alpha, y))

        # Complementary slackness
        comp_slack = 0.0
        for i in range(len(alpha)):
            if alpha[i] > 1e-5 and alpha[i] < self.config.C - 1e-5:
                # Should have margin = 1
                comp_slack += abs(margins[i] - 1)
            elif alpha[i] >= self.config.C - 1e-5:
                # Should have margin ≤ 1
                if margins[i] > 1 + 1e-3:
                    comp_slack += margins[i] - 1

        return dual_feas + constraint_viol + comp_slack

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using dual weights"""
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        # print("Dual.predict")
        # print(X.shape)
        # print(self.weights.shape)
        return np.sign(X_with_bias @ self.weights)


class ProjectionDualSVM(DualSVM):
    """Dual SVM with projection onto constraints"""

    def __init__(
        self, config: SVMConfig, use_nesterov: bool = False, momentum: float = 0.9
    ):
        super().__init__(config)
        self.use_nesterov = use_nesterov
        self.momentum = momentum

    def project_to_constraints(self, alpha: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Project onto: {α | 0 ≤ αᵢ ≤ C, ∑αᵢyᵢ = 0}

        1. Clip to box constraints
        2. Project onto hyperplane
        """
        # Box projection
        alpha = np.clip(alpha, 0, self.config.C)

        # Hyperplane projection: α - (yᵀα / yᵀy) · y
        scalar = np.dot(y, alpha) / np.dot(y, y)
        alpha = alpha - scalar * y

        # Clip again to ensure box constraints
        alpha = np.clip(alpha, 0, self.config.C)

        return alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Train using projected gradient (with optional Nesterov)"""
        start_time = time.time()

        # Add bias and convert labels
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        y = np.where(y == 0, -1, y)

        # Compute kernel
        K = self.compute_kernel_matrix(X_with_bias)

        # Initialize
        alpha = np.zeros(len(y))
        velocity = np.zeros(len(y)) if self.use_nesterov else None

        # History
        obj_history = []
        grad_norm_history = []
        constraint_viol_history = []

        for iteration in range(self.config.max_iter):
            # Nesterov lookahead
            if self.use_nesterov:
                alpha_grad = alpha + self.momentum * velocity
            else:
                alpha_grad = alpha

            # Compute gradient
            grad = self.compute_dual_gradient(alpha_grad, K, y)
            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)

            # Update
            if self.use_nesterov:
                velocity = self.momentum * velocity + self.config.learning_rate * grad
                alpha_new = alpha + velocity
            else:
                alpha_new = alpha + self.config.learning_rate * grad

            # Project onto constraints
            alpha_new = self.project_to_constraints(alpha_new, y)

            # Compute objective and constraint violation
            obj_val = self.compute_dual_objective(alpha_new, K, y)
            obj_history.append(obj_val)

            constraint_viol = abs(np.dot(alpha_new, y))
            constraint_viol_history.append(constraint_viol)

            # Check convergence
            obj_change = (
                abs(obj_history[-1] - obj_history[-2])
                if len(obj_history) > 1
                else float("inf")
            )
            if self._check_convergence(grad_norm, obj_change, iteration):
                converged = True
                break

            alpha = alpha_new
        else:
            converged = False

        wall_time = time.time() - start_time

        # Reconstruct weights and identify support vectors
        self.weights = self.reconstruct_primal_weights(alpha, X_with_bias, y)
        self.alpha = alpha

        sv_mask = alpha > 1e-5
        num_sv = np.sum(sv_mask)

        # Compute KKT violation
        kkt_viol = self.compute_kkt_violation(alpha, X_with_bias, y)

        result = OptimizationResult(
            weights=self.weights,
            alpha=alpha,
            objective_history=obj_history,
            gradient_norm_history=grad_norm_history,
            constraint_violation_history=constraint_viol_history,
            iterations=iteration + 1,
            wall_time=wall_time,
            converged=converged,
            kkt_violation=kkt_viol,
            num_support_vectors=num_sv,
        )

        self.result = result
        self.X_train = X_with_bias
        self.y_train = y

        return result


class PenaltyDualSVM(DualSVM):
    """Dual SVM with penalty method for constraints"""

    def __init__(
        self,
        config: SVMConfig,
        use_nesterov: bool = False,
        momentum: float = 0.9,
        penalty_weight: float = 100.0,
    ):
        super().__init__(config)
        self.use_nesterov = use_nesterov
        self.momentum = momentum
        self.penalty_weight = penalty_weight

    def compute_penalized_objective(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> float:
        """Add penalty for constraint violation"""
        base_obj = self.compute_dual_objective(alpha, K, y)
        penalty = self.penalty_weight * (np.dot(y, alpha) ** 2)
        return base_obj - penalty

    def compute_penalized_gradient(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Gradient with penalty term"""
        base_grad = self.compute_dual_gradient(alpha, K, y)
        penalty_grad = 2 * self.penalty_weight * y * np.dot(y, alpha)
        return base_grad - penalty_grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Train using penalty method (with optional Nesterov)"""
        start_time = time.time()

        # Add bias and convert labels
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        y = np.where(y == 0, -1, y)

        # Compute kernel
        K = self.compute_kernel_matrix(X_with_bias)

        # Initialize
        alpha = np.zeros(len(y))
        velocity = np.zeros(len(y)) if self.use_nesterov else None

        # History
        obj_history = []
        grad_norm_history = []
        constraint_viol_history = []

        for iteration in range(self.config.max_iter):
            # Nesterov lookahead
            if self.use_nesterov:
                alpha_grad = alpha + self.momentum * velocity
            else:
                alpha_grad = alpha

            # Compute gradient with penalty
            grad = self.compute_penalized_gradient(alpha_grad, K, y)
            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)

            # Update
            if self.use_nesterov:
                velocity = self.momentum * velocity + self.config.learning_rate * grad
                alpha_new = alpha + velocity
            else:
                alpha_new = alpha + self.config.learning_rate * grad

            # Only enforce box constraints (penalty handles equality constraint)
            alpha_new = np.clip(alpha_new, 0, self.config.C)

            # Compute objectives
            obj_val = self.compute_penalized_objective(alpha_new, K, y)
            obj_history.append(obj_val)

            constraint_viol = abs(np.dot(alpha_new, y))
            constraint_viol_history.append(constraint_viol)

            # Check convergence
            obj_change = (
                abs(obj_history[-1] - obj_history[-2])
                if len(obj_history) > 1
                else float("inf")
            )
            if self._check_convergence(grad_norm, obj_change, iteration):
                converged = True
                break

            alpha = alpha_new
        else:
            converged = False

        wall_time = time.time() - start_time

        # Reconstruct weights and identify support vectors
        self.weights = self.reconstruct_primal_weights(alpha, X_with_bias, y)
        self.alpha = alpha

        sv_mask = alpha > 1e-5
        num_sv = np.sum(sv_mask)

        # Compute KKT violation
        kkt_viol = self.compute_kkt_violation(alpha, X_with_bias, y)

        result = OptimizationResult(
            weights=self.weights,
            alpha=alpha,
            objective_history=obj_history,
            gradient_norm_history=grad_norm_history,
            constraint_violation_history=constraint_viol_history,
            iterations=iteration + 1,
            wall_time=wall_time,
            converged=converged,
            kkt_violation=kkt_viol,
            num_support_vectors=num_sv,
        )

        self.result = result
        self.X_train = X_with_bias
        self.y_train = y

        return result
