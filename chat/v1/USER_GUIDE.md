# SVM Experimental Framework - User Guide

## Overview

This refactored framework provides a clean, extensible structure for comparing SVM optimization methods. It fixes bugs in the original code and makes experiments much easier to run and analyze.

## Quick Start

```python
from svm_framework import *
from experiment_utils import ExperimentRunner
from run_experiments import prepare_breast_cancer_data

# Load data
X_train, y_train, X_test, y_test = prepare_breast_cancer_data()

# Create experiment
runner = ExperimentRunner(X_train, y_train, X_test, y_test)

# Add methods to compare
config = SVMConfig(C=0.1, learning_rate=0.001, max_iter=1000, tol=1e-6)

runner.add_method("Primal", PrimalSVM(config))
runner.add_method("Dual Nesterov", 
                 ProjectionDualSVM(config, use_nesterov=True, momentum=0.9))

# Run and compare
runner.run_all()
print(runner.create_comparison_table())
runner.plot_convergence()
```

## Architecture

### Core Components

**1. svm_framework.py**
- `SVMConfig`: Dataclass for hyperparameters
- `OptimizationResult`: Stores all metrics from a run
- `BaseSVM`: Abstract base class
- `PrimalSVM`: Primal gradient descent
- `DualSVM`: Base class for dual methods
- `ProjectionDualSVM`: Dual with projection (supports Nesterov)
- `PenaltyDualSVM`: Dual with penalty method (supports Nesterov)

**2. experiment_utils.py**
- `ExperimentRunner`: Run and compare multiple methods
- `HyperparameterSearch`: Grid search for optimal parameters
- Comparison and visualization utilities

**3. run_experiments.py**
- Pre-configured experiments on breast cancer data
- 5 different experimental setups
- Ready-to-run examples

## Key Improvements Over Original Code

### 1. Bug Fixes

✅ **Fixed dimension mismatch**: All methods now use consistent data with bias
```python
# Old code (WRONG):
primal(y_train, X_train_with_b, ...)  # Has bias
dual_svm.fit(X_train, y_train)        # No bias!

# New code (CORRECT):
X_with_bias = np.c_[X, np.ones(X.shape[0])]  # Always add bias
```

✅ **Consistent label conversion**: All methods use {-1, +1}
```python
y = np.where(y == 0, -1, y)
```

✅ **Proper result tracking**: All metrics stored in `OptimizationResult`

### 2. Better Organization

- Clean class hierarchy
- Separation of concerns
- Type hints throughout
- Comprehensive docstrings

### 3. Experiment Management

- Easy to add new methods
- Automatic comparison tables
- Built-in visualization
- Hyperparameter search

### 4. Metrics

Now tracking:
- Convergence: objective, gradient norm, constraint violation
- Performance: train/test accuracy, F1 score
- Efficiency: iterations, wall time
- Quality: duality gap, KKT violation, support vectors

## Available Experiments

### Experiment 1: Basic Comparison
Compare all methods with same hyperparameters
```python
experiment_1_basic_comparison()
```

### Experiment 2: Hyperparameter Search
Find optimal C, learning rate, momentum
```python
experiment_2_hyperparameter_search()
```

### Experiment 3: Learning Rate Study
Detailed analysis of learning rate sensitivity
```python
experiment_3_learning_rate_study()
```

### Experiment 4: Momentum Study
How does momentum affect Nesterov acceleration?
```python
experiment_4_momentum_study()
```

### Experiment 5: Projection vs Penalty
Compare constraint handling methods
```python
experiment_5_projection_vs_penalty()
```

## Adding New Methods

Easy to extend! Just inherit from `BaseSVM` or `DualSVM`:

```python
class MyNewMethod(DualSVM):
    def fit(self, X, y):
        # Your optimization code here
        ...
        return OptimizationResult(...)
    
    def predict(self, X):
        return np.sign(X @ self.weights)
```

## Adding New Datasets

Simple pattern:

```python
def prepare_my_data():
    X, y = load_my_dataset()
    # Preprocessing...
    return X_train, y_train, X_test, y_test

# Use it
X_train, y_train, X_test, y_test = prepare_my_data()
runner = ExperimentRunner(X_train, y_train, X_test, y_test)
# ... rest of experiment
```

## Understanding the Results

### Convergence Plots
- **Objective history**: Should be monotonically improving
- **Gradient norm**: Should decrease to near zero
- **Constraint violation**: Should be < 1e-6 for good solutions

### Performance Metrics
- **Test accuracy/F1**: Main evaluation metric
- **Number of support vectors**: Fewer is often better (simpler model)
- **KKT violation**: Measures solution optimality (lower is better)

### Red Flags
- ⚠️ Duality gap > 0.1: Poor convergence
- ⚠️ KKT violation > 1.0: Suboptimal solution
- ⚠️ Test accuracy << train accuracy: Overfitting

## Common Issues and Solutions

### Issue 1: Methods don't converge
**Symptoms**: gradient norm stays high, max iterations reached

**Solutions**:
1. Increase `max_iter`
2. Decrease `learning_rate`
3. Check data scaling (features should be standardized)

### Issue 2: Nesterov worse than standard GD
**Possible causes**:
1. Learning rate too small (momentum benefits disappear)
2. Projection disrupts acceleration
3. Problem poorly conditioned

**Try**:
1. Increase learning rate
2. Adjust momentum (try 0.7, 0.8, 0.9, 0.95)
3. Use penalty method instead of projection

### Issue 3: Large duality gap
**Means**: Solutions haven't converged

**Solutions**:
1. Decrease tolerance
2. Increase max_iterations  
3. Tune learning rate
4. Check for bugs in gradient computation

## Recommended Workflow

### Phase 1: Verify Correctness (Week 1)
1. Test on synthetic linearly separable data
2. Check duality gap < 0.01
3. Verify all methods give same accuracy
4. Confirm KKT violations are small

### Phase 2: Method Comparison (Week 2)
1. Run Experiment 1 with multiple C values
2. Run Experiment 3 (learning rate study)
3. Run Experiment 4 (momentum study)
4. Identify best configuration for each method

### Phase 3: Robustness Testing (Week 3)
1. Test on 3-5 different datasets
2. Vary train/test splits (5-fold CV)
3. Run with different random seeds
4. Check consistency of findings

### Phase 4: Analysis (Week 4)
1. Create comprehensive comparison tables
2. Statistical significance testing
3. Understand when each method works best
4. Write up findings

## Expected Results (Based on Theory)

### When Nesterov Should Win:
- Well-conditioned problems (κ < 100)
- Smooth objectives
- When learning rate is well-tuned
- Long optimization horizons

### When Projection Beats Penalty:
- When exact constraint satisfaction is critical
- For poorly scaled problems
- When penalty weight is hard to tune

### When Primal Beats Dual:
- Very high-dimensional problems
- When you don't need support vector interpretation
- Simpler implementation might have fewer bugs

## Next Steps

1. Run `experiment_1_basic_comparison()` to see if framework works
2. If results look good, run hyperparameter search
3. Try on your own datasets
4. Extend with new methods (e.g., coordinate descent, SMO)
5. Write up your findings!

## Troubleshooting

If you encounter errors:

1. Check data shapes: `print(X_train.shape, y_train.shape)`
2. Check label values: `print(np.unique(y_train))`
3. Verify convergence: `print(result.gradient_norm_history[-10:])`
4. Enable verbose mode: `config = SVMConfig(..., verbose=True)`

## References for Further Reading

- Boyd & Vandenberghe: Convex Optimization (Chapters 9-10)
- Bottou: Large-Scale Machine Learning with Stochastic Gradient Descent
- Hsieh et al.: A Dual Coordinate Descent Method for Large-scale Linear SVM
- Nesterov: A method for solving the convex programming problem

## Contact

For questions or issues with the framework, check:
- Function docstrings for detailed parameter explanations
- OptimizationResult dataclass for available metrics
- experiment_utils.py for visualization options

Good luck with your experiments! 🚀
