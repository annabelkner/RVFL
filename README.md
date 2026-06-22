# RVFL

This repository contains a custom implementation of the RVFL (Random Vector Functional Link) estimator and a set of experiments comparing its performance against classical MLP (Multi-Layer Perceptron) neural networks in a regression task based on spectral band data.

## Czym jest RVFL?

RVFL is a single-hidden-layer feedforward neural network in which the hidden-layer weights and biases are randomly generated from a probability distribution and remain fixed throughout training.
Only the output-layer weights are learned, which typically reduces training to solving a convex optimization problem (e.g., ridge regression). This architecture completely eliminates the need for the computationally expensive backpropagation algorithm.

---

## Implementation

The RVFL model was implemented from scratch in Python as a class inheriting from BaseEstimator and RegressorMixin from the scikit-learn library.

This ensures full compatibility with the ecosystem, including objects such as Pipeline and cross_validate.

The implementation supports:

- Configurable number of neurons in the random hidden layer,
- Multiple nonlinear activation functions:
  - ReLU,
  - Sigmoid,
  - Tanh,
- Weight initialization from either uniform or normal distributions with configurable scaling (scale, bias_scale),
- Optional direct input-to-output connections (direct_linkage),
- Output weight optimization via ridge regression (Ridge) with regularization parameter alpha,
- Strict randomness control through random_state for full reproducibility.

---

## ata and Preprocessing

The model was trained using the spectral_train.csv dataset.

To ensure an unbiased evaluation and prevent data leakage, the following procedure was applied:

### Held-Out Test Set

Before any model development or hyperparameter tuning, 20% of the data (1000 observations) were separated as a held-out test set.
The remaining 80% (4000 observations) formed the working dataset used for training and validation.

### Reproducibility

A global random seed was used:

```python
SEED = 123
```

### Feature Scaling

A StandardScaler was embedded directly inside the Pipeline.
This guarantees that standardization is fitted independently within each cross-validation fold.

---

## Hyperparameter Optimization (Optuna)

Hyperparameter tuning was performed using Optuna with the TPE sampler and a budget of 50 trials per model.
To improve computational efficiency, a MedianPruner was employed to terminate poorly performing configurations early during cross-validation

### Experimental Design
#### Single-Layer MLP vs RVFL

A single-hidden-layer MLP was first evaluated to compare both approaches under an identical shallow architecture.The single-layer MLP performed substantially worse than RVFL.

#### Deep MLP vs RVFL (Main Comparison)

Since the primary strength of perceptrons lies in deep learning, the final experiment allowed Optuna to search architectures ranging from 2 to 5 hidden layers.
This gave the MLP the best possible opportunity to compete against the custom RVFL implementation.

---

## Results on the Unseen Test Set

Despite allowing powerful deep architectures for the MLP (the best-performing configuration was a three-layer network), RVFL clearly outperformed the classical neural network.

| Metric | RVFL | MLP |
|----------|----------:|----------:|
| Mean Squared Error (MSE) | **0.1407** | 0.2098 |
| Coefficient of Determination ($R^2$) | **0.9159** | 0.8745 |

---

## Bootstrap Analysis and Statistical Significance

To validate the robustness of the results, a paired bootstrap procedure was conducted with $$B = 1000$$

### Confidence Interval for the MSE Difference

The 95% confidence interval for $$Z_{MSE}$$ was $$[-0.10408,\,-0.04121]$$.
Since the interval lies entirely below zero, it provides strong evidence that RVFL achieves a significantly lower prediction error.

### Confidence Interval for the $R^2$ Difference

The 95% confidence interval for $$Z_{R^2}$$ was $$[0.02523,\;0.06075]$$. 
Since the interval lies entirely above zero, it confirms the superiority of RVFL in terms of explained variance as well.

---

## Training Dynamics and Diagnostic Analysis

### Stability and Convergence (Optuna)

The optimization history revealed that the classical MLP converged very quickly.
Near-optimal configurations were discovered within approximately the first 10 trials, making the model relatively robust to small hyperparameter changes.
RVFL, in contrast, required substantially deeper exploration of the search space, with the best-performing configurations emerging only around trial 40.
This behavior is largely explained by the model's sensitivity to the randomly initialized hidden-layer weights.

### Hyperparameter Importance

For MLP:

- 54% of the model's performance depended on the number of hidden layers (n_layers),
- network width had only a marginal impact.

For RVFL, the influence was distributed much more evenly among:

- activation function,
- number of hidden neurons,
- scale,
- bias_scale.
- Direct Linkage Connections

Optuna ultimately discarded direct input-to-output connections (direct_linkage) in the optimal RVFL configuration.
This suggests that the raw input features did not contain sufficiently informative linear relationships.
Only after transformation through the random nonlinear hidden layer could the model effectively capture the underlying structure of the data.

### Function Shape and Residual Analysis

The MLP produced a relatively smooth regression function that captured the global trend of the dataset.
However, it tended to oversmooth local relationships, flatten predictions in densely populated regions and exhibit increasing residual variance near the boundaries of the input domain.

RVFL generated a more irregular ("jagged") function but followed local data clusters much more closely.
Its residual distribution was narrower, more symmetric and more tightly concentrated around zero.

---

## Critical Evaluation of RVFL

In this task, RVFL achieved superior predictive performance while requiring only a fraction of the training time needed by the MLP.
Since no backpropagation is involved, training reduces to solving a single regression problem.
As a result training is extremely fast, there is no risk of getting trapped in local minima, vanishing-gradient issues are completely avoided and the training process becomes deterministic once the random hidden layer is fixed.

In this study, these advantages were sufficient to outperform a significantly more complex deep neural network.

However the speed and strong predictive performance come at a memory cost.
To surpass the MLP, RVFL compensated for its lack of depth by employing a very wide architecture:

- RVFL: approximately 1900 neurons,
- MLP: fewer than 500 neurons in total.

For extremely large datasets or high-dimensional feature spaces, storing the resulting matrices in memory may become a significant hardware limitation.
In the present problem, however, this trade-off proved highly effective.
