# Random Forest with SVM Local Experts

This page provides a deep dive into Slither's unique architecture that combines Random Forest ensemble learning with Support Vector Machine (SVM) local experts at tree nodes.

## Algorithm Overview

Traditional Random Forest uses simple threshold-based decisions at each node:

```
if feature[i] < threshold:
    go_left()
else:
    go_right()
```

Slither Random Forest replaces this with SVM classifiers:

```
if svm_classifier.predict(features) == 0:
    go_left()
else:
    go_right()
```

This fundamental change enables more complex decision boundaries and better handling of non-linearly separable data.

## Mathematical Foundation

### Traditional Random Forest Split

A standard Random Forest node splits based on a single feature:

$$
\text{split}(x) = \begin{cases}
\text{left} & \text{if } x_j < \theta \\
\text{right} & \text{otherwise}
\end{cases}
$$

Where:
- $x_j$ is feature $j$
- $\theta$ is the threshold

### Slither SVM Split

A Slither node uses a linear combination of features:

$$
\text{split}(x) = \begin{cases}
\text{left} & \text{if } \mathbf{w}^T\mathbf{x} + b < 0 \\
\text{right} & \text{otherwise}
\end{cases}
$$

Where:
- $\mathbf{w}$ is the weight vector learned by SVM
- $b$ is the bias term
- $\mathbf{x}$ is the feature vector

### Information Gain Criterion

Both methods use Information Gain for split quality assessment:

$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where:
- $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$ is entropy
- $S$ is the current sample set
- $S_v$ are subsets after split
- $p_i$ is the proportion of class $i$

## Tree Construction Algorithm

### 1. Node Training Process

```python
def train_node(data, labels, depth, params):
    """Train a single node with SVM classifier."""
    
    # Check stopping criteria
    if should_stop(data, labels, depth, params):
        return create_leaf_node(labels)
    
    # Sample candidate features randomly
    candidate_features = random_sample(
        all_features, 
        params.n_candidate_features
    )
    
    # Train SVM with selected features
    X_subset = data[:, candidate_features]
    svm = LibSVM(C=params.svm_c)
    svm.fit(X_subset, binary_labels)
    
    # Split data based on SVM predictions
    predictions = svm.predict(X_subset)
    left_mask = (predictions == 0)
    right_mask = (predictions == 1)
    
    # Calculate information gain
    gain = calculate_information_gain(
        labels, labels[left_mask], labels[right_mask]
    )
    
    if gain < params.min_information_gain:
        return create_leaf_node(labels)
    
    # Recursively train children
    left_child = train_node(
        data[left_mask], labels[left_mask], 
        depth + 1, params
    )
    right_child = train_node(
        data[right_mask], labels[right_mask], 
        depth + 1, params
    )
    
    return Node(svm, left_child, right_child, candidate_features)
```

### 2. SVM Training Details

At each node, the SVM optimization problem is:

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^n \xi_i
$$

Subject to:
$$
y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

Where:
- $C$ is the regularization parameter (svm_c)
- $\xi_i$ are slack variables
- $y_i \in \{-1, +1\}$ are binary class labels

### 3. Multi-class Handling

For multi-class problems, Slither uses One-vs-Rest strategy:

1. **Binary Conversion**: At each node, convert multi-class to binary based on class distributions
2. **Majority Split**: Most frequent class becomes positive, others negative
3. **Balanced Split**: Alternatively, split classes to balance left/right subtrees

```python
def convert_to_binary(labels):
    """Convert multi-class to binary for SVM training."""
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    if len(unique_classes) <= 2:
        return labels  # Already binary
    
    # Strategy 1: Most frequent vs others
    majority_class = unique_classes[np.argmax(counts)]
    binary_labels = (labels == majority_class).astype(int)
    
    return binary_labels
```

## Feature Selection Strategy

### Random Feature Sampling

At each node, Slither randomly selects features to prevent overfitting:

```python
def select_candidate_features(n_features, n_candidates):
    """Randomly sample features for SVM training."""
    if n_candidates >= n_features:
        return np.arange(n_features)
    
    return np.random.choice(
        n_features, 
        size=n_candidates, 
        replace=False
    )
```

### Feature Subset Size

The number of candidate features affects model behavior:

- **Too few features**: Underfitting, poor decision boundaries
- **Too many features**: Overfitting, slow training
- **Recommended**: $\sqrt{n\_features}$ to $\log_2(n\_features)$

### Feature Scaling

SVM is sensitive to feature scales. Slither handles this by:

1. **Automatic Normalization**: Features are normalized within each node
2. **Standardization**: Mean removal and unit variance scaling
3. **Robust Scaling**: Using median and IQR for outlier resistance

## Ensemble Prediction

### Tree Prediction

For a single tree, prediction follows the SVM decisions:

```python
def predict_tree(sample, tree):
    """Predict using a single tree."""
    node = tree.root
    
    while not node.is_leaf():
        # Extract features used by this node
        features = sample[node.feature_indices]
        
        # Use SVM to decide direction
        if node.svm.predict(features.reshape(1, -1))[0] == 0:
            node = node.left_child
        else:
            node = node.right_child
    
    return node.class_probabilities
```

### Forest Ensemble

The forest combines predictions from all trees:

```python
def predict_forest(sample, forest):
    """Predict using the entire forest."""
    all_predictions = []
    
    for tree in forest.trees:
        pred = predict_tree(sample, tree)
        all_predictions.append(pred)
    
    # Average probabilities across trees
    ensemble_prob = np.mean(all_predictions, axis=0)
    
    # Return class with highest probability
    return np.argmax(ensemble_prob)
```

### Voting Strategies

Slither supports different ensemble strategies:

1. **Soft Voting** (default): Average class probabilities
2. **Hard Voting**: Majority vote of class predictions
3. **Weighted Voting**: Weight trees by their training accuracy

## Performance Analysis

### Computational Complexity

| Operation | Traditional RF | Slither RF |
|-----------|---------------|------------|
| **Node Training** | O(n log n) | O(n³) |
| **Tree Training** | O(n m log n) | O(n³ m d) |
| **Forest Training** | O(k n m log n) | O(k n³ m d) |
| **Prediction** | O(d) | O(d m) |

Where:
- n = number of samples
- m = number of features
- d = tree depth
- k = number of trees

### Memory Requirements

```python
# Estimated memory per tree (MB)
memory_per_tree = (
    (2 ** max_depth) *          # Maximum nodes
    n_candidate_features *      # SVM features
    8 *                         # Float64 bytes
    svm_support_vectors         # SVM model size
) / (1024 ** 2)

# Total forest memory
total_memory = n_estimators * memory_per_tree
```

### Training Time Factors

Training time is dominated by SVM optimization:

```python
# Approximate training time (seconds)
training_time = (
    n_estimators *              # Number of trees
    (2 ** max_depth) *          # Nodes per tree
    svm_training_time *         # SVM optimization
    feature_sampling_overhead   # Feature selection
)
```

## Hyperparameter Guidelines

### SVM Regularization (svm_c)

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.01 | High regularization | Small datasets, simple patterns |
| 0.1 | Moderate regularization | Standard classification |
| 1.0 | Balanced | Complex patterns, medium datasets |
| 10.0 | Low regularization | Large datasets, complex boundaries |

### Tree Depth (max_depth)

```python
# Recommended depth based on dataset size
if n_samples < 1000:
    max_depth = 5
elif n_samples < 10000:
    max_depth = 8
else:
    max_depth = 12
```

### Number of Trees (n_estimators)

```python
# Recommended tree count
if n_samples < 1000:
    n_estimators = 10
elif n_samples < 10000:
    n_estimators = 25
else:
    n_estimators = 50
```

### Feature Sampling (n_candidate_features)

```python
# Feature sampling strategies
sqrt_features = int(np.sqrt(n_features))
log_features = int(np.log2(n_features))

# Conservative: sqrt(n_features)
n_candidate_features = sqrt_features

# Aggressive: log2(n_features) 
n_candidate_features = log_features

# Maximum: all features (may overfit)
n_candidate_features = n_features
```

## Comparison with Other Methods

### vs Standard Random Forest

| Aspect | Standard RF | Slither RF |
|--------|-------------|------------|
| **Decision Boundary** | Axis-aligned | Oblique (any angle) |
| **Training Speed** | Fast | Moderate |
| **Prediction Speed** | Fast | Moderate |
| **Memory Usage** | Low | Medium |
| **Accuracy on Linear Data** | Good | Good |
| **Accuracy on Non-linear Data** | Good | Excellent |

### vs Gradient Boosting

| Aspect | Gradient Boosting | Slither RF |
|--------|-------------------|------------|
| **Training** | Sequential | Parallel |
| **Overfitting Risk** | High | Medium |
| **Interpretability** | Medium | Low |
| **Hyperparameter Sensitivity** | High | Medium |
| **Early Stopping** | Built-in | Manual |

### vs Neural Networks

| Aspect | Neural Networks | Slither RF |
|--------|-----------------|------------|
| **Training Data Requirements** | Large | Medium |
| **Feature Engineering** | Automatic | Manual |
| **Training Time** | Long | Medium |
| **Interpretability** | Very Low | Low |
| **Hyperparameter Tuning** | Complex | Moderate |

## Algorithm Limitations

### Current Limitations

1. **Linear SVMs Only**: No RBF or polynomial kernels (yet)
2. **Binary Splits**: Each node splits into exactly two children
3. **No Bagging**: Trees train on full dataset (not random samples)
4. **Memory Intensive**: SVM models stored at each node
5. **Training Time**: Slower than standard Random Forest

### Future Enhancements

1. **Kernel Support**: RBF and polynomial SVM kernels
2. **Bagging Implementation**: Random sampling of training data
3. **Pruning**: Post-training tree simplification
4. **Incremental Learning**: Online model updates
5. **GPU Acceleration**: CUDA-based SVM training

## Research Applications

Slither was specifically designed for computer vision tasks:

### Hypercolumn Features

```python
# Extract CNN features at multiple scales
def extract_hypercolumn_features(image, cnn_model):
    """Extract features from multiple CNN layers."""
    features = []
    
    for layer_name in ['conv1', 'conv3', 'conv5']:
        layer_output = cnn_model.get_layer(layer_name).output
        layer_features = extract_features(image, layer_output)
        features.append(layer_features)
    
    # Concatenate multi-scale features
    hypercolumn = np.concatenate(features, axis=-1)
    return hypercolumn

# Train Slither on hypercolumn features
clf = SlitherClassifier(
    n_estimators=30,
    max_depth=10,
    n_candidate_features=100,
    svm_c=0.5
)
clf.fit(hypercolumn_features, pixel_labels)
```

### Road Segmentation

The original application was road segmentation in satellite imagery:

1. **Feature Extraction**: CNN hypercolumn features
2. **Pixel Classification**: Per-pixel road/non-road prediction
3. **Spatial Consistency**: Post-processing with CRF
4. **Evaluation**: IoU and pixel accuracy metrics

## Implementation Details

### libSVM Integration

Slither uses libSVM for SVM optimization:

```cpp
// SVM training in C++
svm_parameter param;
param.svm_type = C_SVC;
param.kernel_type = LINEAR;
param.C = svm_c_value;
param.eps = 1e-3;

svm_model* model = svm_train(&problem, &param);
```

### Memory Management

Smart pointers ensure proper cleanup:

```cpp
// Modern C++ memory management
std::vector<std::unique_ptr<Tree<F,S>>> trees_;

// Automatic cleanup when forest is destroyed
```

### JSON Serialization

Models are saved in human-readable format:

```json
{
  "format_version": "1.0",
  "tree_count": 20,
  "trees": [
    {
      "root": {
        "node_type": "split",
        "svm_weights": [0.5, -0.3, 0.8],
        "svm_bias": 0.1,
        "feature_indices": [0, 5, 12],
        "left_child": {...},
        "right_child": {...}
      }
    }
  ]
}
```

This detailed explanation provides the mathematical foundation and implementation details needed to understand Slither's unique approach to ensemble learning with SVM local experts.