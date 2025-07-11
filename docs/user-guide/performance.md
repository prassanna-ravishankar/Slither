# Performance Guide

This guide covers optimization strategies, benchmarking, and best practices for achieving optimal performance with Slither Random Forest.

## Performance Overview

Slither's performance characteristics differ from standard Random Forest due to SVM training at each node. Understanding these differences is key to optimal usage.

### Performance Profile

| Metric | Traditional RF | Slither RF | Notes |
|--------|---------------|------------|-------|
| **Training Speed** | Fast | Moderate | SVM optimization overhead |
| **Prediction Speed** | Fast | Moderate | SVM evaluation per node |
| **Memory Usage** | Low | Medium | SVM models stored per node |
| **Accuracy** | Good | Excellent | Better decision boundaries |
| **Scalability** | Excellent | Good | Limited by SVM complexity |

## Training Performance

### Time Complexity Analysis

Training time is dominated by SVM optimization:

```python
# Approximate training time factors
training_time = (
    n_estimators *           # Number of trees (linear)
    n_nodes *               # Nodes per tree (exponential in depth)
    svm_training_time *     # SVM optimization (cubic in samples)
    feature_overhead        # Feature selection (linear)
)

# SVM training complexity
svm_training_time = O(n_samples³) + O(n_features * n_samples²)
```

### Optimization Strategies

#### 1. Reduce Number of Trees

```python
# Performance vs Accuracy tradeoff
configurations = [
    {'n_estimators': 5,  'speed': 'very_fast', 'accuracy': 'good'},
    {'n_estimators': 10, 'speed': 'fast',      'accuracy': 'better'},
    {'n_estimators': 25, 'speed': 'moderate',  'accuracy': 'excellent'},
    {'n_estimators': 50, 'speed': 'slow',      'accuracy': 'excellent+'},
]

# Recommended starting point
clf = SlitherClassifier(n_estimators=10)
```

#### 2. Limit Tree Depth

```python
# Depth affects number of nodes exponentially
depth_analysis = {
    3: {'nodes': 8,   'training': 'very_fast'},
    5: {'nodes': 32,  'training': 'fast'},
    8: {'nodes': 256, 'training': 'moderate'},
    12: {'nodes': 4096, 'training': 'slow'},
}

# Conservative depth for speed
clf = SlitherClassifier(max_depth=5)
```

#### 3. Feature Sampling

```python
# Reduce features per SVM
import numpy as np

n_features = X.shape[1]
sampling_strategies = {
    'conservative': max(1, int(np.sqrt(n_features))),
    'balanced': max(1, int(np.log2(n_features))),  
    'aggressive': max(1, n_features // 10),
    'maximum': n_features  # Use all (slowest)
}

# Fast training configuration
clf = SlitherClassifier(
    n_candidate_features=sampling_strategies['aggressive'],
    n_candidate_thresholds=5  # Fewer threshold evaluations
)
```

#### 4. SVM Regularization

```python
# Higher C = more complex SVMs = slower training
regularization_guide = {
    0.01: 'Fast training, simple boundaries',
    0.1:  'Balanced speed/accuracy',
    1.0:  'Standard setting',
    10.0: 'Slow training, complex boundaries'
}

# Fast training setup
clf = SlitherClassifier(svm_c=0.01)
```

### Parallel Training

Currently, parallel training has limitations:

```python
# Single-threaded (recommended for stability)
clf = SlitherClassifier(n_jobs=1)

# Multi-threaded (experimental)
clf = SlitherClassifier(n_jobs=2)  # Use with caution
```

**Note**: Parallel training may cause instability due to libSVM threading issues.

### Training Performance Benchmark

```python
import time
import numpy as np
from slither import SlitherClassifier
from sklearn.datasets import make_classification

def benchmark_training_speed():
    """Benchmark training performance across configurations."""
    
    # Generate test dataset
    X, y = make_classification(
        n_samples=1000, n_features=50, n_classes=3, random_state=42
    )
    
    configurations = [
        {'name': 'Fast', 'n_estimators': 5, 'max_depth': 3, 'svm_c': 0.01},
        {'name': 'Balanced', 'n_estimators': 10, 'max_depth': 5, 'svm_c': 0.1},
        {'name': 'Accurate', 'n_estimators': 25, 'max_depth': 8, 'svm_c': 1.0},
        {'name': 'Maximum', 'n_estimators': 50, 'max_depth': 12, 'svm_c': 10.0},
    ]
    
    results = []
    for config in configurations:
        name = config.pop('name')
        
        clf = SlitherClassifier(verbose=False, **config)
        
        start_time = time.time()
        clf.fit(X, y)
        training_time = time.time() - start_time
        
        accuracy = clf.score(X, y)
        
        results.append({
            'config': name,
            'time': training_time,
            'accuracy': accuracy,
            'time_per_tree': training_time / config['n_estimators']
        })
    
    return results

# Run benchmark
results = benchmark_training_speed()
for r in results:
    print(f"{r['config']:8s}: {r['time']:5.1f}s, "
          f"acc={r['accuracy']:.3f}, {r['time_per_tree']:.2f}s/tree")
```

## Prediction Performance

### Prediction Complexity

```python
# Prediction time factors
prediction_time = (
    n_estimators *          # Number of trees to evaluate
    avg_tree_depth *        # Tree traversal depth
    svm_evaluation_time     # SVM computation per node
)

# SVM evaluation is O(n_features)
svm_evaluation_time = O(n_candidate_features)
```

### Prediction Optimization

#### 1. Model Simplification

```python
# Train with fewer trees for faster prediction
clf = SlitherClassifier(n_estimators=10)
clf.fit(X_train, y_train)

# Optional: Prune trees (manual implementation)
def prune_forest(clf, min_accuracy=0.95):
    """Remove trees that don't significantly contribute."""
    individual_scores = []
    
    for i, tree in enumerate(clf.forest_.trees_):
        # Evaluate individual tree performance
        single_tree_clf = SlitherClassifier(n_estimators=1)
        single_tree_clf.forest_.trees_ = [tree]
        score = single_tree_clf.score(X_test, y_test)
        individual_scores.append((i, score))
    
    # Keep only trees above threshold
    good_trees = [i for i, score in individual_scores if score > min_accuracy]
    clf.forest_.trees_ = [clf.forest_.trees_[i] for i in good_trees]
    
    return clf
```

#### 2. Feature Reduction

```python
# Reduce features for faster SVM evaluation
from sklearn.feature_selection import SelectKBest, f_classif

# Select most informative features
selector = SelectKBest(f_classif, k=20)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

# Train on reduced features
clf = SlitherClassifier()
clf.fit(X_train_reduced, y_train)
```

#### 3. Batch Prediction

```python
# Predict in batches for better memory efficiency
def predict_in_batches(clf, X, batch_size=1000):
    """Predict large datasets in batches."""
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pred = clf.predict(X[start:end])
        predictions[start:end] = batch_pred
    
    return predictions

# Use for large prediction tasks
large_predictions = predict_in_batches(clf, X_large)
```

### Prediction Benchmark

```python
def benchmark_prediction_speed():
    """Benchmark prediction performance."""
    
    X, y = make_classification(n_samples=5000, n_features=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    configurations = [
        {'trees': 5, 'depth': 3},
        {'trees': 10, 'depth': 5}, 
        {'trees': 25, 'depth': 8},
        {'trees': 50, 'depth': 12},
    ]
    
    for config in configurations:
        clf = SlitherClassifier(
            n_estimators=config['trees'],
            max_depth=config['depth'],
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        # Benchmark prediction time
        start_time = time.time()
        predictions = clf.predict(X_test)
        pred_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Trees: {config['trees']:2d}, Depth: {config['depth']:2d} | "
              f"Pred time: {pred_time:.3f}s, "
              f"Samples/sec: {len(X_test)/pred_time:.0f}, "
              f"Accuracy: {accuracy:.3f}")
```

## Memory Optimization

### Memory Usage Analysis

```python
def estimate_memory_usage(n_estimators, max_depth, n_features, 
                         n_candidate_features):
    """Estimate memory usage in MB."""
    
    # Approximate nodes per tree (worst case: full binary tree)
    max_nodes_per_tree = 2 ** (max_depth + 1) - 1
    
    # SVM model size per node
    svm_model_size = (
        n_candidate_features * 8 +  # Weight vector (float64)
        8 +                         # Bias term
        100                         # libSVM overhead
    )
    
    # Total memory estimate
    total_memory = (
        n_estimators *              # Number of trees
        max_nodes_per_tree *        # Nodes per tree
        svm_model_size              # Memory per node
    ) / (1024 ** 2)                # Convert to MB
    
    return total_memory

# Example usage
memory_mb = estimate_memory_usage(
    n_estimators=25, 
    max_depth=8, 
    n_features=100, 
    n_candidate_features=20
)
print(f"Estimated memory usage: {memory_mb:.1f} MB")
```

### Memory Reduction Strategies

#### 1. Reduce Tree Complexity

```python
# Memory-efficient configuration
clf = SlitherClassifier(
    n_estimators=10,           # Fewer trees
    max_depth=5,               # Shallower trees
    n_candidate_features=10,   # Fewer features per SVM
    n_jobs=1                   # Single thread (less memory)
)
```

#### 2. Feature Selection

```python
# Reduce feature dimensionality before training
from sklearn.decomposition import PCA

# Use PCA for dimensionality reduction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# Train on reduced data
clf = SlitherClassifier()
clf.fit(X_reduced, y)
```

#### 3. Data Type Optimization

```python
# Use float32 instead of float64 for memory savings
X_float32 = X.astype(np.float32)
y_int32 = y.astype(np.int32)

clf = SlitherClassifier()
clf.fit(X_float32, y_int32)
```

## Scalability Guidelines

### Dataset Size Recommendations

```python
# Configuration by dataset size
def get_optimal_config(n_samples, n_features):
    """Get optimal configuration based on dataset characteristics."""
    
    if n_samples < 1000:
        return {
            'n_estimators': 5,
            'max_depth': 3,
            'n_candidate_features': max(1, int(np.sqrt(n_features))),
            'svm_c': 0.1,
            'note': 'Small dataset - avoid overfitting'
        }
    elif n_samples < 10000:
        return {
            'n_estimators': 15,
            'max_depth': 6,
            'n_candidate_features': max(1, int(np.sqrt(n_features))),
            'svm_c': 0.5,
            'note': 'Medium dataset - balanced configuration'
        }
    elif n_samples < 100000:
        return {
            'n_estimators': 25,
            'max_depth': 8,
            'n_candidate_features': max(1, int(np.log2(n_features))),
            'svm_c': 1.0,
            'note': 'Large dataset - favor accuracy'
        }
    else:
        return {
            'n_estimators': 50,
            'max_depth': 10,
            'n_candidate_features': max(1, n_features // 20),
            'svm_c': 1.0,
            'note': 'Very large dataset - consider preprocessing'
        }

# Example usage
config = get_optimal_config(n_samples=5000, n_features=200)
print(f"Recommended config: {config}")

clf = SlitherClassifier(**{k: v for k, v in config.items() if k != 'note'})
```

### Feature Dimensionality Handling

```python
# High-dimensional data strategies
def handle_high_dimensions(X, y, max_features=100):
    """Handle high-dimensional data efficiently."""
    
    n_features = X.shape[1]
    
    if n_features <= max_features:
        # No reduction needed
        return X, y, None
    
    print(f"Reducing features from {n_features} to {max_features}")
    
    # Method 1: Feature selection
    selector = SelectKBest(f_classif, k=max_features)
    X_reduced = selector.fit_transform(X, y)
    
    return X_reduced, y, selector

# Usage
X_processed, y_processed, preprocessor = handle_high_dimensions(X, y)
clf = SlitherClassifier()
clf.fit(X_processed, y_processed)
```

## Comparative Benchmarks

### vs Other Algorithms

```python
def compare_algorithms(X, y):
    """Compare Slither with other algorithms."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    algorithms = {
        'Slither RF': SlitherClassifier(n_estimators=10, verbose=False),
        'Standard RF': RandomForestClassifier(n_estimators=10, random_state=42),
        'SVM': SVC(random_state=42),
        'Neural Net': MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    results = {}
    for name, clf in algorithms.items():
        print(f"Testing {name}...")
        
        # Training time
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Prediction time
        start = time.time()
        predictions = clf.predict(X_test)
        pred_time = time.time() - start
        
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        results[name] = {
            'train_time': train_time,
            'pred_time': pred_time,
            'accuracy': accuracy,
            'pred_per_sec': len(X_test) / pred_time
        }
    
    return results

# Run comparison
results = compare_algorithms(X, y)
for name, metrics in results.items():
    print(f"{name:12s}: acc={metrics['accuracy']:.3f}, "
          f"train={metrics['train_time']:.1f}s, "
          f"pred={metrics['pred_per_sec']:.0f} samples/s")
```

## Performance Monitoring

### Training Progress Tracking

```python
class PerformanceTracker:
    """Track training performance metrics."""
    
    def __init__(self):
        self.tree_times = []
        self.node_counts = []
        self.svm_iterations = []
    
    def track_training(self, clf, X, y):
        """Track training with detailed timing."""
        start_time = time.time()
        
        # Train with verbose output to capture SVM details
        clf.verbose = True
        clf.fit(X, y)
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'time_per_tree': total_time / clf.n_estimators,
            'estimated_nodes': (2 ** clf.max_depth - 1) * clf.n_estimators
        }

# Usage
tracker = PerformanceTracker()
metrics = tracker.track_training(clf, X_train, y_train)
print(f"Training completed in {metrics['total_time']:.1f}s")
print(f"Average time per tree: {metrics['time_per_tree']:.2f}s")
```

### Memory Monitoring

```python
import psutil
import os

def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage during function execution."""
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Peak memory
    peak = process.memory_info().rss / 1024 / 1024  # MB
    
    return result, {
        'baseline_mb': baseline,
        'peak_mb': peak,
        'increase_mb': peak - baseline
    }

# Monitor training memory usage
clf = SlitherClassifier(n_estimators=10)
_, memory_stats = monitor_memory_usage(clf.fit, X_train, y_train)

print(f"Memory usage during training:")
print(f"  Baseline: {memory_stats['baseline_mb']:.1f} MB")
print(f"  Peak: {memory_stats['peak_mb']:.1f} MB") 
print(f"  Increase: {memory_stats['increase_mb']:.1f} MB")
```

## Best Practices Summary

### For Training Performance
1. **Start small**: Begin with `n_estimators=10`, `max_depth=5`
2. **Use conservative regularization**: `svm_c=0.1`
3. **Limit features**: `n_candidate_features=sqrt(n_features)`
4. **Single threading**: `n_jobs=1` for stability

### For Prediction Performance
1. **Minimize trees**: Use smallest `n_estimators` that gives good accuracy
2. **Shallow trees**: Keep `max_depth` as low as possible
3. **Batch processing**: Use batches for large prediction tasks
4. **Feature selection**: Reduce dimensionality before training

### For Memory Efficiency
1. **Conservative parameters**: Smaller trees and fewer features
2. **Data types**: Use `float32` instead of `float64`
3. **Incremental processing**: Process large datasets in chunks
4. **Model pruning**: Remove underperforming trees post-training

### For Production Deployment
1. **Model persistence**: Save trained models with `save_model()`
2. **Input validation**: Validate feature dimensions and types
3. **Error handling**: Wrap predictions in try-catch blocks
4. **Monitoring**: Track prediction latency and memory usage

This comprehensive performance guide should help users optimize Slither Random Forest for their specific use cases and constraints.