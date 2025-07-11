# Quick Start Guide

Get up and running with Slither Random Forest in minutes!

## Basic Usage

### Your First Classifier

```python
from slither import SlitherClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(
    n_samples=500, 
    n_features=20, 
    n_classes=3,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train classifier
clf = SlitherClassifier(
    n_estimators=20,    # 20 trees in the forest
    max_depth=8,        # Maximum tree depth
    svm_c=0.5,         # SVM regularization
    verbose=True        # Show training progress
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate performance
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### Model Persistence

```python
# Save trained model
clf.save_model("my_forest.json")

# Load model later
new_clf = SlitherClassifier()
new_clf.load_model("my_forest.json")

# Use loaded model
predictions = new_clf.predict(X_test)
```

## Real-World Example: Image Classification

Here's how to use Slither for a computer vision task:

```python
import numpy as np
from slither import SlitherClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load face dataset
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Features (pixels): {X_train.shape[1]}")
print(f"Classes (people): {len(np.unique(y))}")

# Create classifier optimized for high-dimensional data
clf = SlitherClassifier(
    n_estimators=30,           # More trees for better performance
    max_depth=10,              # Deeper trees for complex patterns
    n_candidate_features=100,  # More features to consider
    svm_c=1.0,                # Higher regularization for stability
    n_jobs=1,                  # Single thread for stability
    verbose=False              # Quiet training
)

# Train classifier
print("Training Slither Random Forest...")
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
predictions = clf.predict(X_test)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Classes detected: {len(clf.classes_)}")

# Show detailed results
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

## Parameter Guidelines

### For Small Datasets (< 1,000 samples)

```python
clf = SlitherClassifier(
    n_estimators=10,           # Fewer trees to avoid overfitting
    max_depth=5,               # Shallow trees
    n_candidate_features=10,   # Conservative feature sampling
    svm_c=0.1,                # Lower regularization
    verbose=True
)
```

### For Large Datasets (> 10,000 samples)

```python
clf = SlitherClassifier(
    n_estimators=100,          # More trees for better performance
    max_depth=15,              # Deeper trees
    n_candidate_features=50,   # More feature candidates
    svm_c=1.0,                # Higher regularization
    n_jobs=4,                  # Parallel training
    verbose=False
)
```

### For High-Dimensional Data (Computer Vision)

```python
clf = SlitherClassifier(
    n_estimators=50,           # Moderate number of trees
    max_depth=12,              # Deep enough for complex patterns
    n_candidate_features=100,  # Many features to choose from
    n_candidate_thresholds=20, # More threshold options
    svm_c=0.5,                # Balanced regularization
    n_jobs=2                   # Moderate parallelization
)
```

## Understanding the Output

### Training Progress

When `verbose=True`, you'll see:

```
[Loading Data ]
[Loading data DONE]
3 Classes
Running training...
Training tree 0...
optimization finished, #iter = 1245
nu = 0.324854
obj = -36.211168, rho = 0.250282
nSV = 85, nBSV = 68
Total nSV = 229
```

This shows:
- **Data loading**: Input validation and formatting
- **Class detection**: Number of unique classes found
- **Tree training**: Each tree trains sequentially  
- **SVM optimization**: Each node trains an SVM classifier
- **Support vectors**: Number of support vectors in each SVM

### Prediction Output

```python
# Predictions are class labels
predictions = clf.predict(X_test)
# Output: [0, 1, 2, 1, 0, ...]

# Probabilities sum to 1.0 for each sample
probabilities = clf.predict_proba(X_test)
# Output shape: (n_samples, n_classes)
# Example: [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], ...]
```

## Common Patterns

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Evaluate with cross-validation
clf = SlitherClassifier(n_estimators=20, verbose=False)
scores = cross_val_score(clf, X, y, cv=5)

print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [10, 20, 50],
    'max_depth': [5, 8, 12],
    'svm_c': [0.1, 0.5, 1.0]
}

# Grid search (warning: can be slow due to SVM training)
clf = SlitherClassifier(verbose=False)
grid_search = GridSearchCV(
    clf, param_grid, cv=3, 
    scoring='accuracy', n_jobs=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Feature Importance (Workaround)

Slither doesn't directly provide feature importance, but you can estimate it:

```python
from sklearn.inspection import permutation_importance

# Train classifier
clf = SlitherClassifier(n_estimators=20, verbose=False)
clf.fit(X_train, y_train)

# Compute permutation importance
importance = permutation_importance(
    clf, X_test, y_test, n_repeats=5, random_state=42
)

# Show top features
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
sorted_idx = importance.importances_mean.argsort()[::-1]

print("Top 10 most important features:")
for i in range(10):
    idx = sorted_idx[i]
    print(f"{feature_names[idx]}: {importance.importances_mean[idx]:.3f}")
```

## Performance Tips

### Training Speed

- **Reduce `n_estimators`**: Fewer trees = faster training
- **Limit `max_depth`**: Shallower trees = fewer SVM optimizations
- **Use `n_jobs=1`**: Parallel training can be unstable
- **Lower `svm_c`**: Faster SVM convergence

### Memory Usage

- **Smaller datasets**: Process in batches if needed
- **Reduce `n_candidate_features`**: Less memory per node
- **Single threading**: `n_jobs=1` uses less memory

### Prediction Speed

- **Fewer trees**: Faster ensemble predictions
- **Shallower trees**: Faster tree traversal
- **Model persistence**: Save/load trained models

## Next Steps

Now that you've got the basics, explore:

- [Examples](examples.md) - More detailed examples
- [Python API Reference](../api/python.md) - Complete API documentation
- [Performance Guide](../user-guide/performance.md) - Optimization tips
- [Random Forest with SVM](../user-guide/forest-svm.md) - Understanding the algorithm