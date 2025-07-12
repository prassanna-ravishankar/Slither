# Python API Reference

Complete reference for the Slither Python API.

## SlitherClassifier

::: slither.SlitherClassifier
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Base Classes

::: slither.base.SlitherBase
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Exceptions

::: slither.exceptions
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Module Information

::: slither
    options:
      show_root_heading: false
      show_source: false
      members:
        - __version__
        - __author__
        - __email__

## Example Usage

### Basic Classification

```python
from slither import SlitherClassifier
import numpy as np

# Create classifier
clf = SlitherClassifier(
    n_estimators=50,
    max_depth=10,
    svm_c=0.5
)

# Fit model
X = np.random.random((100, 10))
y = np.random.randint(0, 3, 100)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### Parameter Configuration

```python
# Access parameters
params = clf.get_params()
print(params)

# Update parameters
clf.set_params(
    n_estimators=100,
    max_depth=15,
    svm_c=1.0
)

# Check specific parameters
print(f"Number of estimators: {clf.n_estimators}")
print(f"Maximum depth: {clf.max_depth}")
print(f"SVM regularization: {clf.svm_c}")
```

### Model Persistence

```python
# Save model
clf.save_model("my_model.json")

# Load model
new_clf = SlitherClassifier()
new_clf.load_model("my_model.json")

# Verify loaded model
assert new_clf._fitted == True
```

### Error Handling

```python
from slither.exceptions import SlitherNotFittedError, SlitherValidationError

try:
    # This will raise an error if model is not fitted
    unfitted_clf = SlitherClassifier()
    unfitted_clf.predict(X)
except SlitherNotFittedError as e:
    print(f"Model not fitted: {e}")

try:
    # This will raise an error for invalid parameters
    invalid_clf = SlitherClassifier(n_estimators=-5)
    invalid_clf._validate_parameters()
except SlitherValidationError as e:
    print(f"Invalid parameters: {e}")
```

## Parameter Guidelines

### For Different Dataset Sizes

#### Small Datasets (< 1,000 samples)
```python
clf = SlitherClassifier(
    n_estimators=10,           # Avoid overfitting
    max_depth=5,               # Shallow trees
    n_candidate_features=5,    # Conservative feature sampling
    svm_c=0.1                  # Lower regularization
)
```

#### Medium Datasets (1,000 - 10,000 samples)
```python
clf = SlitherClassifier(
    n_estimators=25,           # Balanced performance
    max_depth=8,               # Moderate depth
    n_candidate_features=15,   # More feature exploration
    svm_c=0.5                  # Standard regularization
)
```

#### Large Datasets (> 10,000 samples)
```python
clf = SlitherClassifier(
    n_estimators=50,           # Better ensemble
    max_depth=12,              # Deeper trees
    n_candidate_features=30,   # More feature candidates
    svm_c=1.0,                 # Higher regularization
    n_jobs=4                   # Parallel processing
)
```

### For Different Feature Dimensions

#### Low Dimensional (< 50 features)
```python
clf = SlitherClassifier(
    n_candidate_features=10,   # Most features available
    n_candidate_thresholds=10, # Standard threshold search
    svm_c=0.1                  # Lower complexity
)
```

#### High Dimensional (> 1,000 features)
```python
clf = SlitherClassifier(
    n_candidate_features=100,  # Sample many features
    n_candidate_thresholds=20, # More threshold options
    svm_c=0.5,                 # Balanced regularization
    max_depth=8                # Control overfitting
)
```

### For Different Problem Types

#### Computer Vision
```python
clf = SlitherClassifier(
    n_estimators=30,
    max_depth=10,
    n_candidate_features=50,
    svm_c=0.5,
    n_jobs=2
)
```

#### Text Classification
```python
clf = SlitherClassifier(
    n_estimators=20,
    max_depth=12,
    n_candidate_features=100,
    svm_c=0.1,                 # Lower C for sparse features
    n_jobs=1
)
```

#### Time Series Features
```python
clf = SlitherClassifier(
    n_estimators=40,
    max_depth=8,
    n_candidate_features=20,
    svm_c=1.0,                 # Higher regularization
    verbose=True
)
```

## Performance Considerations

### Memory Usage

The memory footprint depends on:

```python
# Approximate memory usage factors
memory_estimate = (
    n_estimators *           # Number of trees
    (2 ** max_depth) *       # Nodes per tree (worst case)
    n_features *             # Features per SVM
    8                        # Bytes per float64
) / (1024 ** 2)             # Convert to MB

print(f"Estimated memory: {memory_estimate:.1f} MB")
```

### Training Time

Training time scales with:

```python
# Training time factors
training_factors = {
    'n_estimators': 'Linear scaling',
    'max_depth': 'Exponential scaling (nodes per tree)',
    'n_candidate_features': 'Linear scaling (SVM complexity)',
    'n_candidate_thresholds': 'Linear scaling (optimization steps)',
    'svm_c': 'Affects SVM convergence time',
    'dataset_size': 'Linear to super-linear scaling'
}
```

### Prediction Speed

Prediction time is affected by:

```python
# Prediction speed factors
prediction_speed = {
    'n_estimators': 'Linear (more trees to evaluate)',
    'max_depth': 'Logarithmic (tree traversal)',
    'n_features': 'Linear (SVM evaluation)',
    'batch_size': 'Linear (more samples to predict)'
}
```

## Scikit-learn Compatibility

Slither follows scikit-learn conventions:

### Estimator Interface
- `fit(X, y)` - Train the model
- `predict(X)` - Make class predictions  
- `predict_proba(X)` - Get class probabilities
- `score(X, y)` - Calculate accuracy

### Parameter Management
- `get_params(deep=True)` - Get all parameters
- `set_params(**params)` - Set parameters

### Validation
- Input validation using `sklearn.utils.validation`
- Proper error messages following sklearn conventions
- Support for both numpy arrays and pandas DataFrames

### Cross-validation Support
```python
from sklearn.model_selection import cross_val_score

clf = SlitherClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Grid Search Support
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [5, 8, 10],
    'svm_c': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(
    SlitherClassifier(), param_grid, cv=3
)
grid_search.fit(X, y)
```

### Pipeline Support
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SlitherClassifier())
])

pipeline.fit(X, y)
predictions = pipeline.predict(X)
```