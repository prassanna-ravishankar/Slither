# Slither Random Forest

**A Random Forest library with SVM local experts for computer vision tasks**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/prassanna-ravishankar/Slither)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## What is Slither?

Slither is a unique Random Forest implementation that uses **Support Vector Machine (SVM) local experts** at tree nodes instead of simple threshold-based decision rules. This design makes it particularly powerful for computer vision and high-dimensional classification tasks.

## Key Features

🌳 **Random Forest with SVM Local Experts**
: Each tree node contains a trained SVM classifier for complex decision boundaries

🔬 **Computer Vision Optimized**
: Designed for hypercolumn features and high-dimensional image data

🐍 **Scikit-learn Compatible**
: Modern Python API with fit/predict/predict_proba methods

⚡ **High Performance C++**
: Core implementation in modern C++17 with smart pointers and JSON serialization

📊 **Information Gain Splitting**
: Uses Shannon entropy for optimal tree structure

🔧 **Modern Development**
: Type hints, comprehensive testing, and excellent documentation

## Quick Example

```python
from slither import SlitherClassifier
import numpy as np

# Create classifier
clf = SlitherClassifier(
    n_estimators=50,
    max_depth=10,
    svm_c=0.5
)

# Train on your data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate
accuracy = clf.score(X_test, y_test)
```

## Why Slither?

### Traditional Random Forest
```
Tree Node: if feature[i] < threshold → left else right
```

### Slither Random Forest
```
Tree Node: SVM(features) → complex decision boundary
```

This architecture enables:

- **Complex Decision Boundaries**: SVMs can capture non-linear relationships
- **High-Dimensional Data**: Excellent performance on image features
- **Feature Interactions**: Linear combinations rather than single thresholds
- **Robust Classification**: Ensemble of SVM experts

## Research Background

Slither was originally developed for **road segmentation using hypercolumn features** and implements the methodology described in computer vision research. The library combines:

- Random Forest ensemble learning
- Support Vector Machine local experts
- Information Gain-based splitting
- Hypercolumn feature support

## Getting Started

=== "Installation"

    ```bash
    # Install from PyPI (coming soon)
    pip install slither-rf
    
    # Or build from source
    git clone https://github.com/prassanna-ravishankar/Slither.git
    cd Slither
    pip install -e .
    ```

=== "Quick Start"

    ```python
    from slither import SlitherClassifier
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=10)
    
    # Train classifier
    clf = SlitherClassifier(n_estimators=10)
    clf.fit(X, y)
    
    # Make predictions
    print(f"Accuracy: {clf.score(X, y):.3f}")
    ```

=== "Advanced Usage"

    ```python
    # Configure for computer vision
    clf = SlitherClassifier(
        n_estimators=100,
        max_depth=15,
        n_candidate_features=50,
        svm_c=0.5,
        n_jobs=4,
        verbose=True
    )
    
    # Train and save model
    clf.fit(X_train, y_train)
    clf.save_model("my_forest.json")
    
    # Load and use later
    clf_loaded = SlitherClassifier()
    clf_loaded.load_model("my_forest.json")
    ```

## Performance Characteristics

| Aspect | Traditional RF | Slither RF |
|--------|---------------|------------|
| **Node Decision** | Simple threshold | SVM classifier |
| **Training Speed** | Fast | Moderate (SVM training) |
| **Prediction Speed** | Fast | Fast |
| **Memory Usage** | Low | Moderate |
| **High-Dim Data** | Good | Excellent |
| **Complex Boundaries** | Limited | Excellent |

## Project Status

✅ **Phase 1 Complete**: C++ modernization with JSON serialization and smart pointers  
✅ **Phase 2 Complete**: Python API redesign for scikit-learn compatibility  
🔄 **Phase 3 In Progress**: Documentation and examples  
📋 **Phase 4 Planned**: Performance optimization and CI/CD

## Community

- **Documentation**: [https://prassanna-ravishankar.github.io/Slither/](https://prassanna-ravishankar.github.io/Slither/)
- **Source Code**: [https://github.com/prassanna-ravishankar/Slither](https://github.com/prassanna-ravishankar/Slither)
- **Issues**: [GitHub Issues](https://github.com/prassanna-ravishankar/Slither/issues)

## License

Slither is released under the [MIT License](about/license.md).