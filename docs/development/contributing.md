# Contributing to Slither

Thank you for your interest in contributing to Slither Random Forest! This guide will help you get started with contributing code, documentation, and other improvements.

## Getting Started

### Development Environment Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Slither.git
   cd Slither
   
   # Add upstream remote
   git remote add upstream https://github.com/prassanna-ravishankar/Slither.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Install development dependencies
   pip install -e ".[dev,test]"
   
   # Install pre-commit hooks
   pre-commit install
   
   # Build the project
   ./build.sh
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   python test/test_pyslither.py
   cd slither_build && ./slither_cpp --help
   ```

## Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Keep your branch updated
git fetch upstream
git rebase upstream/master
```

### Making Changes

1. **Code Changes**: Implement your feature or fix
2. **Tests**: Add or update tests for your changes
3. **Documentation**: Update relevant documentation
4. **Commit**: Make clear, descriptive commits

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: brief description

More detailed explanation of what this commit does and why.
Fixes #123 (if applicable)"
```

### Pull Request Process

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub
   - Use the PR template
   - Link related issues
   - Describe changes clearly

3. **Code Review Process**
   - Address reviewer feedback
   - Make requested changes
   - Keep discussion focused

## Code Style Guidelines

### C++ Code Style

Follow modern C++17 practices:

```cpp
// Good: Modern C++ style
class Forest {
private:
    std::vector<std::unique_ptr<Tree>> trees_;
    
public:
    Forest(size_t n_trees) : trees_(n_trees) {}
    
    void addTree(std::unique_ptr<Tree> tree) {
        trees_.push_back(std::move(tree));
    }
    
    [[nodiscard]] size_t size() const noexcept {
        return trees_.size();
    }
};

// Avoid: C-style patterns
class BadForest {
public:
    Tree** trees;  // Use std::vector instead
    int count;     // Use size_t for sizes
    
    BadForest() {
        trees = (Tree**)malloc(10 * sizeof(Tree*));  // Use RAII
    }
};
```

#### C++ Naming Conventions

```cpp
// Classes: PascalCase
class ForestTrainer;
class SvmClassifier;

// Methods and variables: camelCase  
void trainForest();
int numberOfTrees;

// Constants: UPPER_SNAKE_CASE
static constexpr int MAX_TREE_DEPTH = 20;

// Template parameters: single uppercase letter
template<typename T, typename U>
class MyClass;
```

#### Memory Management

```cpp
// Good: RAII and smart pointers
class Node {
private:
    std::unique_ptr<Node> leftChild_;
    std::unique_ptr<Node> rightChild_;
    
public:
    void setLeftChild(std::unique_ptr<Node> child) {
        leftChild_ = std::move(child);
    }
};

// Avoid: Manual memory management
class BadNode {
public:
    Node* left;
    Node* right;
    
    ~BadNode() {
        delete left;   // Error-prone
        delete right;
    }
};
```

### Python Code Style

Follow PEP 8 and modern Python practices:

```python
# Good: PEP 8 compliant
class SlitherClassifier:
    """Random Forest classifier with SVM local experts.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum depth of trees.
    """
    
    def __init__(self, n_estimators: int = 10, max_depth: int = 5) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._fitted = False
    
    def fit(self, X: NDArray, y: NDArray) -> "SlitherClassifier":
        """Fit the classifier to training data."""
        X, y = self._validate_input(X, y)
        # Implementation...
        return self
```

#### Type Hints

Use comprehensive type hints:

```python
from typing import Optional, Union, List, Dict, Any
from numpy.typing import NDArray
import numpy as np

class SlitherBase:
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        """Predict class labels for samples."""
        pass
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities for samples.""" 
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        pass
```

#### Error Handling

```python
# Custom exception hierarchy
class SlitherError(Exception):
    """Base exception for Slither-related errors."""
    pass

class SlitherNotFittedError(SlitherError):
    """Raised when using unfitted estimator."""
    pass

class SlitherValidationError(SlitherError):
    """Raised for input validation errors."""
    pass

# Usage in methods
def predict(self, X: NDArray) -> NDArray:
    if not self._fitted:
        raise SlitherNotFittedError(
            "This classifier has not been fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )
    
    X = self._validate_input(X)
    return self._predict_impl(X)
```

## Testing Guidelines

### Test Structure

Organize tests by functionality:

```
test/
├── test_pyslither.py           # Main Python tests
├── test_classifier.py          # Classifier-specific tests
├── test_serialization.py       # Model persistence tests
├── test_compatibility.py       # Scikit-learn compatibility
├── test_performance.py         # Performance benchmarks
└── cpp/
    ├── test_forest.cpp         # C++ forest tests
    ├── test_tree.cpp           # C++ tree tests
    └── test_serialization.cpp  # C++ serialization tests
```

### Python Test Examples

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from slither import SlitherClassifier

class TestSlitherClassifier:
    """Test suite for SlitherClassifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=200, 
            n_features=10, 
            n_classes=3, 
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_fit_predict_basic(self, sample_data):
        """Test basic fit and predict functionality."""
        X_train, X_test, y_train, y_test = sample_data
        
        clf = SlitherClassifier(n_estimators=5, verbose=False)
        clf.fit(X_train, y_train)
        
        # Test predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert probabilities.shape == (len(X_test), len(np.unique(y_train)))
        assert np.all(predictions >= 0)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            SlitherClassifier(n_estimators=-1)
        
        with pytest.raises(ValueError):
            SlitherClassifier(max_depth=0)
    
    def test_unfitted_error(self, sample_data):
        """Test error when using unfitted classifier."""
        X_train, X_test, y_train, y_test = sample_data
        
        clf = SlitherClassifier()
        
        with pytest.raises(SlitherNotFittedError):
            clf.predict(X_test)
        
        with pytest.raises(SlitherNotFittedError):
            clf.predict_proba(X_test)
    
    @pytest.mark.parametrize("n_estimators", [1, 5, 10])
    @pytest.mark.parametrize("max_depth", [3, 5, 8])
    def test_parameter_combinations(self, sample_data, n_estimators, max_depth):
        """Test various parameter combinations."""
        X_train, X_test, y_train, y_test = sample_data
        
        clf = SlitherClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

# Performance tests
@pytest.mark.slow
class TestSlitherPerformance:
    """Performance and benchmark tests."""
    
    def test_large_dataset_performance(self):
        """Test performance on larger datasets."""
        X, y = make_classification(
            n_samples=5000, 
            n_features=50, 
            n_classes=5,
            random_state=42
        )
        
        clf = SlitherClassifier(n_estimators=10, verbose=False)
        
        import time
        start_time = time.time()
        clf.fit(X, y)
        training_time = time.time() - start_time
        
        # Reasonable training time (adjust based on hardware)
        assert training_time < 300, f"Training took {training_time:.1f}s, expected < 300s"
        
        # Test prediction speed
        start_time = time.time()
        predictions = clf.predict(X[:1000])
        prediction_time = time.time() - start_time
        
        # Should predict 1000 samples in under 10 seconds
        assert prediction_time < 10, f"Prediction took {prediction_time:.1f}s, expected < 10s"
```

### C++ Test Examples

```cpp
#include <gtest/gtest.h>
#include "Forest.h"
#include "ForestTrainer.h"
#include "Classification.h"

using namespace Sherwood;

class ForestTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test data
        generateTestData();
        
        // Set up training parameters
        params.NumberOfTrees = 5;
        params.MaxDecisionLevels = 3;
        params.NumberOfCandidateFeatures = 3;
        params.Verbose = false;
    }
    
    void generateTestData() {
        // Generate simple 2D classification data
        cv::Mat features(100, 2, CV_32F);
        cv::Mat labels(100, 1, CV_32S);
        
        cv::randu(features, 0.0, 1.0);
        
        for (int i = 0; i < 100; i++) {
            float x = features.at<float>(i, 0);
            float y = features.at<float>(i, 1);
            labels.at<int>(i, 0) = (x + y > 1.0) ? 1 : 0;
        }
        
        testData.LoadData(features, labels);
    }
    
    DataPointCollection testData;
    TrainingParameters params;
};

TEST_F(ForestTest, BasicTraining) {
    ClassificationTrainingContext<LinearFeatureResponse> context(2);
    
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
        ::TrainForest(testData, params, context);
    
    ASSERT_NE(forest, nullptr);
    EXPECT_EQ(forest->TreeCount(), params.NumberOfTrees);
}

TEST_F(ForestTest, PredictionAccuracy) {
    ClassificationTrainingContext<LinearFeatureResponse> context(2);
    
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
        ::TrainForest(testData, params, context);
    
    // Test predictions on training data
    int correct = 0;
    for (int i = 0; i < testData.Count(); i++) {
        auto dataPoint = testData.GetDataPoint(i);
        auto prediction = forest->Apply(dataPoint);
        
        int predictedClass = prediction.GetMaxBin();
        int actualClass = testData.GetIntegerLabel(i);
        
        if (predictedClass == actualClass) {
            correct++;
        }
    }
    
    double accuracy = static_cast<double>(correct) / testData.Count();
    EXPECT_GT(accuracy, 0.7);  // Should achieve reasonable accuracy
}

TEST_F(ForestTest, Serialization) {
    ClassificationTrainingContext<LinearFeatureResponse> context(2);
    
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
        ::TrainForest(testData, params, context);
    
    // Save to JSON
    std::string filename = "test_forest.json";
    forest->SerializeJson(filename);
    
    // Load from JSON
    auto loadedForest = std::make_unique<Forest<LinearFeatureResponse, HistogramAggregator>>();
    loadedForest->DeserializeJson(filename);
    
    // Compare predictions
    auto dataPoint = testData.GetDataPoint(0);
    auto originalPred = forest->Apply(dataPoint);
    auto loadedPred = loadedForest->Apply(dataPoint);
    
    EXPECT_EQ(originalPred.GetMaxBin(), loadedPred.GetMaxBin());
    
    // Cleanup
    std::remove(filename.c_str());
}
```

## Documentation Guidelines

### Code Documentation

#### C++ Documentation

Use Doxygen-style comments:

```cpp
/**
 * @brief Random Forest with SVM local experts.
 * 
 * This class implements a Random Forest where each tree node
 * uses an SVM classifier instead of simple threshold splits.
 * 
 * @tparam F Feature response function type
 * @tparam S Statistics aggregator type
 */
template<class F, class S>
class Forest {
public:
    /**
     * @brief Apply the forest to a data point.
     * 
     * @param dataPoint Input data point to classify
     * @return Aggregated prediction from all trees
     * 
     * @throws std::runtime_error if forest is empty
     */
    S Apply(const DataPoint& dataPoint) const;
    
private:
    std::vector<std::unique_ptr<Tree<F, S>>> trees_;  ///< Forest trees
};
```

#### Python Documentation

Use NumPy-style docstrings:

```python
class SlitherClassifier:
    """Random Forest classifier with SVM local experts.
    
    This classifier implements a Random Forest where each tree node
    uses Support Vector Machine (SVM) classifiers instead of simple
    threshold-based splits, enabling more complex decision boundaries.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    max_depth : int, default=5
        The maximum depth of the trees.
    svm_c : float, default=1.0
        Regularization parameter for SVM classifiers.
    verbose : bool, default=False
        Whether to print training progress.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels (single output problem).
    n_classes_ : int
        The number of classes (single output problem).
    n_features_in_ : int
        Number of features seen during fit.
    
    Examples
    --------
    >>> from slither import SlitherClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2)
    >>> clf = SlitherClassifier(n_estimators=10)
    >>> clf.fit(X, y)
    SlitherClassifier(n_estimators=10)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    
    References
    ----------
    .. [1] Criminisi, A., Shotton, J., & Konukoglu, E. (2012). 
           Decision forests: A unified framework for classification, 
           regression, density estimation, manifold learning and 
           semi-supervised learning.
    """
    
    def fit(self, X, y):
        """Fit the Random Forest classifier.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        self : object
            Fitted estimator.
        
        Raises
        ------
        ValueError
            If X and y have inconsistent numbers of samples.
        SlitherValidationError
            If input validation fails.
        """
        pass
```

### Documentation Structure

Follow the established structure:

```
docs/
├── index.md                    # Main landing page
├── getting-started/           # User onboarding
│   ├── installation.md
│   ├── quickstart.md
│   └── examples.md
├── user-guide/               # Detailed usage guides
│   ├── overview.md
│   ├── forest-svm.md
│   ├── performance.md
│   └── cpp-library.md
├── api/                      # API reference
│   ├── python.md
│   └── cpp.md
└── development/              # Development docs
    ├── building.md
    ├── contributing.md
    └── architecture.md
```

## Performance Considerations

### Benchmarking Changes

Always benchmark performance-critical changes:

```python
# benchmark_script.py
import time
import numpy as np
from slither import SlitherClassifier
from sklearn.datasets import make_classification

def benchmark_training(n_samples_list, n_features_list):
    """Benchmark training performance."""
    results = []
    
    for n_samples in n_samples_list:
        for n_features in n_features_list:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=3,
                random_state=42
            )
            
            clf = SlitherClassifier(n_estimators=10, verbose=False)
            
            start_time = time.time()
            clf.fit(X, y)
            training_time = time.time() - start_time
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'training_time': training_time
            })
            
            print(f"Samples: {n_samples:4d}, Features: {n_features:2d}, "
                  f"Time: {training_time:.2f}s")
    
    return results

# Run benchmark
results = benchmark_training([100, 500, 1000], [10, 50, 100])
```

### Memory Profiling

```python
# memory_profile.py
import tracemalloc
from slither import SlitherClassifier

def profile_memory_usage():
    """Profile memory usage during training."""
    tracemalloc.start()
    
    # Training code
    X, y = make_classification(n_samples=1000, n_features=50)
    clf = SlitherClassifier(n_estimators=20)
    clf.fit(X, y)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

profile_memory_usage()
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ master, feat/* ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: [3.8, 3.9, "3.10", "3.11"]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        python -m pytest test/ -v
        
    - name: Run C++ tests
      run: |
        ./build.sh
        cd slither_build
        ./slither_cpp --help
```

## Release Process

### Version Management

Follow semantic versioning (SemVer):

- **Major**: Breaking changes (2.0.0 → 3.0.0)
- **Minor**: New features (2.0.0 → 2.1.0)  
- **Patch**: Bug fixes (2.0.0 → 2.0.1)

### Release Checklist

1. **Pre-release**
   - [ ] Update CHANGELOG.md
   - [ ] Update version numbers
   - [ ] Run full test suite
   - [ ] Update documentation
   - [ ] Performance benchmarks

2. **Release**
   - [ ] Create release branch
   - [ ] Tag release
   - [ ] Build packages
   - [ ] Upload to PyPI
   - [ ] Update documentation site

3. **Post-release**
   - [ ] Announce release
   - [ ] Update example notebooks
   - [ ] Close milestone issues

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Code Review**: Pull request feedback

### Mentorship

New contributors can:

- Look for "good first issue" labels
- Ask questions in GitHub Discussions
- Request code review guidance
- Join development discussions

## Recognition

Contributors are recognized through:

- **Commit Attribution**: Co-authored-by tags
- **Release Notes**: Contributor acknowledgments  
- **Documentation**: Contributors list
- **GitHub**: Automatic contributor tracking

Thank you for contributing to Slither Random Forest!