# Documentation Style Guide

This guide ensures consistency across all Slither Random Forest documentation.

## Naming Conventions

### Project Name
- **Official Name**: Slither Random Forest
- **Package Name**: `slither` (Python import)
- **PyPI Name**: `slither-rf` (when available)
- **GitHub Repo**: `Slither`

### Class Names
- **Python**: `SlitherClassifier`, `SlitherRegressor` 
- **C++ Namespace**: `MicrosoftResearch::Cambridge::Sherwood`
- **C++ Templates**: `Forest<F,S>`, `Tree<F,S>`, etc.

## Installation Standards

### Primary Installation (User)
```bash
# Current development installation
pip install -e .

# Future PyPI installation (mention as "coming soon")
pip install slither-rf
```

### Development Installation
```bash
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither
pip install -e ".[dev,test]"
```

## Code Example Standards

### Python Parameter Defaults
```python
# Standard example configuration
clf = SlitherClassifier(
    n_estimators=20,        # Standard: 20 for examples
    max_depth=8,           # Standard: 8 for examples  
    n_candidate_features='sqrt',  # Use string, not int
    svm_c=0.5,            # Standard: 0.5 for examples
    random_state=42,      # Always use 42 for reproducibility
    n_jobs=-1,            # Use all cores when demonstrating
    verbose=False         # False unless specifically demonstrating verbosity
)
```

### C++ Parameter Defaults
```cpp
// Standard example configuration
TrainingParameters params;
params.NumberOfTrees = 20;                    // Match Python examples
params.MaxDecisionLevels = 8;                 // Match Python max_depth
params.NumberOfCandidateFeatures = 0;        // 0 = sqrt(n_features)
params.NumberOfCandidateThresholds = 10;     // Standard value
params.Verbose = false;                       // Match Python default
```

## API Documentation Standards

### Method Signatures (Python)
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'SlitherClassifier':
def predict(self, X: np.ndarray) -> np.ndarray:
def predict_proba(self, X: np.ndarray) -> np.ndarray:
def score(self, X: np.ndarray, y: np.ndarray) -> float:
```

### Parameter Names (Python)
- `n_estimators` (not `n_trees`)
- `max_depth` (not `max_decision_levels`)
- `n_candidate_features` (not `n_features`)
- `svm_c` (not `c` or `C`)
- `random_state` (not `seed`)
- `n_jobs` (not `n_threads`)

### Header Includes (C++)
```cpp
// Modern includes (always use these)
#include <slither/Forest.h>
#include <slither/Tree.h>
#include <slither/Classification.h>

// Never use legacy includes like:
// #include "lib/Forest.h"
// #include "source/Classification.h"
```

## File Structure References

### Always Reference Current Structure
```
Slither/
├── include/slither/     # C++ headers
├── src/                 # C++ implementation
├── python/slither/      # Python package
├── tests/               # All tests
├── benchmarks/          # Performance tests
├── docs/                # Documentation
└── data/                # Sample datasets
```

## Build System Standards

### CMake (C++)
```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

### Python Build
```bash
pip install -e .  # Development
# OR (future)
pip install slither-rf  # Release
```

## Serialization Standards

### Python Serialization
```python
# Primary method: pickle (always works)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Secondary: native JSON (if implemented)
clf.save_model('model.json')
clf_loaded = SlitherClassifier()
clf_loaded.load_model('model.json')
```

### C++ Serialization
```cpp
// Modern JSON serialization
#include <nlohmann/json.hpp>
nlohmann::json forest_json;
forest->Serialize(forest_json);
```

## Version References

- **Current Version**: 1.0.0 (development)
- **Python Requirements**: 3.8+
- **CMake Requirements**: 3.16+
- **C++ Standard**: C++17

## Cross-References

### Internal Links
- Use relative paths: `[Installation](../getting-started/installation.md)`
- Always verify links work in MkDocs build
- Use consistent section titles across files

### External Links
- GitHub repo: `https://github.com/prassanna-ravishankar/Slither`
- Documentation: `https://prassanna-ravishankar.github.io/Slither/`

## Voice and Tone

- **Technical but Accessible**: Explain concepts clearly without dumbing down
- **Consistent Terminology**: Use the same terms for the same concepts
- **Action-Oriented**: Use imperative mood for instructions
- **Example-Heavy**: Always include working code examples
- **Professional**: Avoid casual language, but keep it friendly

## Example Dataset Standards

### For Small Examples
```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)
```

### For Computer Vision Examples
```python
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target
```

This style guide ensures consistency across all documentation and should be followed for all new content and updates.