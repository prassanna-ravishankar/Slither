# Modern C++ API Reference

This page documents the modern C++ API with the new project structure using `include/slither/` headers.

## Header Organization

All headers are now properly namespaced under `include/slither/`:

```cpp
#include <slither/Forest.h>           // Core forest implementation
#include <slither/Tree.h>             // Individual tree structures  
#include <slither/Node.h>             // Tree node implementation
#include <slither/ForestTrainer.h>    // Training algorithms
#include <slither/Classification.h>   // Classification tasks
#include <slither/Regression.h>       // Regression tasks
#include <slither/DataPointCollection.h> // Data management
```

## Modern Usage Pattern

### Basic Classification Example

```cpp
#include <slither/Sherwood.h>
#include <slither/Classification.h>
#include <slither/ForestTrainer.h>
#include <slither/DataPointCollection.h>
#include <random>
#include <memory>

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main() {
    // Modern random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Configure training parameters
    TrainingParameters params;
    params.NumberOfTrees = 20;
    params.MaxDecisionLevels = 8;
    params.NumberOfCandidateFeatures = 5;
    params.NumberOfCandidateThresholds = 10;
    params.Verbose = true;
    
    // Prepare training data
    DataPointCollection trainingData;
    // ... add your data points ...
    
    // Train forest with modern interface
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
        gen,  // Modern RNG
        params,
        ClassificationTrainingContext(n_features, n_classes),
        trainingData
    );
    
    // Make predictions
    std::vector<float> probabilities(n_classes);
    forest->Apply(testPoint, probabilities);
    
    // Get predicted class
    int predicted_class = std::max_element(probabilities.begin(), probabilities.end()) 
                         - probabilities.begin();
    
    return 0;
}
```

### Advanced Forest Configuration

```cpp
#include <slither/ParallelForestTrainer.h>
#include <nlohmann/json.hpp>

// Configure for high-performance training
TrainingParameters params;
params.NumberOfTrees = 100;
params.MaxDecisionLevels = 15;
params.NumberOfCandidateFeatures = static_cast<int>(std::sqrt(n_features));
params.NumberOfCandidateThresholds = 20;
params.Verbose = true;

// Use parallel training for better performance
auto forest = ParallelForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
    gen,
    params,
    ClassificationTrainingContext(n_features, n_classes),
    trainingData
);

// Modern serialization with JSON
nlohmann::json forest_json;
forest->Serialize(forest_json);

// Save to file
std::ofstream file("my_forest.json");
file << forest_json.dump(4);  // Pretty-printed JSON
```

## Core Classes

### Forest<F, S>

The main forest container with modern smart pointer management.

```cpp
template<class F, class S>
class Forest {
public:
    // Modern constructor with smart pointers
    Forest() = default;
    
    // Add tree with move semantics
    void AddTree(std::unique_ptr<Tree<F, S>> tree);
    
    // Modern prediction interface
    void Apply(const DataPoint& dataPoint, std::vector<float>& result) const;
    
    // Serialization support
    void Serialize(nlohmann::json& json) const;
    void Deserialize(const nlohmann::json& json);
    
    // Modern accessors
    size_t TreeCount() const noexcept;
    const Tree<F, S>& GetTree(size_t index) const;
};
```

### Tree<F, S>

Individual tree implementation with RAII principles.

```cpp
template<class F, class S>
class Tree {
public:
    // Modern constructor
    explicit Tree(int maxDecisionLevels);
    
    // Prediction with const correctness
    void Apply(const DataPoint& dataPoint, std::vector<float>& result) const;
    
    // Node access
    const Node<F, S>& GetNode(NodeIndex nodeIndex) const;
    
    // Serialization
    void Serialize(nlohmann::json& json) const;
    
private:
    std::vector<std::unique_ptr<Node<F, S>>> nodes_;
    int maxDecisionLevels_;
};
```

### TrainingParameters

Modern configuration with sensible defaults.

```cpp
struct TrainingParameters {
    // Core parameters
    int NumberOfTrees = 10;
    int MaxDecisionLevels = 8;
    int NumberOfCandidateFeatures = 0;  // 0 = sqrt(n_features)
    int NumberOfCandidateThresholds = 10;
    
    // Performance
    bool Verbose = false;
    int NumberOfThreads = 0;  // 0 = hardware_concurrency()
    
    // Advanced
    float BaggingRatio = 1.0f;
    unsigned int RandomSeed = 0;  // 0 = random seed
    
    // Validation
    bool IsValid() const noexcept;
    void SetDefaults(int n_features) noexcept;
};
```

## Feature Response Functions

### LinearFeatureResponse

Modern SVM-based feature response with Eigen integration.

```cpp
class LinearFeatureResponse {
public:
    // Constructor with modern initialization
    LinearFeatureResponse() = default;
    explicit LinearFeatureResponse(int dimensions);
    
    // Modern prediction interface
    float GetResponse(const DataPoint& dataPoint) const;
    
    // Training with const correctness
    void TrainFromExample(const DataPoint& dataPoint, float target);
    
    // Serialization
    void Serialize(nlohmann::json& json) const;
    void Deserialize(const nlohmann::json& json);
    
private:
    Eigen::VectorXf weights_;
    float bias_ = 0.0f;
    bool trained_ = false;
};
```

## Statistics Aggregators

### HistogramAggregator

Efficient histogram-based statistics with modern containers.

```cpp
class HistogramAggregator {
public:
    // Modern constructors
    HistogramAggregator() = default;
    explicit HistogramAggregator(int nClasses);
    
    // Statistics accumulation
    void Aggregate(const DataPoint& dataPoint);
    void Aggregate(const HistogramAggregator& other);
    
    // Information gain calculation
    float CalculateInformationGain() const;
    
    // Modern accessors
    int GetClassCount() const noexcept;
    float GetClassProbability(int classIndex) const;
    const std::vector<int>& GetHistogram() const noexcept;
    
private:
    std::vector<int> histogram_;
    int sampleCount_ = 0;
    int nClasses_ = 0;
};
```

## Building with Modern CMake

### Using find_package()

```cmake
# In your CMakeLists.txt
find_package(Slither REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Slither::Slither)
```

### Manual Integration

```cmake
# Add as subdirectory
add_subdirectory(external/Slither)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Slither::Slither)
```

## Performance Considerations

### Memory Management

- All containers use modern RAII principles
- Smart pointers prevent memory leaks
- Move semantics for efficient data transfer

### Parallelization

```cpp
// Use ParallelForestTrainer for multi-core training
#include <slither/ParallelForestTrainer.h>

// OpenMP support is automatically detected
auto forest = ParallelForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
    gen, params, context, data
);
```

### Vectorization

- Eigen3 integration for SIMD operations
- Optimized linear algebra operations
- Cache-friendly data structures

## Error Handling

Modern exception safety with RAII:

```cpp
try {
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
        gen, params, context, data
    );
    
    // Forest is automatically cleaned up even if exceptions occur
    forest->Apply(testPoint, result);
    
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid training parameters: " << e.what() << std::endl;
} catch (const std::runtime_error& e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
}
```

## Migration from Legacy API

### Header Changes

```cpp
// Old (legacy)
#include "lib/Forest.h"
#include "source/Classification.h"

// New (modern)
#include <slither/Forest.h>
#include <slither/Classification.h>
```

### Memory Management

```cpp
// Old (manual)
Forest<LinearFeatureResponse, HistogramAggregator>* forest = 
    ForestTrainer<...>::TrainForest(...);
// ... use forest ...
delete forest;  // Manual cleanup

// New (automatic)
auto forest = ForestTrainer<...>::TrainForest(...);
// Automatic cleanup when forest goes out of scope
```

### Serialization

```cpp
// Old (binary/boost)
std::ofstream file("forest.dat", std::ios::binary);
boost::archive::binary_oarchive archive(file);
archive << forest;

// New (JSON)
nlohmann::json json;
forest->Serialize(json);
std::ofstream file("forest.json");
file << json.dump(4);
```

This modern API provides better performance, safety, and maintainability while preserving the core algorithmic advantages of Slither Random Forest.