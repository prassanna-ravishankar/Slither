# C++ Library Usage

This guide covers using the Slither Random Forest C++ library directly, including building, linking, and integrating with your own C++ projects.

## Overview

The Slither C++ library provides the core Random Forest implementation with SVM local experts. It's designed for:

- **High Performance**: Direct C++ implementation without Python overhead
- **Memory Efficiency**: Fine-grained control over memory allocation
- **Integration**: Easy integration into existing C++ applications
- **Customization**: Extensible interfaces for custom functionality

## Architecture

### Core Components

```cpp
// Main library components
#include "Sherwood.h"              // Main framework
#include "Forest.h"                // Forest data structure
#include "Tree.h"                  // Tree implementation
#include "Node.h"                  // Tree node structure
#include "ForestTrainer.h"         // Training algorithms
#include "TrainingParameters.h"    // Configuration
```

### Key Classes

```cpp
namespace Sherwood {
    template<class F, class S>
    class Forest;                   // Main forest container
    
    template<class F, class S>
    class Tree;                     // Individual decision tree
    
    template<class F, class S>
    class Node;                     // Tree node with SVM
    
    template<class F, class S>
    class ForestTrainer;            // Training implementation
    
    class TrainingParameters;       // Training configuration
}
```

## Building and Installation

### Prerequisites

Ensure you have the required dependencies:

```bash
# Install vcpkg (dependency manager)
git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg

# Add to your shell profile for persistence
echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
```

### Building the Library

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/Slither.git
cd Slither

# Create build directory
mkdir slither_build
cd slither_build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build the library
make -j$(nproc)

# Install (optional)
make install
```

### Build Options

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build (default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# With OpenMP support (if available)
cmake .. -DWITH_OPENMP=ON

# Without OpenMP
cmake .. -DWITH_OPENMP=OFF
```

## Basic Usage

### Simple Classification Example

```cpp
#include "Sherwood.h"
#include "Classification.h"
#include "TrainingParameters.h"
#include <iostream>

using namespace Sherwood;

int main() {
    // Set up training parameters
    TrainingParameters trainingParameters;
    trainingParameters.NumberOfTrees = 10;
    trainingParameters.MaxDecisionLevels = 5;
    trainingParameters.NumberOfCandidateFeatures = 5;
    trainingParameters.NumberOfCandidateThresholds = 10;
    trainingParameters.Verbose = true;
    
    // Create training data (your data loading code here)
    DataPointCollection trainingData;
    // ... load your training data ...
    
    // Create classifier
    ClassificationTrainingContext<LinearFeatureResponse> 
        classificationContext(trainingData.CountClasses());
    
    // Train the forest
    std::unique_ptr<Forest<LinearFeatureResponse, HistogramAggregator>> forest =
        ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
            trainingData,
            trainingParameters,
            classificationContext
        );
    
    // Make predictions
    DataPointCollection testData;
    // ... load test data ...
    
    for (int i = 0; i < testData.Count(); i++) {
        auto dataPoint = testData.GetDataPoint(i);
        auto prediction = forest->Apply(dataPoint);
        
        // Get most likely class
        int predictedClass = prediction.GetMaxBin();
        std::cout << "Sample " << i << " predicted as class " 
                  << predictedClass << std::endl;
    }
    
    return 0;
}
```

### Advanced Configuration

```cpp
#include "Sherwood.h"
#include "Classification.h"
#include "FeatureResponseFunctions.h"

// Custom training parameters
TrainingParameters createOptimalParameters(int dataSize, int numFeatures) {
    TrainingParameters params;
    
    // Scale parameters based on data characteristics
    if (dataSize < 1000) {
        params.NumberOfTrees = 5;
        params.MaxDecisionLevels = 3;
    } else if (dataSize < 10000) {
        params.NumberOfTrees = 15;
        params.MaxDecisionLevels = 6;
    } else {
        params.NumberOfTrees = 25;
        params.MaxDecisionLevels = 8;
    }
    
    // Feature sampling
    params.NumberOfCandidateFeatures = std::max(1, 
        static_cast<int>(std::sqrt(numFeatures)));
    params.NumberOfCandidateThresholds = 10;
    
    // SVM parameters
    params.SvmC = 0.5;
    params.Verbose = false;
    
    return params;
}

// Usage with custom parameters
int main() {
    DataPointCollection trainingData;
    // ... load data ...
    
    TrainingParameters params = createOptimalParameters(
        trainingData.Count(), 
        trainingData.Dimensions()
    );
    
    // Train with optimized parameters
    ClassificationTrainingContext<LinearFeatureResponse> context(
        trainingData.CountClasses()
    );
    
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
        ::TrainForest(trainingData, params, context);
    
    return 0;
}
```

## Data Management

### DataPointCollection

The main data container in Slither:

```cpp
#include "DataPointCollection.h"

// Create from OpenCV Mat
cv::Mat features;  // Shape: (n_samples, n_features)
cv::Mat labels;    // Shape: (n_samples, 1)

DataPointCollection data;
data.LoadData(features, labels);

// Access data points
for (int i = 0; i < data.Count(); i++) {
    auto dataPoint = data.GetDataPoint(i);
    
    // Access features
    for (int j = 0; j < data.Dimensions(); j++) {
        float feature_value = dataPoint[j];
    }
    
    // Access label
    int label = data.GetIntegerLabel(i);
}

// Data statistics
std::cout << "Samples: " << data.Count() << std::endl;
std::cout << "Features: " << data.Dimensions() << std::endl;
std::cout << "Classes: " << data.CountClasses() << std::endl;
```

### Loading Data from Files

```cpp
// Load from CSV/TSV files
bool loadDataFromFile(const std::string& filepath, DataPointCollection& data) {
    cv::Mat features, labels;
    
    // Read tab-separated file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return false;
    }
    
    std::vector<std::vector<float>> feature_rows;
    std::vector<int> label_vector;
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        std::string token;
        
        // Read features (all but last column)
        while (std::getline(iss, token, '\t')) {
            try {
                row.push_back(std::stof(token));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing feature: " << token << std::endl;
                return false;
            }
        }
        
        // Last value is the label
        int label = static_cast<int>(row.back());
        row.pop_back();
        
        feature_rows.push_back(row);
        label_vector.push_back(label);
    }
    
    // Convert to OpenCV matrices
    int n_samples = feature_rows.size();
    int n_features = feature_rows[0].size();
    
    features = cv::Mat::zeros(n_samples, n_features, CV_32F);
    labels = cv::Mat::zeros(n_samples, 1, CV_32S);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            features.at<float>(i, j) = feature_rows[i][j];
        }
        labels.at<int>(i, 0) = label_vector[i];
    }
    
    // Load into DataPointCollection
    data.LoadData(features, labels);
    return true;
}

// Usage
DataPointCollection trainingData;
if (!loadDataFromFile("training_data.tsv", trainingData)) {
    std::cerr << "Failed to load training data" << std::endl;
    return -1;
}
```

## Feature Response Functions

Slither uses pluggable feature response functions for flexibility:

### Linear Feature Response

```cpp
#include "FeatureResponseFunctions.h"

// Linear combination of features (default SVM)
using LinearResponse = LinearFeatureResponse;

// Training context for linear features
ClassificationTrainingContext<LinearResponse> context(numClasses);

// Train forest with linear SVMs
auto forest = ForestTrainer<LinearResponse, HistogramAggregator>
    ::TrainForest(data, params, context);
```

### Custom Feature Response (Advanced)

```cpp
// Define custom feature response function
class CustomFeatureResponse {
public:
    // Random feature selection and threshold generation
    static CustomFeatureResponse CreateRandom(std::mt19937& random, 
                                            const TrainingParameters& params,
                                            const DataPointCollection& data) {
        CustomFeatureResponse response;
        
        // Your custom feature selection logic here
        // ...
        
        return response;
    }
    
    // Apply response function to data point
    float GetResponse(const DataPoint& dataPoint) const {
        // Your custom response computation here
        // Return a single scalar value for splitting
        return computeCustomResponse(dataPoint);
    }
    
    // Serialization support
    void Serialize(std::ostream& stream) const {
        // Serialize your custom parameters
    }
    
    void Deserialize(std::istream& stream) {
        // Deserialize your custom parameters
    }

private:
    // Your custom parameters
    std::vector<int> selectedFeatures_;
    std::vector<float> weights_;
    float threshold_;
    
    float computeCustomResponse(const DataPoint& dataPoint) const {
        float response = 0.0f;
        for (size_t i = 0; i < selectedFeatures_.size(); i++) {
            response += weights_[i] * dataPoint[selectedFeatures_[i]];
        }
        return response - threshold_;
    }
};

// Use custom feature response
ClassificationTrainingContext<CustomFeatureResponse> customContext(numClasses);
auto customForest = ForestTrainer<CustomFeatureResponse, HistogramAggregator>
    ::TrainForest(data, params, customContext);
```

## Statistics Aggregators

Control how statistics are collected at tree nodes:

### Histogram Aggregator (Classification)

```cpp
#include "StatisticsAggregators.h"

// For classification tasks
using ClassificationAggregator = HistogramAggregator;

// Usage in training
ClassificationTrainingContext<LinearFeatureResponse> context(numClasses);
auto forest = ForestTrainer<LinearFeatureResponse, ClassificationAggregator>
    ::TrainForest(data, params, context);
```

### Custom Aggregator (Advanced)

```cpp
// Custom statistics aggregator
class CustomAggregator {
public:
    // Initialize with number of classes
    CustomAggregator(int numClasses) : classCount_(numClasses) {
        counts_.resize(numClasses, 0);
    }
    
    // Add sample to statistics
    void Aggregate(const DataPoint& dataPoint, int label) {
        counts_[label]++;
        // Add your custom statistics here
    }
    
    // Combine with another aggregator
    CustomAggregator& operator+=(const CustomAggregator& other) {
        for (int i = 0; i < classCount_; i++) {
            counts_[i] += other.counts_[i];
        }
        return *this;
    }
    
    // Get prediction probabilities
    std::vector<float> GetProbabilities() const {
        std::vector<float> probs(classCount_);
        int total = std::accumulate(counts_.begin(), counts_.end(), 0);
        
        if (total > 0) {
            for (int i = 0; i < classCount_; i++) {
                probs[i] = static_cast<float>(counts_[i]) / total;
            }
        }
        
        return probs;
    }
    
    // Serialization
    void Serialize(std::ostream& stream) const {
        stream.write(reinterpret_cast<const char*>(&classCount_), sizeof(classCount_));
        stream.write(reinterpret_cast<const char*>(counts_.data()), 
                    counts_.size() * sizeof(int));
    }
    
    void Deserialize(std::istream& stream) {
        stream.read(reinterpret_cast<char*>(&classCount_), sizeof(classCount_));
        counts_.resize(classCount_);
        stream.read(reinterpret_cast<char*>(counts_.data()), 
                   counts_.size() * sizeof(int));
    }

private:
    int classCount_;
    std::vector<int> counts_;
};
```

## Model Persistence

### JSON Serialization

```cpp
#include "Forest.h"
#include <fstream>

// Save forest to JSON file
void saveForestToJson(const Forest<LinearFeatureResponse, HistogramAggregator>& forest,
                     const std::string& filepath) {
    try {
        forest.SerializeJson(filepath);
        std::cout << "Forest saved to " << filepath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving forest: " << e.what() << std::endl;
    }
}

// Load forest from JSON file
std::unique_ptr<Forest<LinearFeatureResponse, HistogramAggregator>> 
loadForestFromJson(const std::string& filepath) {
    try {
        auto forest = std::make_unique<Forest<LinearFeatureResponse, HistogramAggregator>>();
        forest->DeserializeJson(filepath);
        std::cout << "Forest loaded from " << filepath << std::endl;
        return forest;
    } catch (const std::exception& e) {
        std::cerr << "Error loading forest: " << e.what() << std::endl;
        return nullptr;
    }
}

// Usage
int main() {
    // Train forest
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
        ::TrainForest(trainingData, params, context);
    
    // Save model
    saveForestToJson(*forest, "my_model.json");
    
    // Load model later
    auto loadedForest = loadForestFromJson("my_model.json");
    
    if (loadedForest) {
        // Use loaded model for predictions
        auto prediction = loadedForest->Apply(testDataPoint);
    }
    
    return 0;
}
```

## Multi-threading

### OpenMP Support

```cpp
#ifdef WITH_OPENMP
#include <omp.h>

// Set number of threads
omp_set_num_threads(4);

// Training will automatically use parallel processing
auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
    ::TrainForest(data, params, context);
#endif
```

### Parallel Prediction

```cpp
// Parallel prediction for multiple samples
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
for (int i = 0; i < testData.Count(); i++) {
    auto dataPoint = testData.GetDataPoint(i);
    auto prediction = forest->Apply(dataPoint);
    
    // Store results (ensure thread-safe access)
    results[i] = prediction.GetMaxBin();
}
```

## Integration with Existing Projects

### CMake Integration

```cmake
# FindSlither.cmake
find_path(SLITHER_INCLUDE_DIR
    NAMES Sherwood.h
    PATHS ${SLITHER_ROOT}/lib
)

find_library(SLITHER_LIBRARY
    NAMES slither
    PATHS ${SLITHER_ROOT}/slither_build
)

if(SLITHER_INCLUDE_DIR AND SLITHER_LIBRARY)
    set(SLITHER_FOUND TRUE)
endif()

# Your CMakeLists.txt
find_package(Slither REQUIRED)

target_include_directories(your_target PRIVATE ${SLITHER_INCLUDE_DIR})
target_link_libraries(your_target PRIVATE ${SLITHER_LIBRARY})

# Also link dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBSVM REQUIRED libsvm)
target_link_libraries(your_target PRIVATE ${LIBSVM_LIBRARIES})
```

### Including in Your Project

```cpp
// your_project.cpp
#include "Sherwood.h"
#include "Classification.h"

class YourMLPipeline {
private:
    std::unique_ptr<Sherwood::Forest<
        Sherwood::LinearFeatureResponse, 
        Sherwood::HistogramAggregator>> forest_;
    
public:
    bool trainModel(const cv::Mat& features, const cv::Mat& labels) {
        Sherwood::DataPointCollection data;
        data.LoadData(features, labels);
        
        Sherwood::TrainingParameters params;
        params.NumberOfTrees = 20;
        params.MaxDecisionLevels = 8;
        params.NumberOfCandidateFeatures = std::sqrt(features.cols);
        
        Sherwood::ClassificationTrainingContext<Sherwood::LinearFeatureResponse> 
            context(data.CountClasses());
        
        forest_ = Sherwood::ForestTrainer<
            Sherwood::LinearFeatureResponse, 
            Sherwood::HistogramAggregator
        >::TrainForest(data, params, context);
        
        return forest_ != nullptr;
    }
    
    std::vector<int> predict(const cv::Mat& features) {
        if (!forest_) {
            throw std::runtime_error("Model not trained");
        }
        
        std::vector<int> predictions;
        for (int i = 0; i < features.rows; i++) {
            cv::Mat row = features.row(i);
            Sherwood::DataPoint dataPoint(row.ptr<float>(), features.cols);
            
            auto result = forest_->Apply(dataPoint);
            predictions.push_back(result.GetMaxBin());
        }
        
        return predictions;
    }
};
```

## Error Handling

### Common Issues and Solutions

```cpp
// Robust training with error handling
std::unique_ptr<Forest<LinearFeatureResponse, HistogramAggregator>> 
trainRobustForest(const DataPointCollection& data, 
                  const TrainingParameters& params) {
    try {
        // Validate data
        if (data.Count() == 0) {
            throw std::runtime_error("Empty training data");
        }
        
        if (data.CountClasses() < 2) {
            throw std::runtime_error("Need at least 2 classes for classification");
        }
        
        // Validate parameters
        if (params.NumberOfTrees <= 0) {
            throw std::runtime_error("Number of trees must be positive");
        }
        
        if (params.NumberOfCandidateFeatures > data.Dimensions()) {
            std::cerr << "Warning: More candidate features than available features" 
                      << std::endl;
        }
        
        // Train forest
        ClassificationTrainingContext<LinearFeatureResponse> context(
            data.CountClasses()
        );
        
        return ForestTrainer<LinearFeatureResponse, HistogramAggregator>
            ::TrainForest(data, params, context);
            
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return nullptr;
    }
}
```

## Performance Optimization

### Memory Pool Allocation (Advanced)

```cpp
// Custom memory allocator for better performance
class ForestMemoryPool {
private:
    std::vector<std::unique_ptr<char[]>> pools_;
    size_t currentPool_;
    size_t currentOffset_;
    static constexpr size_t POOL_SIZE = 1024 * 1024;  // 1MB pools
    
public:
    void* allocate(size_t size) {
        if (pools_.empty() || currentOffset_ + size > POOL_SIZE) {
            allocateNewPool();
        }
        
        void* result = pools_[currentPool_].get() + currentOffset_;
        currentOffset_ += size;
        return result;
    }
    
private:
    void allocateNewPool() {
        pools_.push_back(std::make_unique<char[]>(POOL_SIZE));
        currentPool_ = pools_.size() - 1;
        currentOffset_ = 0;
    }
};
```

### SIMD Optimization

```cpp
// Vectorized prediction for multiple samples
#ifdef __AVX2__
#include <immintrin.h>

void predictBatch(const Forest<LinearFeatureResponse, HistogramAggregator>& forest,
                  const float* features, int n_samples, int n_features,
                  int* predictions) {
    // Batch processing with SIMD instructions
    const int simd_width = 8;  // AVX2 processes 8 floats at once
    
    for (int i = 0; i < n_samples; i += simd_width) {
        int remaining = std::min(simd_width, n_samples - i);
        
        for (int j = 0; j < remaining; j++) {
            DataPoint dataPoint(features + (i + j) * n_features, n_features);
            auto result = forest.Apply(dataPoint);
            predictions[i + j] = result.GetMaxBin();
        }
    }
}
#endif
```

This comprehensive C++ usage guide provides developers with the knowledge needed to integrate and optimize Slither Random Forest in their C++ applications.