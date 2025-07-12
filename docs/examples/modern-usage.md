# Modern Usage Examples

This page demonstrates how to use Slither Random Forest with the new modern project structure and APIs.

## Python Examples (Scikit-learn Compatible)

### Basic Classification

```python
from slither import SlitherClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train classifier
clf = SlitherClassifier(
    n_estimators=20,
    max_depth=8,
    n_candidate_features='sqrt',  # sqrt(n_features)
    svm_c=0.5,
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbose=True
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
```

### Computer Vision Example

```python
from slither import SlitherClassifier
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load face dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Configure for high-dimensional image data
clf = SlitherClassifier(
    n_estimators=50,
    max_depth=12,
    n_candidate_features=100,  # Higher for image features
    svm_c=0.1,  # Lower C for high-dimensional data
    random_state=42,
    n_jobs=4,
    verbose=True
)

# Train on image features
print("Training on {} image features...".format(X_train.shape[1]))
clf.fit(X_train, y_train)

# Evaluate performance
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.3f}")
print(f"Test Accuracy: {test_score:.3f}")

# Analyze predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Show confidence for each prediction
for i in range(min(10, len(y_test))):
    confidence = np.max(y_proba[i])
    print(f"Sample {i}: True={y_test[i]}, Pred={y_pred[i]}, Confidence={confidence:.3f}")
```

### Model Persistence

```python
from slither import SlitherClassifier
import pickle
import numpy as np

# Train a model
clf = SlitherClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# Method 1: Using pickle (Python standard)
with open('slither_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load the model
with open('slither_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

# Verify it works
assert np.allclose(clf.predict_proba(X_test), clf_loaded.predict_proba(X_test))

# Method 2: Using native save/load (if implemented)
clf.save_model('slither_model.json')
clf_loaded2 = SlitherClassifier()
clf_loaded2.load_model('slither_model.json')
```

## C++ Examples (Modern API)

### Basic Forest Training

```cpp
#include <slither/Sherwood.h>
#include <slither/Classification.h>
#include <slither/ForestTrainer.h>
#include <slither/DataPointCollection.h>
#include <iostream>
#include <random>
#include <vector>

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main() {
    // Modern random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Dataset parameters
    const int n_samples = 1000;
    const int n_features = 10;
    const int n_classes = 3;
    
    // Generate synthetic dataset
    DataPointCollection trainingData;
    std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> features(n_features);
        for (int j = 0; j < n_features; ++j) {
            features[j] = feature_dist(gen);
        }
        
        // Simple classification rule
        int label = static_cast<int>(std::abs(features[0] + features[1] * 0.5f)) % n_classes;
        trainingData.AddDataPoint(DataPoint(features, label));
    }
    
    // Configure training parameters
    TrainingParameters params;
    params.NumberOfTrees = 20;
    params.MaxDecisionLevels = 8;
    params.NumberOfCandidateFeatures = static_cast<int>(std::sqrt(n_features));
    params.NumberOfCandidateThresholds = 10;
    params.Verbose = true;
    
    std::cout << "Training forest with " << params.NumberOfTrees << " trees..." << std::endl;
    
    // Train the forest
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
        gen,
        params,
        ClassificationTrainingContext(n_features, n_classes),
        trainingData
    );
    
    std::cout << "Training completed!" << std::endl;
    
    // Test predictions
    std::vector<float> test_features = {1.0f, -0.5f, 0.8f, -1.2f, 0.3f, 
                                       1.5f, -0.8f, 0.2f, -0.1f, 0.9f};
    DataPoint test_point(test_features, 0);  // Label doesn't matter for prediction
    
    std::vector<float> probabilities(n_classes);
    forest->Apply(test_point, probabilities);
    
    // Find predicted class
    int predicted_class = std::max_element(probabilities.begin(), probabilities.end()) 
                         - probabilities.begin();
    
    std::cout << "Test prediction:" << std::endl;
    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Class probabilities: ";
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << "P(class " << i << ") = " << probabilities[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### JSON Serialization Example

```cpp
#include <slither/Forest.h>
#include <slither/ForestTrainer.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

// Train a forest (same as above)
auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
    gen, params, context, trainingData
);

// Serialize to JSON
nlohmann::json forest_json;
forest->Serialize(forest_json);

// Save to file with pretty printing
std::ofstream file("my_forest.json");
file << forest_json.dump(4);  // 4-space indentation
file.close();

std::cout << "Forest saved to my_forest.json" << std::endl;

// Load from file
std::ifstream load_file("my_forest.json");
nlohmann::json loaded_json;
load_file >> loaded_json;

// Create new forest and deserialize
auto loaded_forest = std::make_unique<Forest<LinearFeatureResponse, HistogramAggregator>>();
loaded_forest->Deserialize(loaded_json);

// Verify it works
std::vector<float> original_result(n_classes);
std::vector<float> loaded_result(n_classes);

forest->Apply(test_point, original_result);
loaded_forest->Apply(test_point, loaded_result);

bool results_match = true;
for (size_t i = 0; i < n_classes; ++i) {
    if (std::abs(original_result[i] - loaded_result[i]) > 1e-6) {
        results_match = false;
        break;
    }
}

std::cout << "Serialization test: " << (results_match ? "PASSED" : "FAILED") << std::endl;
```

### Parallel Training Example

```cpp
#include <slither/ParallelForestTrainer.h>
#include <chrono>

// Large dataset parameters
const int large_n_samples = 10000;
const int large_n_features = 50;

// Generate larger dataset
DataPointCollection large_training_data;
// ... populate with data ...

// Configure for parallel training
TrainingParameters parallel_params;
parallel_params.NumberOfTrees = 100;
parallel_params.MaxDecisionLevels = 12;
parallel_params.NumberOfCandidateFeatures = static_cast<int>(std::sqrt(large_n_features));
parallel_params.Verbose = true;

// Benchmark training time
auto start_time = std::chrono::high_resolution_clock::now();

// Use parallel trainer for better performance
auto parallel_forest = ParallelForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
    gen,
    parallel_params,
    ClassificationTrainingContext(large_n_features, n_classes),
    large_training_data
);

auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

std::cout << "Parallel training of " << parallel_params.NumberOfTrees 
          << " trees completed in " << duration.count() << " ms" << std::endl;
```

## Performance Benchmarking

### Python Benchmarking

```python
import time
import numpy as np
from slither import SlitherClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate benchmark dataset
X, y = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=30,
    n_classes=5,
    random_state=42
)

# Slither Random Forest
print("Benchmarking Slither Random Forest...")
start_time = time.time()
slither_clf = SlitherClassifier(n_estimators=50, n_jobs=-1)
slither_clf.fit(X, y)
slither_train_time = time.time() - start_time

start_time = time.time()
slither_pred = slither_clf.predict(X)
slither_pred_time = time.time() - start_time

# Scikit-learn Random Forest
print("Benchmarking scikit-learn Random Forest...")
start_time = time.time()
sklearn_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
sklearn_clf.fit(X, y)
sklearn_train_time = time.time() - start_time

start_time = time.time()
sklearn_pred = sklearn_clf.predict(X)
sklearn_pred_time = time.time() - start_time

# Compare results
slither_accuracy = slither_clf.score(X, y)
sklearn_accuracy = sklearn_clf.score(X, y)

print(f"Results:")
print(f"Slither RF - Training: {slither_train_time:.3f}s, Prediction: {slither_pred_time:.3f}s, Accuracy: {slither_accuracy:.3f}")
print(f"Sklearn RF - Training: {sklearn_train_time:.3f}s, Prediction: {sklearn_pred_time:.3f}s, Accuracy: {sklearn_accuracy:.3f}")
print(f"Training speedup: {sklearn_train_time/slither_train_time:.2f}x")
print(f"Prediction speedup: {sklearn_pred_time/slither_pred_time:.2f}x")
```

### C++ Benchmarking

```cpp
#include <slither/ForestTrainer.h>
#include <chrono>
#include <iostream>

void benchmark_forest_sizes() {
    std::vector<int> tree_counts = {10, 20, 50, 100};
    const int n_samples = 2000;
    const int n_features = 20;
    const int n_classes = 3;
    
    // Generate consistent dataset
    std::mt19937 gen(42);
    DataPointCollection data;
    // ... populate data ...
    
    for (int n_trees : tree_counts) {
        TrainingParameters params;
        params.NumberOfTrees = n_trees;
        params.MaxDecisionLevels = 8;
        params.Verbose = false;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
            gen, params, ClassificationTrainingContext(n_features, n_classes), data
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Trees: " << n_trees << ", Training time: " << duration.count() << " ms" << std::endl;
    }
}
```

## Integration Examples

### Using with OpenCV

```cpp
#include <slither/Sherwood.h>
#include <opencv2/opencv.hpp>
#include <vector>

// Convert OpenCV Mat to DataPoint
DataPoint matToDataPoint(const cv::Mat& features, int label = 0) {
    std::vector<float> feature_vector;
    
    if (features.type() == CV_32F) {
        feature_vector.assign(features.ptr<float>(), features.ptr<float>() + features.total());
    } else {
        cv::Mat float_features;
        features.convertTo(float_features, CV_32F);
        feature_vector.assign(float_features.ptr<float>(), float_features.ptr<float>() + float_features.total());
    }
    
    return DataPoint(feature_vector, label);
}

// Example usage with image patches
void trainOnImagePatches() {
    cv::Mat image = cv::imread("training_image.jpg");
    DataPointCollection training_data;
    
    // Extract patches and convert to DataPoints
    const int patch_size = 32;
    for (int y = 0; y < image.rows - patch_size; y += patch_size) {
        for (int x = 0; x < image.cols - patch_size; x += patch_size) {
            cv::Rect patch_rect(x, y, patch_size, patch_size);
            cv::Mat patch = image(patch_rect);
            
            // Flatten patch to feature vector
            cv::Mat flat_patch = patch.reshape(1, 1);
            
            // Assuming you have ground truth labels
            int label = getGroundTruthLabel(x, y);
            
            training_data.AddDataPoint(matToDataPoint(flat_patch, label));
        }
    }
    
    // Train forest on image patches
    // ... training code ...
}
```

This comprehensive example collection demonstrates the modern APIs and best practices for using Slither Random Forest in both Python and C++ environments.