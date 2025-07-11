# C++ API Reference

Complete reference for the Slither Random Forest C++ library.

## Core Classes

### Forest<F, S>

Main container class for the Random Forest ensemble.

```cpp
template<class F, class S>
class Forest
```

**Template Parameters:**
- `F`: Feature response function type (e.g., `LinearFeatureResponse`)
- `S`: Statistics aggregator type (e.g., `HistogramAggregator`)

#### Public Methods

##### Constructor

```cpp
Forest()
```
Creates an empty forest.

##### Tree Management

```cpp
void AddTree(std::unique_ptr<Tree<F, S>> tree)
```
Adds a tree to the forest.

**Parameters:**
- `tree`: Unique pointer to the tree to add

```cpp
size_t TreeCount() const
```
Returns the number of trees in the forest.

**Returns:** Number of trees

##### Prediction

```cpp
S Apply(const DataPoint& dataPoint) const
```
Applies the entire forest to classify a data point.

**Parameters:**
- `dataPoint`: Input data point to classify

**Returns:** Aggregated statistics from all trees

##### Serialization

```cpp
void SerializeJson(const std::string& path) const
```
Saves the forest to a JSON file.

**Parameters:**
- `path`: File path to save to

**Throws:** `std::runtime_error` if file cannot be written

```cpp
void DeserializeJson(const std::string& path)
```
Loads a forest from a JSON file.

**Parameters:**
- `path`: File path to load from

**Throws:** `std::runtime_error` if file cannot be read or format is invalid

#### Usage Example

```cpp
#include "Forest.h"
#include "Classification.h"

// Create and train forest
ClassificationTrainingContext<LinearFeatureResponse> context(3);
auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
    ::TrainForest(data, params, context);

// Make predictions
auto prediction = forest->Apply(testDataPoint);
int predictedClass = prediction.GetMaxBin();

// Save model
forest->SerializeJson("model.json");
```

---

### Tree<F, S>

Individual decision tree within the forest.

```cpp
template<class F, class S>
class Tree
```

#### Public Methods

##### Constructor

```cpp
Tree(std::unique_ptr<Node<F, S>> root)
```
Creates a tree with the given root node.

##### Prediction

```cpp
S Apply(const DataPoint& dataPoint) const
```
Applies the tree to classify a data point.

**Returns:** Statistics from the reached leaf node

##### Tree Access

```cpp
const Node<F, S>* GetRoot() const
```
Returns the root node of the tree.

##### Serialization

```cpp
nlohmann::json SerializeJson() const
```
Serializes the tree to JSON format.

```cpp
void DeserializeJson(const nlohmann::json& json)
```
Deserializes the tree from JSON format.

---

### Node<F, S>

Individual node in a decision tree.

```cpp
template<class F, class S>
class Node
```

#### Public Methods

##### Node Type

```cpp
bool IsLeaf() const
```
Checks if this is a leaf node.

**Returns:** `true` if leaf node, `false` if internal node

##### Tree Navigation

```cpp
const Node<F, S>* GetLeftChild() const
const Node<F, S>* GetRightChild() const
```
Get pointers to child nodes (null for leaf nodes).

##### Prediction

```cpp
S Apply(const DataPoint& dataPoint) const
```
Applies the node's decision function to a data point.

For internal nodes, routes to appropriate child.
For leaf nodes, returns training statistics.

##### Statistics Access

```cpp
const S& GetTrainingDataStatistics() const
```
Gets the training statistics (leaf nodes only).

**Returns:** Reference to statistics aggregator

#### Node Creation

```cpp
// Create leaf node
template<class F, class S>
std::unique_ptr<Node<F, S>> Node<F, S>::CreateLeafNode(const S& statistics)

// Create internal node
template<class F, class S>
std::unique_ptr<Node<F, S>> Node<F, S>::CreateInternalNode(
    const F& featureResponse,
    std::unique_ptr<Node<F, S>> leftChild,
    std::unique_ptr<Node<F, S>> rightChild
)
```

---

## Training Classes

### ForestTrainer<F, S>

Static class for training Random Forest models.

```cpp
template<class F, class S>
class ForestTrainer
```

#### Static Methods

```cpp
static std::unique_ptr<Forest<F, S>> TrainForest(
    const DataPointCollection& trainingData,
    const TrainingParameters& trainingParameters,
    ITrainingContext<F, S>& trainingContext
)
```
Trains a complete Random Forest.

**Parameters:**
- `trainingData`: Training dataset
- `trainingParameters`: Training configuration
- `trainingContext`: Training context (provides feature response and aggregator types)

**Returns:** Trained forest

**Example:**
```cpp
TrainingParameters params;
params.NumberOfTrees = 20;
params.MaxDecisionLevels = 8;

ClassificationTrainingContext<LinearFeatureResponse> context(numClasses);
auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
    ::TrainForest(data, params, context);
```

---

### TrainingParameters

Configuration class for training parameters.

```cpp
class TrainingParameters
```

#### Public Members

##### Forest Structure
```cpp
int NumberOfTrees = 10;           // Number of trees in forest
int MaxDecisionLevels = 5;        // Maximum tree depth
```

##### Feature Selection
```cpp
int NumberOfCandidateFeatures = 10;     // Features to consider per split
int NumberOfCandidateThresholds = 10;   // Threshold candidates per split
```

##### SVM Parameters
```cpp
double SvmC = 1.0;               // SVM regularization parameter
double SvmGamma = 1.0;           // SVM gamma parameter (for RBF kernels)
```

##### Stopping Criteria
```cpp
double MinInformationGain = 0.0;  // Minimum information gain for split
int MinSamplesPerNode = 2;        // Minimum samples required to split
```

##### Miscellaneous
```cpp
bool Verbose = false;            // Print training progress
int RandomSeed = 0;              // Random seed for reproducibility
```

#### Example Configuration

```cpp
TrainingParameters params;
params.NumberOfTrees = 50;
params.MaxDecisionLevels = 12;
params.NumberOfCandidateFeatures = 20;
params.SvmC = 0.5;
params.Verbose = true;
```

---

## Data Management

### DataPointCollection

Main data container for training and testing data.

```cpp
class DataPointCollection
```

#### Public Methods

##### Data Loading

```cpp
void LoadData(const cv::Mat& features, const cv::Mat& labels)
```
Loads data from OpenCV matrices.

**Parameters:**
- `features`: Feature matrix (n_samples × n_features)
- `labels`: Label vector (n_samples × 1)

```cpp
void LoadFromFile(const std::string& filepath)
```
Loads data from tab-separated text file.

**File Format:**
```
feature1    feature2    feature3    label
0.5         -0.3        1.2         0
-0.1        0.8         -0.5        1
...
```

##### Data Access

```cpp
DataPoint GetDataPoint(int index) const
```
Gets a data point by index.

```cpp
int GetIntegerLabel(int index) const
```
Gets the class label for a sample.

##### Data Properties

```cpp
int Count() const              // Number of samples
int Dimensions() const         // Number of features
int CountClasses() const       // Number of unique classes
```

##### Data Subsets

```cpp
std::vector<int> GetIndicesWithLabel(int label) const
```
Gets indices of all samples with given label.

```cpp
DataPointCollection ExtractSubset(const std::vector<int>& indices) const
```
Creates a new collection with subset of samples.

#### Usage Example

```cpp
// Load from OpenCV matrices
cv::Mat features(100, 10, CV_32F);
cv::Mat labels(100, 1, CV_32S);
// ... populate matrices ...

DataPointCollection data;
data.LoadData(features, labels);

// Access data
for (int i = 0; i < data.Count(); i++) {
    auto point = data.GetDataPoint(i);
    int label = data.GetIntegerLabel(i);
    
    // Use data point
    for (int j = 0; j < point.Count(); j++) {
        float feature = point[j];
    }
}
```

---

### DataPoint

Lightweight wrapper for individual data points.

```cpp
class DataPoint
```

#### Public Methods

##### Constructor

```cpp
DataPoint(const float* features, int dimensions)
```
Creates a data point from feature array.

##### Data Access

```cpp
float operator[](int index) const
```
Access feature by index.

```cpp
int Count() const
```
Get number of features.

##### Iteration Support

```cpp
const float* begin() const
const float* end() const
```
STL-compatible iterators for range-based loops.

#### Usage Example

```cpp
// Create data point
float features[] = {1.0f, -0.5f, 2.3f};
DataPoint point(features, 3);

// Access features
float first = point[0];
int numFeatures = point.Count();

// Range-based loop
for (float feature : point) {
    std::cout << feature << " ";
}
```

---

## Feature Response Functions

### LinearFeatureResponse

Default feature response using linear SVM classifiers.

```cpp
class LinearFeatureResponse
```

#### Public Methods

##### Training

```cpp
static LinearFeatureResponse CreateRandom(
    std::mt19937& random,
    const TrainingParameters& parameters,
    const DataPointCollection& data
)
```
Creates and trains a linear feature response.

##### Prediction

```cpp
float GetResponse(const DataPoint& dataPoint) const
```
Applies the linear classifier to a data point.

**Returns:** SVM decision value (positive or negative)

##### Serialization

```cpp
void Serialize(std::ostream& stream) const
void Deserialize(std::istream& stream)
```

#### Implementation Details

The linear response computes:
```
response = bias + Σ(weight[i] × feature[selected_features[i]])
```

Where:
- `selected_features` are randomly chosen features
- `weight` and `bias` are learned from SVM training

---

## Statistics Aggregators

### HistogramAggregator

Classification statistics aggregator maintaining class histograms.

```cpp
class HistogramAggregator
```

#### Public Methods

##### Constructor

```cpp
HistogramAggregator(int binCount)
```
Creates aggregator for given number of classes.

##### Data Aggregation

```cpp
void Aggregate(const DataPoint& dataPoint, int label)
```
Adds a sample to the statistics.

```cpp
HistogramAggregator& operator+=(const HistogramAggregator& other)
```
Combines statistics from another aggregator.

##### Prediction

```cpp
int GetMaxBin() const
```
Gets the most frequent class.

```cpp
std::vector<double> GetProbabilities() const
```
Gets class probability distribution.

##### Information Theory

```cpp
double CalculateEntropy() const
```
Calculates Shannon entropy of the distribution.

#### Usage Example

```cpp
// Create aggregator for 3 classes
HistogramAggregator stats(3);

// Add samples
DataPoint point1(features1, 10);
stats.Aggregate(point1, 0);  // Class 0

DataPoint point2(features2, 10);
stats.Aggregate(point2, 1);  // Class 1

// Get prediction
int mostFrequentClass = stats.GetMaxBin();
auto probabilities = stats.GetProbabilities();
double entropy = stats.CalculateEntropy();
```

---

## Training Contexts

### ClassificationTrainingContext

Training context for classification problems.

```cpp
template<class F>
class ClassificationTrainingContext : public ITrainingContext<F, HistogramAggregator>
```

#### Constructor

```cpp
ClassificationTrainingContext(int classCount)
```
Creates context for given number of classes.

#### Methods

```cpp
F CreateRandomFeatureResponse(
    std::mt19937& random,
    const TrainingParameters& parameters,
    const DataPointCollection& data
) const override
```

```cpp
HistogramAggregator CreateStatisticsAggregator() const override
```

#### Usage

```cpp
// For 3-class classification problem
ClassificationTrainingContext<LinearFeatureResponse> context(3);

// Use in training
auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
    ::TrainForest(data, params, context);
```

---

## Utility Functions

### Information Gain Calculation

```cpp
namespace Sherwood {
    double CalculateInformationGain(
        const HistogramAggregator& parent,
        const HistogramAggregator& leftChild,
        const HistogramAggregator& rightChild
    );
}
```

Calculates information gain for a potential split:
```
IG = H(parent) - [N_left/N × H(left) + N_right/N × H(right)]
```

### Random Number Generation

```cpp
namespace Sherwood {
    class Random {
    public:
        static void Seed(unsigned int seed);
        static int NextInt(int maxValue);
        static double NextDouble();
        static std::vector<int> SampleWithoutReplacement(
            int populationSize, 
            int sampleSize
        );
    };
}
```

**Note:** Modern C++ applications should use `<random>` library instead.

---

## Example: Complete Training Pipeline

```cpp
#include "Sherwood.h"
#include "Classification.h"

int main() {
    try {
        // 1. Load training data
        DataPointCollection trainingData;
        trainingData.LoadFromFile("training_data.txt");
        
        // 2. Configure training parameters
        TrainingParameters params;
        params.NumberOfTrees = 20;
        params.MaxDecisionLevels = 8;
        params.NumberOfCandidateFeatures = 
            std::max(1, static_cast<int>(std::sqrt(trainingData.Dimensions())));
        params.SvmC = 1.0;
        params.Verbose = true;
        
        // 3. Create training context
        ClassificationTrainingContext<LinearFeatureResponse> context(
            trainingData.CountClasses()
        );
        
        // 4. Train forest
        std::cout << "Training Random Forest..." << std::endl;
        auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>
            ::TrainForest(trainingData, params, context);
        
        // 5. Save trained model
        forest->SerializeJson("trained_model.json");
        std::cout << "Model saved to trained_model.json" << std::endl;
        
        // 6. Test on training data
        int correct = 0;
        for (int i = 0; i < trainingData.Count(); i++) {
            auto dataPoint = trainingData.GetDataPoint(i);
            auto prediction = forest->Apply(dataPoint);
            
            int predictedClass = prediction.GetMaxBin();
            int actualClass = trainingData.GetIntegerLabel(i);
            
            if (predictedClass == actualClass) correct++;
        }
        
        double accuracy = static_cast<double>(correct) / trainingData.Count();
        std::cout << "Training accuracy: " << accuracy << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

---

## Error Handling

### Common Exceptions

- `std::runtime_error`: File I/O errors, invalid model format
- `std::invalid_argument`: Invalid parameter values
- `std::bad_alloc`: Memory allocation failures
- `cv::Exception`: OpenCV-related errors

### Best Practices

```cpp
// Validate parameters before training
void validateParameters(const TrainingParameters& params) {
    if (params.NumberOfTrees <= 0) {
        throw std::invalid_argument("NumberOfTrees must be positive");
    }
    
    if (params.MaxDecisionLevels <= 0) {
        throw std::invalid_argument("MaxDecisionLevels must be positive");
    }
    
    if (params.SvmC <= 0) {
        throw std::invalid_argument("SvmC must be positive");
    }
}

// Use RAII for resource management
auto forest = std::make_unique<Forest<LinearFeatureResponse, HistogramAggregator>>();
// Automatically cleaned up when out of scope
```

This C++ API reference provides comprehensive documentation for all public interfaces in the Slither Random Forest library.