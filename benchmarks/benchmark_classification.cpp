#include <slither/Sherwood.h>
#include <slither/Classification.h>
#include <slither/DataPointCollection.h>
#include <chrono>
#include <iostream>
#include <random>

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main() {
    std::cout << "Slither Random Forest - Classification Benchmark\n";
    std::cout << "===============================================\n";
    
    // Generate synthetic data for benchmarking
    const int n_samples = 1000;
    const int n_features = 10;
    const int n_classes = 3;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Create training data
    DataPointCollection trainingData;
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> features(n_features);
        for (int j = 0; j < n_features; ++j) {
            features[j] = dis(gen);
        }
        
        // Simple classification rule for synthetic data
        int label = static_cast<int>(std::abs(features[0] + features[1])) % n_classes;
        
        trainingData.AddDataPoint(DataPoint(features, label));
    }
    
    // Training parameters
    TrainingParameters trainingParameters;
    trainingParameters.NumberOfTrees = 10;
    trainingParameters.NumberOfCandidateFeatures = static_cast<int>(std::sqrt(n_features));
    trainingParameters.NumberOfCandidateThresholds = 10;
    trainingParameters.MaxDecisionLevels = 8;
    trainingParameters.Verbose = true;
    
    // Benchmark training
    auto start = std::chrono::high_resolution_clock::now();
    
    auto forest = ForestTrainer<LinearFeatureResponse, HistogramAggregator>::TrainForest(
        std::mt19937(),
        trainingParameters,
        ClassificationTrainingContext(n_features, n_classes),
        trainingData
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training completed in " << training_duration.count() << " ms\n";
    
    // Benchmark prediction
    const int n_test_samples = 100;
    std::vector<DataPoint> testData;
    for (int i = 0; i < n_test_samples; ++i) {
        std::vector<float> features(n_features);
        for (int j = 0; j < n_features; ++j) {
            features[j] = dis(gen);
        }
        testData.push_back(DataPoint(features, 0)); // Label doesn't matter for prediction
    }
    
    start = std::chrono::high_resolution_clock::now();
    
    for (const auto& testPoint : testData) {
        std::vector<float> distribution(n_classes);
        forest->Apply(testPoint, distribution);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto prediction_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Prediction for " << n_test_samples << " samples completed in " 
              << prediction_duration.count() << " μs\n";
    std::cout << "Average prediction time: " 
              << prediction_duration.count() / n_test_samples << " μs per sample\n";
    
    return 0;
}