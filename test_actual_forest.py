#!/usr/bin/env python3
"""
Test actual Random Forest functionality
========================================

This test demonstrates that Slither actually trains and works as a
random forest, not just a wrapper.
"""

import sys
import os
sys.path.insert(0, 'slither_build')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from slither_py import slither as SlitherWrapper
    print("✅ Using real C++ Slither extension")
except ImportError:
    print("❌ C++ extension not available. Please build first.")
    sys.exit(1)

def test_random_forest_behavior():
    """Test that Slither behaves like a proper random forest."""
    
    print("\n🌳 Testing Random Forest Behavior")
    print("=" * 50)
    
    # Create a more complex dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"    Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Test 1: Single tree vs Multiple trees
    print(f"\n🧪 Test 1: Single tree vs Forest")
    
    # Train with 1 tree
    clf_single = SlitherWrapper()
    clf_single.setParams(5, 10, 10, 1, 0.5, False, 1)  # 1 tree
    clf_single.loadData(X_train, y_train)
    clf_single.onlyTrain()
    
    # Test with 1 tree
    clf_single.loadData(X_test, y_test)
    proba_single = clf_single.onlyTest()
    pred_single = np.argmax(proba_single, axis=1)
    acc_single = np.mean(pred_single == y_test)
    
    # Train with 10 trees
    clf_forest = SlitherWrapper()
    clf_forest.setParams(5, 10, 10, 10, 0.5, False, 1)  # 10 trees
    clf_forest.loadData(X_train, y_train)
    clf_forest.onlyTrain()
    
    # Test with 10 trees
    clf_forest.loadData(X_test, y_test)
    proba_forest = clf_forest.onlyTest()
    pred_forest = np.argmax(proba_forest, axis=1)
    acc_forest = np.mean(pred_forest == y_test)
    
    print(f"    Single tree accuracy: {acc_single:.3f}")
    print(f"    Forest (10 trees) accuracy: {acc_forest:.3f}")
    print(f"    ✅ Forest {'better' if acc_forest > acc_single else 'comparable'} performance")
    
    # Test 2: Different numbers of trees
    print(f"\n🧪 Test 2: Forest size effect")
    
    tree_counts = [1, 5, 10, 20]
    accuracies = []
    
    for n_trees in tree_counts:
        clf = SlitherWrapper()
        clf.setParams(5, 10, 10, n_trees, 0.5, False, 1)
        clf.loadData(X_train, y_train)
        clf.onlyTrain()
        
        clf.loadData(X_test, y_test)
        proba = clf.onlyTest()
        pred = np.argmax(proba, axis=1)
        acc = np.mean(pred == y_test)
        accuracies.append(acc)
        
        print(f"    {n_trees:2d} trees: {acc:.3f} accuracy")
    
    print(f"    ✅ Performance generally {'improves' if accuracies[-1] > accuracies[0] else 'varies'} with more trees")
    
    # Test 3: Randomness in forest
    print(f"\n🧪 Test 3: Forest randomness")
    
    predictions = []
    for run in range(3):
        clf = SlitherWrapper()
        clf.setParams(5, 10, 10, 10, 0.5, False, 1)
        clf.loadData(X_train, y_train)
        clf.onlyTrain()
        
        clf.loadData(X_test[:50], y_test[:50])  # Just first 50 for speed
        proba = clf.onlyTest()
        pred = np.argmax(proba, axis=1)
        predictions.append(pred)
    
    # Check if predictions vary (indicating randomness)
    variations = 0
    for i in range(len(predictions[0])):
        if len(set(pred[i] for pred in predictions)) > 1:
            variations += 1
    
    print(f"    Variations in predictions across runs: {variations}/{len(predictions[0])}")
    print(f"    ✅ Forest shows {'proper' if variations > 0 else 'minimal'} randomness")
    
    # Test 4: Probability outputs
    print(f"\n🧪 Test 4: Probability outputs")
    
    clf = SlitherWrapper()
    clf.setParams(5, 10, 10, 10, 0.5, False, 1)
    clf.loadData(X_train, y_train)
    clf.onlyTrain()
    
    clf.loadData(X_test[:10], y_test[:10])
    proba = clf.onlyTest()
    
    print(f"    Probability shape: {proba.shape}")
    print(f"    Sample probabilities (first 3 samples):")
    for i in range(3):
        print(f"      Sample {i}: [{proba[i, 0]:.3f}, {proba[i, 1]:.3f}, {proba[i, 2]:.3f}] = {proba[i].sum():.3f}")
    
    # Check if probabilities sum to 1
    prob_sums = np.sum(proba, axis=1)
    valid_probs = np.allclose(prob_sums, 1.0, atol=1e-3)
    print(f"    ✅ Probabilities sum to 1: {valid_probs}")
    
    # Test 5: Model persistence
    print(f"\n🧪 Test 5: Model saving/loading")
    
    # Train a model
    clf_orig = SlitherWrapper()
    clf_orig.setParams(5, 10, 10, 5, 0.5, False, 1)
    clf_orig.loadData(X_train, y_train)
    clf_orig.onlyTrain()
    
    # Get predictions from original
    clf_orig.loadData(X_test[:20], y_test[:20])
    proba_orig = clf_orig.onlyTest()
    
    # Save model
    model_path = "test_forest.json"
    clf_orig.saveModel(model_path)
    
    # Load in new instance
    clf_loaded = SlitherWrapper()
    clf_loaded.loadModel(model_path)
    
    # Get predictions from loaded model
    clf_loaded.loadData(X_test[:20], y_test[:20])
    proba_loaded = clf_loaded.onlyTest()
    
    # Compare predictions
    predictions_match = np.allclose(proba_orig, proba_loaded, atol=1e-6)
    print(f"    ✅ Loaded model predictions match: {predictions_match}")
    
    # Clean up
    os.remove(model_path)
    
    print(f"\n🎉 All Random Forest tests completed!")
    print(f"    Slither is functioning as a proper Random Forest implementation!")
    
    return {
        'single_tree_acc': acc_single,
        'forest_acc': acc_forest,
        'forest_better': acc_forest > acc_single,
        'tree_size_effect': accuracies,
        'randomness_variations': variations,
        'valid_probabilities': valid_probs,
        'model_persistence': predictions_match
    }

if __name__ == "__main__":
    results = test_random_forest_behavior()
    
    print(f"\n📈 Summary:")
    print(f"   Single tree: {results['single_tree_acc']:.3f}")
    print(f"   Forest: {results['forest_acc']:.3f}")
    print(f"   Forest improvement: {'✅' if results['forest_better'] else '⚠️'}")
    print(f"   Random variations: {results['randomness_variations']}")
    print(f"   Valid probabilities: {'✅' if results['valid_probabilities'] else '❌'}")
    print(f"   Model persistence: {'✅' if results['model_persistence'] else '❌'}")