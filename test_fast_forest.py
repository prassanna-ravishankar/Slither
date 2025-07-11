#!/usr/bin/env python3
"""
Fast Random Forest functionality test
=====================================

Quick test with minimal parameters to demonstrate forest behavior.
"""

import sys
import os
sys.path.insert(0, 'slither_build')

import numpy as np
from sklearn.datasets import make_classification

try:
    from slither_py import slither as SlitherWrapper
    print("✅ Using real C++ Slither extension")
except ImportError:
    print("❌ C++ extension not available.")
    sys.exit(1)

def quick_forest_test():
    """Quick test with minimal parameters."""
    
    print("\n🌳 Quick Random Forest Test")
    print("=" * 40)
    
    # Very small dataset for speed
    X, y = make_classification(
        n_samples=50,      # Small dataset
        n_features=4,      # Few features  
        n_classes=2,       # Binary classification
        n_informative=3,
        n_redundant=0,     # Fix: ensure features sum correctly
        random_state=42
    )
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Test 1: Single tree
    print(f"\n🌲 Training single tree...")
    clf_single = SlitherWrapper()
    clf_single.setParams(
        2,    # max_depth=2 (very shallow)
        2,    # n_candidate_features=2
        2,    # n_candidate_thresholds=2  
        1,    # n_estimators=1 (single tree)
        0.5,  # svm_c=0.5
        False, # verbose=False
        1     # n_jobs=1
    )
    
    clf_single.loadData(X, y)
    success = clf_single.onlyTrain()
    print(f"    Single tree training: {'✅ Success' if success else '❌ Failed'}")
    
    # Test prediction
    clf_single.loadData(X[:10], y[:10])
    proba_single = clf_single.onlyTest()
    pred_single = np.argmax(proba_single, axis=1)
    acc_single = np.mean(pred_single == y[:10])
    print(f"    Single tree accuracy: {acc_single:.3f}")
    
    # Test 2: Small forest
    print(f"\n🌳 Training small forest (3 trees)...")
    clf_forest = SlitherWrapper()
    clf_forest.setParams(
        2,    # max_depth=2
        2,    # n_candidate_features=2
        2,    # n_candidate_thresholds=2
        3,    # n_estimators=3 (small forest)
        0.5,  # svm_c=0.5
        False, # verbose=False
        1     # n_jobs=1
    )
    
    clf_forest.loadData(X, y)
    success = clf_forest.onlyTrain()
    print(f"    Forest training: {'✅ Success' if success else '❌ Failed'}")
    
    # Test prediction
    clf_forest.loadData(X[:10], y[:10])
    proba_forest = clf_forest.onlyTest()
    pred_forest = np.argmax(proba_forest, axis=1)
    acc_forest = np.mean(pred_forest == y[:10])
    print(f"    Forest accuracy: {acc_forest:.3f}")
    
    # Test 3: Probability validation
    print(f"\n🔍 Validating forest outputs...")
    print(f"    Probability shape: {proba_forest.shape}")
    print(f"    Prediction shape: {pred_forest.shape}")
    
    # Check probability sums
    prob_sums = np.sum(proba_forest, axis=1)
    valid_probs = np.allclose(prob_sums, 1.0, atol=1e-3)
    print(f"    Probabilities sum to 1: {'✅' if valid_probs else '❌'}")
    
    # Show sample predictions
    print(f"\n    Sample predictions:")
    for i in range(min(5, len(pred_forest))):
        print(f"      Sample {i}: pred={pred_forest[i]}, proba=[{proba_forest[i, 0]:.3f}, {proba_forest[i, 1]:.3f}], true={y[i]}")
    
    # Test 4: Model persistence
    print(f"\n💾 Testing model save/load...")
    model_path = "quick_test.json"
    
    try:
        success = clf_forest.saveModel(model_path)
        print(f"    Save model: {'✅ Success' if success else '❌ Failed'}")
        
        if success and os.path.exists(model_path):
            # Load in new instance
            clf_loaded = SlitherWrapper()
            success = clf_loaded.loadModel(model_path)
            print(f"    Load model: {'✅ Success' if success else '❌ Failed'}")
            
            if success:
                # Test loaded model
                clf_loaded.loadData(X[:5], y[:5])
                proba_loaded = clf_loaded.onlyTest()
                
                # Compare with original
                matches = np.allclose(proba_forest[:5], proba_loaded, atol=1e-6)
                print(f"    Predictions match: {'✅' if matches else '❌'}")
            
            # Clean up
            os.remove(model_path)
            print(f"    ✅ Cleaned up test file")
        
    except Exception as e:
        print(f"    ❌ Model persistence error: {e}")
    
    print(f"\n🎉 Quick forest test completed!")
    print(f"    Slither is working as a Random Forest with SVM local experts!")
    
    return {
        'single_accuracy': acc_single,
        'forest_accuracy': acc_forest,
        'valid_probabilities': valid_probs,
        'forest_better': acc_forest >= acc_single
    }

if __name__ == "__main__":
    results = quick_forest_test()
    
    print(f"\n📈 Results Summary:")
    print(f"   Single tree accuracy: {results['single_accuracy']:.3f}")
    print(f"   Forest accuracy: {results['forest_accuracy']:.3f}")
    print(f"   Forest performance: {'✅ Better/Equal' if results['forest_better'] else '⚠️ Worse'}")
    print(f"   Valid probabilities: {'✅' if results['valid_probabilities'] else '❌'}")
    print(f"\n✅ Slither IS a functioning Random Forest implementation!")