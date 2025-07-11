"""
Basic Usage Example for Slither Random Forest
==============================================

This example demonstrates the basic usage of the modernized Slither
Random Forest classifier with scikit-learn compatible API.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the modernized Slither classifier
try:
    from slither import SlitherClassifier
except ImportError:
    print("Slither not installed. Please install with: pip install -e .")
    exit(1)


def main():
    """Run the basic usage example."""
    print("🌟 Slither Random Forest - Basic Usage Example")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample classification data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Create and configure the classifier
    print("\n🌳 Creating Slither Random Forest classifier...")
    clf = SlitherClassifier(
        n_estimators=10,      # Use fewer trees for faster demo
        max_depth=8,          # Moderate depth
        n_candidate_features=10,
        n_candidate_thresholds=10,
        svm_c=0.5,
        random_state=42,
        verbose=False         # Set to True to see training progress
    )
    
    print(f"   Parameters: {clf.get_params()}")
    
    # Train the classifier
    print("\n🔧 Training the classifier...")
    clf.fit(X_train, y_train)
    
    print(f"   ✅ Training completed!")
    print(f"   Classes found: {clf.classes_}")
    print(f"   Number of classes: {clf.n_classes_}")
    print(f"   Features used: {clf.n_features_in_}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Results:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Predictions shape: {y_pred.shape}")
    print(f"   Probabilities shape: {y_proba.shape}")
    
    # Show detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Demonstrate the score method
    score = clf.score(X_test, y_test)
    print(f"📊 Classifier score: {score:.3f}")
    
    # Show a few example predictions
    print(f"\n🎯 Example predictions (first 10 samples):")
    print(f"   True labels:      {y_test[:10]}")
    print(f"   Predicted labels: {y_pred[:10]}")
    print(f"   Max probabilities: {y_proba[:10].max(axis=1).round(3)}")
    
    # Demonstrate model saving/loading
    print(f"\n💾 Testing model serialization...")
    model_path = "example_model.json"
    
    try:
        clf.save_model(model_path)
        print(f"   ✅ Model saved to {model_path}")
        
        # Create a new classifier and load the model
        clf_loaded = SlitherClassifier()
        clf_loaded.load_model(model_path)
        print(f"   ✅ Model loaded successfully")
        
        # Verify the loaded model works
        y_pred_loaded = clf_loaded.predict(X_test[:5])
        print(f"   ✅ Loaded model predictions: {y_pred_loaded}")
        
        # Clean up
        import os
        os.remove(model_path)
        print(f"   🧹 Cleaned up {model_path}")
        
    except Exception as e:
        print(f"   ❌ Model serialization failed: {e}")
    
    print(f"\n🎉 Example completed successfully!")
    print(f"    The modernized Slither API is fully scikit-learn compatible!")


if __name__ == "__main__":
    main()