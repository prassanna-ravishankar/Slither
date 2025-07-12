# Examples

Comprehensive examples showing different use cases for Slither Random Forest.

## Basic Classification

### Binary Classification

```python
from slither import SlitherClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Generate binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create and train classifier
clf = SlitherClassifier(
    n_estimators=25,
    max_depth=8,
    n_candidate_features=10,
    svm_c=0.5,
    verbose=True
)

print("Training binary classifier...")
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Show probability distributions
print(f"\nSample Predictions:")
for i in range(5):
    pred_class = y_pred[i]
    confidence = y_proba[i, pred_class]
    true_class = y_test[i]
    status = "✓" if pred_class == true_class else "✗"
    print(f"Sample {i}: pred={pred_class} (conf={confidence:.3f}), true={true_class} {status}")
```

### Multi-class Classification

```python
from slither import SlitherClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Generate multi-class dataset
X, y = make_classification(
    n_samples=1500,
    n_features=15,
    n_informative=12,
    n_redundant=3,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train classifier
clf = SlitherClassifier(
    n_estimators=30,
    max_depth=10,
    n_candidate_features=8,
    svm_c=1.0,
    verbose=False
)

print("Training multi-class classifier...")
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Analyze confidence scores
confidences = np.max(y_proba, axis=1)
print(f"\nConfidence Statistics:")
print(f"Mean confidence: {confidences.mean():.3f}")
print(f"Min confidence: {confidences.min():.3f}")
print(f"Max confidence: {confidences.max():.3f}")

# Show per-class accuracy
for class_id in range(len(clf.classes_)):
    class_mask = (y_test == class_id)
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
        print(f"Class {class_id} accuracy: {class_acc:.3f}")
```

## Computer Vision Examples

### Image Classification with Olivetti Faces

```python
from slither import SlitherClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
print("Loading Olivetti faces dataset...")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

print(f"Dataset info:")
print(f"  Samples: {X.shape[0]}")
print(f"  Features (pixels): {X.shape[1]}")
print(f"  Classes (people): {len(np.unique(y))}")
print(f"  Image shape: {faces.images.shape[1:]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Optional: Apply PCA for dimensionality reduction
use_pca = True
if use_pca:
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    X_train, X_test = X_train_pca, X_test_pca

# Create classifier optimized for faces
clf = SlitherClassifier(
    n_estimators=20,           # Moderate number for speed
    max_depth=8,               # Reasonable depth for faces
    n_candidate_features=50,   # Many features for high-dim data
    svm_c=0.5,                # Balanced regularization
    n_jobs=1,                  # Single thread for stability
    verbose=False
)

# Train the classifier
print(f"\nTraining Slither Random Forest on face data...")
print(f"Training set: {X_train.shape}")
clf.fit(X_train, y_train)

# Evaluate performance
print("Evaluating on test set...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nResults:")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Number of classes: {len(clf.classes_)}")

# Detailed per-person accuracy
print(f"\nPer-person accuracy (showing first 10 people):")
for person_id in range(min(10, len(clf.classes_))):
    mask = (y_test == person_id)
    if mask.sum() > 0:
        person_acc = (y_pred[mask] == y_test[mask]).mean()
        n_samples = mask.sum()
        print(f"Person {person_id:2d}: {person_acc:.3f} ({n_samples} samples)")

# Show some predictions with images
def show_predictions(n_samples=6):
    """Display sample predictions with face images."""
    if not use_pca:  # Only show if we have original pixel data
        fig, axes = plt.subplots(2, n_samples//2, figsize=(12, 6))
        axes = axes.flatten()
        
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Get original image
            img = X_test[idx].reshape(64, 64)
            
            # Prediction info
            true_person = y_test[idx]
            pred_person = y_pred[idx]
            proba = clf.predict_proba(X_test[idx:idx+1])[0]
            confidence = proba[pred_person]
            
            # Plot
            axes[i].imshow(img, cmap='gray')
            status = "✓" if pred_person == true_person else "✗"
            axes[i].set_title(f'{status} T:{true_person} P:{pred_person} ({confidence:.2f})')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('face_predictions.png', dpi=150, bbox_inches='tight')
        print(f"\nSample predictions saved as 'face_predictions.png'")

if __name__ == "__main__":
    show_predictions()
```

### Feature Extraction Example

```python
from slither import SlitherClassifier
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load text data
print("Loading 20 newsgroups dataset...")
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(
    subset='train', 
    categories=categories,
    shuffle=True, 
    random_state=42
)

# Extract TF-IDF features
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit features for manageable size
    stop_words='english',
    max_df=0.8,
    min_df=2
)

X = vectorizer.fit_transform(newsgroups_train.data).toarray()
y = newsgroups_train.target

print(f"Feature extraction complete:")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Slither classifier on text features
clf = SlitherClassifier(
    n_estimators=15,           # Moderate number for text
    max_depth=10,              # Deeper trees for text complexity
    n_candidate_features=100,  # Many text features
    svm_c=0.1,                # Lower C for sparse text data
    verbose=False
)

print("Training on text features...")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nText Classification Results:")
print(f"Accuracy: {accuracy:.3f}")

# Show category mapping
category_names = newsgroups_train.target_names
print(f"\nCategory mapping:")
for i, name in enumerate(category_names):
    print(f"  {i}: {name}")

# Sample predictions
print(f"\nSample predictions:")
for i in range(5):
    true_cat = category_names[y_test[i]]
    pred_cat = category_names[y_pred[i]]
    status = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"  {status} True: {true_cat}, Predicted: {pred_cat}")
```

## Model Persistence and Deployment

### Saving and Loading Models

```python
from slither import SlitherClassifier
from sklearn.datasets import make_classification
import os
import pickle
import json

# Create and train a model
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

clf = SlitherClassifier(n_estimators=10, verbose=False)
clf.fit(X, y)

print("Model trained successfully!")

# Method 1: Slither's built-in JSON serialization
print("\n1. Using Slither's JSON serialization:")
json_path = "slither_model.json"
clf.save_model(json_path)
print(f"   Model saved to {json_path}")

# Load the model
clf_loaded = SlitherClassifier()
clf_loaded.load_model(json_path)
print("   Model loaded successfully!")

# Verify it works
predictions_original = clf.predict(X[:5])
predictions_loaded = clf_loaded.predict(X[:5])
match = np.array_equal(predictions_original, predictions_loaded)
print(f"   Predictions match: {'✓' if match else '✗'}")

# Method 2: Using pickle (saves full Python object)
print("\n2. Using pickle serialization:")
pickle_path = "slither_model.pkl"
with open(pickle_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"   Model saved to {pickle_path}")

# Load with pickle
with open(pickle_path, 'rb') as f:
    clf_pickled = pickle.load(f)
print("   Model loaded successfully!")

# Verify
predictions_pickled = clf_pickled.predict(X[:5])
match = np.array_equal(predictions_original, predictions_pickled)
print(f"   Predictions match: {'✓' if match else '✗'}")

# Cleanup
os.remove(json_path)
os.remove(pickle_path)
print("\n✓ Cleanup completed")
```

### Model Deployment Example

```python
import numpy as np
from slither import SlitherClassifier
import json
from typing import List, Dict, Any

class SlitherModelService:
    """Simple model service for deployment."""
    
    def __init__(self, model_path: str):
        """Initialize the service with a trained model."""
        self.clf = SlitherClassifier()
        self.clf.load_model(model_path)
        self.model_info = self._get_model_info()
        print(f"Model service initialized: {self.model_info}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'n_classes': getattr(self.clf, 'n_classes_', 'Unknown'),
            'n_features': getattr(self.clf, 'n_features_in_', 'Unknown'),
            'classes': getattr(self.clf, 'classes_', []).tolist() if hasattr(self.clf, 'classes_') else [],
            'parameters': self.clf.get_params()
        }
    
    def predict(self, features: List[List[float]]) -> Dict[str, Any]:
        """Make predictions on input features."""
        X = np.array(features)
        
        # Validate input
        if len(X.shape) != 2:
            raise ValueError("Input must be 2D array")
        
        # Make predictions
        predictions = self.clf.predict(X)
        probabilities = self.clf.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'n_samples': len(predictions),
            'model_info': self.model_info
        }
    
    def health_check(self) -> Dict[str, str]:
        """Simple health check."""
        return {
            'status': 'healthy',
            'model_loaded': 'yes' if hasattr(self.clf, '_fitted') and self.clf._fitted else 'no'
        }

# Example usage
if __name__ == "__main__":
    # First train and save a model
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=8, n_classes=3, random_state=42)
    clf = SlitherClassifier(n_estimators=5, verbose=False)
    clf.fit(X, y)
    clf.save_model("deployment_model.json")
    
    # Initialize service
    service = SlitherModelService("deployment_model.json")
    
    # Health check
    health = service.health_check()
    print(f"Health check: {health}")
    
    # Make predictions
    test_data = X[:3].tolist()  # Use first 3 samples
    results = service.predict(test_data)
    
    print(f"\nPrediction results:")
    print(f"  Predictions: {results['predictions']}")
    print(f"  Probabilities shape: {len(results['probabilities'])} x {len(results['probabilities'][0])}")
    print(f"  Model classes: {results['model_info']['classes']}")
    
    # Cleanup
    import os
    os.remove("deployment_model.json")
```

## Performance Benchmarking

### Comparing with Other Algorithms

```python
from slither import SlitherClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time
import numpy as np

# Generate benchmark dataset
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define algorithms to compare
algorithms = {
    'Slither RF': SlitherClassifier(n_estimators=20, max_depth=8, verbose=False),
    'Standard RF': RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

results = {}

print("Algorithm Comparison Benchmark")
print("=" * 50)

for name, clf in algorithms.items():
    print(f"\nTesting {name}...")
    
    # Measure training time
    start_time = time.time()
    try:
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        y_pred = clf.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': training_time,
            'pred_time': prediction_time,
            'status': 'success'
        }
        
        print(f"  ✓ Accuracy: {accuracy:.3f}")
        print(f"  ✓ CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  ✓ Training: {training_time:.2f}s")
        print(f"  ✓ Prediction: {prediction_time:.4f}s")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        results[name] = {'status': 'failed', 'error': str(e)}

# Summary table
print(f"\n\nSummary Results")
print("=" * 70)
print(f"{'Algorithm':<12} {'Accuracy':<10} {'CV Score':<12} {'Train(s)':<10} {'Pred(s)':<10}")
print("-" * 70)

for name, result in results.items():
    if result['status'] == 'success':
        print(f"{name:<12} {result['accuracy']:<10.3f} "
              f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}   "
              f"{result['train_time']:<10.2f} {result['pred_time']:<10.4f}")
    else:
        print(f"{name:<12} Failed: {result['error']}")

print("\nNotes:")
print("- Slither RF uses SVM local experts (more complex than standard RF)")
print("- Training times include SVM optimization at each node")  
print("- Results may vary based on dataset characteristics")
```

## Next Steps

- [Python API Reference](../api/python.md) - Complete API documentation
- [Performance Guide](../user-guide/performance.md) - Optimization strategies
- [C++ Library](../user-guide/cpp-library.md) - Working with the C++ implementation
- [Development Guide](../development/building.md) - Building from source