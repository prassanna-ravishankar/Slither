"""
Slither Random Forest Classifier
=================================

This module provides the SlitherClassifier class, a scikit-learn compatible
random forest classifier with SVM local experts.
"""

from typing import Optional, Union
import tempfile
import os

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from .base import SlitherBase
from .exceptions import (
    SlitherNotFittedError, 
    SlitherTrainingError, 
    SlitherPredictionError,
    SlitherSerializationError
)

# Import the C++ extension with compatibility layer
from ._compat import get_slither_wrapper, is_extension_available

_SlitherWrapper = get_slither_wrapper()

if not is_extension_available():
    import warnings
    warnings.warn(
        "Slither C++ extension not available. "
        "Using mock implementation for development. "
        "Some functionality may be limited.",
        UserWarning
    )


class SlitherClassifier(SlitherBase, ClassifierMixin):
    """Random Forest Classifier with SVM local experts.
    
    This classifier implements a random forest where each tree node contains
    a Support Vector Machine (SVM) local expert, designed specifically for
    computer vision and high-dimensional classification tasks.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=15
        The maximum depth of the trees.
    n_candidate_features : int, default=50
        Number of candidate features to consider at each split.
    n_candidate_thresholds : int, default=50  
        Number of candidate thresholds to consider for each feature.
    svm_c : float, default=0.5
        Regularization parameter for SVM local experts.
    random_state : int, optional
        Random state for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for training.
    verbose : bool, default=False
        Whether to print training progress.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Only defined if X has feature names.
        
    Examples
    --------
    >>> from slither import SlitherClassifier
    >>> import numpy as np
    >>> X = np.random.random((100, 4))
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = SlitherClassifier(n_estimators=10, max_depth=5)
    >>> clf.fit(X, y)
    SlitherClassifier(...)
    >>> clf.predict(X[:5])
    array([0, 1, 0, 1, 1])
    >>> clf.predict_proba(X[:5])
    array([[0.7, 0.3],
           [0.2, 0.8],
           [0.9, 0.1],
           [0.1, 0.9],
           [0.3, 0.7]])
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        n_candidate_features: int = 50,
        n_candidate_thresholds: int = 50,
        svm_c: float = 0.5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_candidate_features=n_candidate_features,
            n_candidate_thresholds=n_candidate_thresholds,
            svm_c=svm_c,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        # Classification-specific attributes
        self.classes_: Optional[NDArray] = None
        self.n_classes_: Optional[int] = None
    
    def fit(self, X: NDArray, y: NDArray) -> "SlitherClassifier":
        """Fit the Random Forest classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Training data labels.
            
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate parameters
        self._validate_parameters()
        
        # Validate and prepare input data
        X, y = self._validate_input(X, y, reset=True)
        
        # Store class information
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Ensure labels are contiguous integers starting from 0
        y_encoded = np.searchsorted(self.classes_, y)
        
        try:
            # Create and configure the C++ wrapper
            self._slither_model = _SlitherWrapper()
            
            # Set training parameters
            self._slither_model.setParams(
                self.max_depth,
                self.n_candidate_features, 
                self.n_candidate_thresholds,
                self.n_estimators,
                self.svm_c,
                self.verbose,
                self.n_jobs
            )
            
            # Load training data
            if not self._slither_model.loadData(X, y_encoded):
                raise SlitherTrainingError("Failed to load training data")
            
            # Train the model
            if not self._slither_model.onlyTrain():
                raise SlitherTrainingError("Training failed")
                
            self._fitted = True
            
        except Exception as e:
            raise SlitherTrainingError(
                f"Training failed: {str(e)}", 
                original_exception=e
            )
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Get probabilities and return the class with highest probability
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Check if fitted
        self._check_fitted()
        
        # Validate input
        X, _ = self._validate_input(X, reset=False)
        
        try:
            # Load test data into the model
            # Create dummy labels (they won't be used for prediction)
            dummy_labels = np.zeros(X.shape[0])
            
            if not self._slither_model.loadData(X, dummy_labels):
                raise SlitherPredictionError("Failed to load test data")
            
            # Get predictions
            probas = self._slither_model.onlyTest()
            
            # Ensure we have the right shape
            if probas.shape[1] != self.n_classes_:
                raise SlitherPredictionError(
                    f"Expected {self.n_classes_} classes, got {probas.shape[1]}"
                )
            
            return probas
            
        except Exception as e:
            raise SlitherPredictionError(
                f"Prediction failed: {str(e)}",
                original_exception=e
            )
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file.
        
        Parameters
        ----------
        filepath : str
            Path where to save the model.
        """
        self._check_fitted()
        
        try:
            if not self._slither_model.saveModel(filepath):
                raise SlitherSerializationError("Failed to save model")
        except Exception as e:
            raise SlitherSerializationError(
                f"Failed to save model: {str(e)}",
                original_exception=e
            )
    
    def load_model(self, filepath: str) -> "SlitherClassifier":
        """Load a trained model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file.
            
        Returns
        -------
        self : object
            Loaded classifier.
        """
        try:
            self._slither_model = _SlitherWrapper()
            
            if not self._slither_model.loadModel(filepath):
                raise SlitherSerializationError("Failed to load model")
                
            self._fitted = True
            
            # Note: When loading a model, we don't have access to the original
            # class information. This is a limitation that could be addressed
            # by saving additional metadata.
            
        except Exception as e:
            raise SlitherSerializationError(
                f"Failed to load model: {str(e)}",
                original_exception=e
            )
        
        return self
    
    def _more_tags(self):
        """Provide additional tags for scikit-learn compatibility."""
        return {
            'requires_positive_X': False,
            'requires_positive_y': False,
            'requires_fit': True,
            'poor_score': True,  # Will be removed once we have proper benchmarks
            'no_validation': False,
            'multiclass_only': True,
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
        }