"""
Tests for SlitherClassifier
============================

Test suite for the scikit-learn compatible SlitherClassifier.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from slither import SlitherClassifier
from slither.exceptions import SlitherNotFittedError, SlitherValidationError


class TestSlitherClassifier:
    """Test class for SlitherClassifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            n_redundant=2,
            n_informative=8,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_init_default_parameters(self):
        """Test classifier initialization with default parameters."""
        clf = SlitherClassifier()
        
        assert clf.n_estimators == 100
        assert clf.max_depth == 15
        assert clf.n_candidate_features == 50
        assert clf.n_candidate_thresholds == 50
        assert clf.svm_c == 0.5
        assert clf.random_state is None
        assert clf.n_jobs == 1
        assert clf.verbose is False
    
    def test_init_custom_parameters(self):
        """Test classifier initialization with custom parameters."""
        clf = SlitherClassifier(
            n_estimators=50,
            max_depth=10,
            n_candidate_features=25,
            n_candidate_thresholds=25,
            svm_c=1.0,
            random_state=42,
            n_jobs=2,
            verbose=True
        )
        
        assert clf.n_estimators == 50
        assert clf.max_depth == 10
        assert clf.n_candidate_features == 25
        assert clf.n_candidate_thresholds == 25
        assert clf.svm_c == 1.0
        assert clf.random_state == 42
        assert clf.n_jobs == 2
        assert clf.verbose is True
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(SlitherValidationError):
            clf = SlitherClassifier(n_estimators=0)
            clf._validate_parameters()
        
        with pytest.raises(SlitherValidationError):
            clf = SlitherClassifier(max_depth=-1)
            clf._validate_parameters()
        
        with pytest.raises(SlitherValidationError):
            clf = SlitherClassifier(svm_c=0)
            clf._validate_parameters()
    
    def test_not_fitted_error(self):
        """Test error when using unfitted classifier."""
        clf = SlitherClassifier()
        X = np.random.random((10, 5))
        
        with pytest.raises(SlitherNotFittedError):
            clf.predict(X)
        
        with pytest.raises(SlitherNotFittedError):
            clf.predict_proba(X)
    
    def test_fit_predict_simple(self, sample_data):
        """Test basic fit and predict functionality."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use smaller parameters for faster testing
        clf = SlitherClassifier(
            n_estimators=5,
            max_depth=5,
            n_candidate_features=5,
            n_candidate_thresholds=5,
            verbose=False
        )
        
        # Fit the classifier
        clf.fit(X_train, y_train)
        
        # Check that it's fitted
        assert clf._fitted is True
        assert clf.n_features_in_ == X_train.shape[1]
        assert clf.classes_ is not None
        assert clf.n_classes_ == 2
        
        # Make predictions
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        # Check output shapes
        assert predictions.shape == (X_test.shape[0],)
        assert probabilities.shape == (X_test.shape[0], 2)
        
        # Check that probabilities sum to 1
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        
        # Check that predictions match argmax of probabilities
        predicted_from_proba = clf.classes_[np.argmax(probabilities, axis=1)]
        np.testing.assert_array_equal(predictions, predicted_from_proba)
    
    def test_score_method(self, sample_data):
        """Test the score method."""
        X_train, X_test, y_train, y_test = sample_data
        
        clf = SlitherClassifier(
            n_estimators=5,
            max_depth=5,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        score = clf.score(X_test, y_test)
        
        # Score should be between 0 and 1
        assert 0 <= score <= 1
        
        # Should match manual calculation
        predictions = clf.predict(X_test)
        manual_score = np.mean(predictions == y_test)
        assert score == manual_score
    
    def test_get_set_params(self):
        """Test parameter getting and setting."""
        clf = SlitherClassifier()
        
        # Get parameters
        params = clf.get_params()
        assert 'n_estimators' in params
        assert 'max_depth' in params
        
        # Set parameters
        clf.set_params(n_estimators=50, max_depth=10)
        assert clf.n_estimators == 50
        assert clf.max_depth == 10
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_redundant=2,
            n_informative=8,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        clf = SlitherClassifier(
            n_estimators=5,
            max_depth=5,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        assert clf.n_classes_ == 3
        assert len(clf.classes_) == 3
        
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        assert probabilities.shape == (X_test.shape[0], 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])