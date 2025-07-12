"""
Base classes for Slither estimators
====================================

This module provides base classes that implement scikit-learn compatible
interfaces for Slither estimators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .exceptions import SlitherNotFittedError, SlitherValidationError


class SlitherBase(BaseEstimator, ABC):
    """Base class for all Slither estimators.
    
    This class provides common functionality and ensures scikit-learn
    compatibility for all Slither estimators.
    
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_candidate_features = n_candidate_features
        self.n_candidate_thresholds = n_candidate_thresholds
        self.svm_c = svm_c
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Internal state
        self._fitted = False
        self._slither_model = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[NDArray] = None
    
    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        if self.n_estimators <= 0:
            raise SlitherValidationError("n_estimators must be positive")
        if self.max_depth <= 0:
            raise SlitherValidationError("max_depth must be positive")
        if self.n_candidate_features <= 0:
            raise SlitherValidationError("n_candidate_features must be positive")
        if self.n_candidate_thresholds <= 0:
            raise SlitherValidationError("n_candidate_thresholds must be positive")
        if self.svm_c <= 0:
            raise SlitherValidationError("svm_c must be positive")
        if self.n_jobs <= 0:
            raise SlitherValidationError("n_jobs must be positive")
    
    def _validate_input(
        self, 
        X: NDArray, 
        y: Optional[NDArray] = None,
        reset: bool = True
    ) -> tuple[NDArray, Optional[NDArray]]:
        """Validate input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,), optional
            Target values.
        reset : bool, default=True
            Whether to reset the fitted state.
            
        Returns
        -------
        X_validated : ndarray
            Validated input features.
        y_validated : ndarray, optional
            Validated target values.
        """
        if y is not None:
            X, y = check_X_y(X, y, dtype=np.float64, accept_sparse=False)
        else:
            X = check_array(X, dtype=np.float64, accept_sparse=False)
            
        if reset:
            self.n_features_in_ = X.shape[1]
            if hasattr(X, 'columns'):
                self.feature_names_in_ = np.array(X.columns)
                
        return X, y
    
    def _check_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not self._fitted:
            raise SlitherNotFittedError()
    
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> "SlitherBase":
        """Fit the estimator to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass
    
    @abstractmethod 
    def predict(self, X: NDArray) -> NDArray:
        """Make predictions on input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return super().get_params(deep=deep)
    
    def set_params(self, **params: Any) -> "SlitherBase":
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        super().set_params(**params)
        self._validate_parameters()
        return self