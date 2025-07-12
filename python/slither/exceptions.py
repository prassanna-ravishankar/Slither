"""
Custom exception hierarchy for Slither
=======================================

This module defines custom exceptions for the Slither library,
providing clear error messages and proper exception handling.
"""

from typing import Optional


class SlitherError(Exception):
    """Base exception class for all Slither-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class SlitherNotFittedError(SlitherError):
    """Exception raised when trying to use an unfitted estimator.
    
    This exception is raised when calling predict, predict_proba, or other
    methods that require a fitted model, but the model has not been trained yet.
    """
    
    def __init__(self, method_name: str = "predict"):
        message = (
            f"This {self.__class__.__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using {method_name}."
        )
        super().__init__(message)


class SlitherValidationError(SlitherError):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Input validation error: {message}")


class SlitherTrainingError(SlitherError):
    """Exception raised during model training."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(f"Training error: {message}", original_exception)


class SlitherPredictionError(SlitherError):
    """Exception raised during prediction."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(f"Prediction error: {message}", original_exception)


class SlitherSerializationError(SlitherError):
    """Exception raised during model serialization/deserialization."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(f"Serialization error: {message}", original_exception)