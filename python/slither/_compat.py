"""
Compatibility layer for the C++ extension
==========================================

This module provides a compatibility layer between the old C++ wrapper
and the new scikit-learn compatible API while we transition.
"""

import sys
from pathlib import Path

# Try to import the existing C++ extension
try:
    # Add the build directory to the path
    build_dir = Path(__file__).parent.parent.parent / "slither_build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))
    
    from slither_py import slither as _SlitherWrapper
    EXTENSION_AVAILABLE = True
    
except ImportError:
    # Create a mock wrapper for development/testing
    class _MockSlitherWrapper:
        """Mock wrapper for testing when C++ extension is not available."""
        
        def __init__(self):
            self._fitted = False
            self._data_loaded = False
        
        def setParams(self, max_depth, n_features, n_thresholds, n_trees, svm_c, verbose, n_jobs):
            return True
            
        def loadData(self, X, y):
            self._data_loaded = True
            return True
            
        def onlyTrain(self):
            if not self._data_loaded:
                return False
            self._fitted = True
            return True
            
        def onlyTest(self):
            if not self._fitted:
                raise RuntimeError("Model not fitted")
            # Return mock probabilities
            import numpy as np
            return np.random.random((10, 2))
            
        def saveModel(self, filepath):
            return self._fitted
            
        def loadModel(self, filepath):
            self._fitted = True
            return True
    
    _SlitherWrapper = _MockSlitherWrapper
    EXTENSION_AVAILABLE = False


def get_slither_wrapper():
    """Get the Slither wrapper class."""
    return _SlitherWrapper


def is_extension_available():
    """Check if the C++ extension is available."""
    return EXTENSION_AVAILABLE