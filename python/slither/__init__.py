"""
Slither: A Random Forest library with SVM local experts
========================================================

Slither provides a scikit-learn compatible implementation of Random Forests
with Support Vector Machine (SVM) local experts at tree nodes, specifically
designed for computer vision tasks.

Main classes:
- SlitherClassifier: Random forest classifier with SVM local experts
- SlitherRegressor: Random forest regressor with SVM local experts (future)

Examples:
---------
>>> from slither import SlitherClassifier
>>> import numpy as np
>>> X = np.random.random((100, 4))
>>> y = np.random.randint(0, 2, 100)
>>> clf = SlitherClassifier(n_estimators=10, max_depth=5)
>>> clf.fit(X, y)
>>> predictions = clf.predict(X)
>>> probabilities = clf.predict_proba(X)
"""

from .classifier import SlitherClassifier
from .exceptions import SlitherError, SlitherNotFittedError

__version__ = "2.0.0"
__author__ = "Prassanna Ravishankar"
__email__ = "prassanna.ravishankar@gmail.com"

__all__ = [
    "SlitherClassifier",
    "SlitherError", 
    "SlitherNotFittedError",
    "__version__",
]