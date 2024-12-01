"""
This file ensures that the `models` directory is treated as a package.
It can be left empty, or you can initialize things or import classes here if needed.

Imports:
    ClassificationModel: The model class used for iris species classification.
    RegressionModel: The model class used for house price prediction.
"""

from .classification_model import ClassificationModel
from .regression_model import RegressionModel
