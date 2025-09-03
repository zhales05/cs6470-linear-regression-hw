from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

class BaseLinearRegression(ABC):
    """
    Abstract base class for linear regression implementations.
    
    This class defines the common interface and shared functionality for
    different linear regression algorithms.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and preprocess input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        X_processed : np.ndarray
            Processed feature matrix
        y_processed : np.ndarray or None
            Processed target vector
        """
        # Convert to numpy array
        pass
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix if fit_intercept is True."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseLinearRegression':
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : BaseLinearRegression
            Returns self for method chaining
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        # Check if model is fitted
        
        # Make predictions
        
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X
            
        Returns
        -------
        score : float
            R score
        """
        pass