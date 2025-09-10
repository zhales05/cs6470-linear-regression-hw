from typing import List, Optional

import numpy as np
from .base import BaseLinearRegression;

class LinearRegressionGD(BaseLinearRegression):
    """
    Linear Regression using Gradient Descent with SSE cost function.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient descent updates
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-6
        Tolerance for convergence criterion
    batch_size : int, default=32
        Size of mini-batches for gradient descent
    random_state : int, optional
        Random seed for reproducibility
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 batch_size: int = 32,
                 random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.cost_history_: List[float] = []
        self.n_iter_: Optional[int] = None

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute the SSE cost function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        cost : float
            SSE cost value
        """
        # TODO: Implement SSE cost function
        # J(theta) = (1/2m) * sum((X @ theta - y)^2)
        pass

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        gradients : np.ndarray
            Gradient vector
        """
        # TODO: Implement gradient computation
        pass

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray) -> list:
        """Generate mini-batches for stochastic gradient descent."""
        
        # Create mini-batches
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionGD':
         # Validate and preprocess input
        X, y = self._validate_input(X, y)
        
        # Add intercept column if needed
        X = self._add_intercept(X)
        
        # Initialize theta (now with correct dimensions)
        theta = np.random.rand(X.shape[1], 1)
            
        """
        Fit the model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : LinearRegressionGD
            Returns self for method chaining
        """

        return self