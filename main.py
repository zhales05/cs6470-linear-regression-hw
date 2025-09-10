import pandas as pd
import numpy as np


def load_data(path='data/housing_data.csv'):
    df = pd.read_csv(path)
    X = df[['size', 'bedrooms', 'age']].values
    y = df['price'].values
    return X,y

def split_data(X, y, test_size=0.2, random_state = 1):
    np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
    