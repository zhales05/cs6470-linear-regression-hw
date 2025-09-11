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

if __name__ == "__main__":
    from lr.gradual_descent import LinearRegressionGD
    
    # Load and split data  
    X, y = load_data()
    
    # Normalize data to prevent numerical issues
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(), y.std()
    
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # train
    model = LinearRegressionGD(learning_rate=0.01, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    print(f"Converged in {model.n_iter_} iterations")