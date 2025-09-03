import pandas as pd

def load_data(path='data/housing_data.csv'):
    df = pd.read_csv(path)
    X = df[['size', 'bedrooms', 'age']].values
    y = df['price'].values
    return X,y