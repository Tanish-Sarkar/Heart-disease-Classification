import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Further split trainval into train and validation sets
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_relative_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

