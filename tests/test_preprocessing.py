import pytest
import pandas as pd
import numpy as np
from src.features import build_transformer, save_transformer, load_transformer
from src.data import load_data, split_data

def test_build_transformer():
    """Test that transformer is built without errors."""
    numeric_features = ['age', 'cholesterol']
    categorical_features = ['sex', 'cp']
    
    preprocessor = build_transformer(numeric_features, categorical_features)
    assert preprocessor is not None
    assert hasattr(preprocessor, 'fit_transform')

def test_save_and_load_transformer(tmp_path):
    """Test saving and loading transformer."""
    numeric_features = ['age', 'cholesterol']
    categorical_features = ['sex', 'cp']
    
    preprocessor = build_transformer(numeric_features, categorical_features)
    path = tmp_path / "test_transformer.joblib"
    
    save_transformer(preprocessor, str(path))
    loaded = load_transformer(str(path))
    
    assert loaded is not None
    assert type(loaded) == type(preprocessor)

def test_split_data():
    """Test data splitting functionality."""
    # Create sample data
    df = pd.DataFrame({
        'age': np.random.randint(30, 80, 100),
        'cholesterol': np.random.randint(100, 300, 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, 'target')
    
    # Check sizes (approximately)
    total_size = len(df)
    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)
    
    assert train_size + val_size + test_size == total_size
    assert test_size / total_size == pytest.approx(0.2, abs=0.05)
