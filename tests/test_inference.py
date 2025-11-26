import pytest
import joblib
import pandas as pd
from unittest.mock import patch, MagicMock
from src.predict import predict_from_json

def test_predict_from_json_classification():
    """Test prediction from JSON for classification model."""
    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    
    with patch('joblib.load', return_value=mock_model):
        result = predict_from_json({'age': 54, 'cholesterol': 233})
        
        assert 'prediction' in result
        assert 'probabilities' in result
        assert result['prediction'] == 1
        assert result['probabilities'] == [0.3, 0.7]

def test_predict_from_json_handles_regression():
    """Test prediction handles regression models gracefully."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [5.5]
    mock_model.predict_proba.side_effect = AttributeError()
    
    with patch('joblib.load', return_value=mock_model):
        result = predict_from_json({'age': 54, 'value': 100})
        
        assert 'prediction' in result
        assert 'probabilities' in result
        assert result['prediction'] == 5.5
        assert result['probabilities'] is None
