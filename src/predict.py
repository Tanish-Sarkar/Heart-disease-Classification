import joblib
import pandas as pd

def predict_from_json(json_data, model_path='models/model.joblib'):
    # Load the trained model
    model = joblib.load(model_path)
    
    # Convert JSON data to DataFrame
    input_data = pd.DataFrame([json_data])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get probabilities if available (for classification)
    try:
        probabilities = model.predict_proba(input_data)[0]
    except:
        probabilities = None
    
    return {
        'prediction': int(prediction[0]) if prediction.dtype == 'int64' else float(prediction[0]),
        'probabilities': probabilities.tolist() if probabilities is not None else None
    }


if __name__ == "__main__":
    sample = {"num_col1": 1.2, "num_col2": 3.4, "cat_col1": "A"}
    print(predict_from_json(sample))