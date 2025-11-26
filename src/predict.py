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
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1}
    print(predict_from_json(sample))