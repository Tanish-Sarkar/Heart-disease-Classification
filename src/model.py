import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def build_model(problem_type):
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif problem_type == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported problem type. Choose 'classification' or 'regression'.")
    
def save_model(model, path='models/model.joblib'):
    joblib.dump(model, path)

def load_model(path='models/model.joblib'):
    return joblib.load(path)