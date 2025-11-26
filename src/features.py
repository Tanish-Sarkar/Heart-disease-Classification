from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def build_transformer(numeric_features, categorical_features):
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.
    Args:
        numeric_features (list): List of names of numeric features.
        categorical_features (list): List of names of categorical features.
        Returns:
        ColumnTransformer: A fitted ColumnTransformer object.
    """
    # Numeric features pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    # Catergorical Feature pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

def save_tranformer(preprocessor, path="models/tranformer.joblib"):
    joblib.dump(preprocessor, path)

def load_tranformer(path="models/tranformer.joblib"):
    return joblib.load(path)