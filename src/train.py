import os 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from src.data import load_data, save_data 
from src.features import build_transformer, save_transformer
from src.model import build_model, save_model

def evaluate_classification(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else None
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def model_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def run_training(raw_path, target_col, numeric_features, classification_features, problem_type='classification'):
    os.makedirs('models', exist_ok=True)
    os.makedirs('transformers', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    df = load_data(raw_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)
    preprocressor = build_transformer(numeric_features, classification_features)
    save_transformer(preprocressor, 'models/transformer.joblib')
    model = build_model(problem_type)
    pipe = Pipeline([
        ('processor', preprocressor),
        ('model', model)
    ])


    # RandomizedSearchCV here
    pipe.fit(X_train, y_train)

    # Save the model
    save_model(pipe, 'models/model.joblib')

    # Evaluate
    if problem_type == 'classification':
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_proba)
    else:
        y_pred = pipe.predict(X_test)
        metrics = model_regression(y_test, y_pred)

    # Save metrics
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics:", metrics)

    # Plot confusion matrix for classification
    if problem_type == 'classification':
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', aplha=0.6)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.savefig('reports/figures/actual_vs_predicted.png')
        plt.close()


if __name__ == "__main__":
    raw_path = 'data/raw/heart_disease_data.csv'
    target_col = 'target'
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    classification_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    run_training(raw_path, target_col, numeric_features, classification_features, problem_type='classification')