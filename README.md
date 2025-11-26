<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.2+-orange" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
  <img src="https://img.shields.io/badge/Model-RandomForestClassifier-forestgreen" />
</p>
# ❤️ Heart Disease Prediction — End-to-End Machine Learning Project

An end-to-end **Machine Learning Classification** system that predicts whether a person has heart disease based on clinical features.
This project includes **EDA, preprocessing pipeline, ML model training, evaluation, saved model artifacts, inference script, and a FastAPI endpoint** for real-time predictions.

This is a **production-style ML pipeline**, not just a notebook demo.

---

# 🚀 **1. Project Overview**

This project demonstrates how to build a **complete ML system**, starting from raw data to a deployable API.

You will learn:

* How to structure a real ML project
* How to perform EDA
* How to build a deterministic preprocessing pipeline
* How to train and evaluate ML models
* How to export and load model artifacts
* How to create an inference script
* How to expose predictions via a FastAPI web service

This project is ideal for:

* Students
* ML beginners
* Developers switching to ML
* Anyone preparing ML portfolios

---

# 📌 **2. Dataset**

This project uses the **Heart Disease** dataset.

Typical columns:

```
age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target
```

**Target column:**
`target`

* **1** → Heart disease
* **0** → No heart disease

Place your dataset here:

```
data/raw/heart.csv
```

---

# 📂 **3. Project Structure**

```
Heart-Disease-Classification/
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/  
|          └── heart_processed.csv     
│
├── notebooks/
│   └── eda.ipynb         ← EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data.py           ← load & split raw dataset
│   ├── features.py       ← preprocessing pipeline
│   ├── train.py          ← training, evaluation, saving model
│   ├── predict.py        ← inference script
│   └── app.py            ← FastAPI prediction API
│
├── models/
│   └── heart_model.joblib
│
├── reports/
│   ├── metrics.json
│   └── figures/
│       ├── confusion_matrix.png
│       └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

# 🔍 **4. Exploratory Data Analysis (EDA)**

EDA is done in:

```
notebooks/eda.ipynb
reports/EDA-report.md
```

It includes:

* Dataset shape & summary
* Missing value analysis
* Distribution plots
* Target balance
* Correlation heatmap
* Pairplot for selected features

Findings:

* Dataset has no missing values
* Slight class imbalance (manageable)
* Features like chest pain type, thalach, and oldpeak show strong correlation with the target
* Numeric features need scaling

---

# ⚙️ **5. Preprocessing Pipeline**

Implemented in:

```
src/features.py
```

Includes:

* Median imputation for numeric values
* Most-frequent imputation for categorical values
* Standard scaling (numerical)
* One-hot encoding (categorical)

Everything is done using a **ColumnTransformer** inside an sklearn **Pipeline** for full reproducibility.

---

# 🤖 **6. Model Training**

Training code lives in:

```
src/train.py
```

Steps:

1. Load raw data
2. Split into train/val/test
3. Apply preprocessing
4. Train a baseline **RandomForestClassifier**
5. Evaluate using:

   * Accuracy
   * ROC-AUC
6. Save:

   * `heart_model.joblib`
   * `metrics.json`
   * confusion matrix plot
   * ROC curve plot

Run training:

```bash
python -m src.train
```

---

# 📊 **7. Evaluation Results**

Evaluation files are saved in:

```
reports/
│
├── metrics.json
└── figures/
    ├── confusion_matrix.png
    └── roc_curve.png
```
<img src="https://github.com/Tanish-Sarkar/Heart-disease-Classification/blob/main/reports/figures/confusion_matrix.png" alt='cm'>
<img src="https://github.com/Tanish-Sarkar/Heart-disease-Classification/blob/main/reports/figures/roc_curve.png" alt='roc'>
Example metrics:

```json
{
  "accuracy": 0.85,
  "roc_auc": 0.91
}
```

---

# 🔮 **8. Inference Script**

File:

```
src/predict.py
```

Usage:

```python
from src.predict import predict_from_json

sample = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    ...
}

result = predict_from_json(sample)

print(result)
```

Output:

```json
{
  "prediction": 1,
  "probabilities": [0.26, 0.74]
}
```

---

# 🌐 **9. FastAPI Web Service**

File:

```
src/app.py
```

Start API:

```bash
uvicorn src.app:app --reload --port 8000
```

Endpoints:

### **GET /**

Health check
→ `{"message": "Heart Disease Prediction API is running. Use POST /predict."}`

### **POST /predict**

Predict heart disease from JSON input.

Example request:

```json
{
  "data": {
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
    "thal": 1
  }
}
```

Example response:

```json
{
  "prediction": 1,
  "probabilities": [0.26, 0.74]
}
```

Interactive LIVE API:
👉 [https://heart-disease-classification-mg7u.onrender.com/docs](https://heart-disease-classification-mg7u.onrender.com/docs)

---

# 🧪 **10. Tests (Optional but recommended)**

Example test:

```
tests/test_preprocessing.py
```

Run tests:

```bash
pytest
```

---

# 🛠️ **11. Installation & Setup**

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python -m src.train
```

### Run API

```bash
uvicorn src.app:app --reload --port 8000
```

---

# 🧱 **12. Technologies Used**

* Python
* Pandas
* NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn

---

# 🏁 **13. Future Improvements**

You can extend this project by:

* Hyperparameter tuning (RandomizedSearchCV)
* Using XGBoost
* Adding SHAP explainability
* Adding CI tests for data & inference
* Dockerizing the API
* Deploying on Render / Railway

---

# 🎉 **14. Conclusion**

This project is a complete, production-style ML pipeline that:

* Loads & cleans data
* Does EDA
* Preprocesses deterministically
* Trains & evaluates a model
* Saves artifacts
* Serves predictions through an API

