# Heart Disease Classification

A machine learning project for classifying heart disease using patient health metrics.

## Project Structure

```
├── data/
│   ├── raw/              # Original data files
│   └── processed/        # Preprocessed data
├── models/               # Trained model artifacts
├── notebooks/            # Jupyter notebooks for EDA
├── reports/              # Generated reports and metrics
├── src/                  # Main source code
│   ├── app.py           # FastAPI application
│   ├── data.py          # Data loading and splitting
│   ├── features.py      # Feature preprocessing
│   ├── model.py         # Model building
│   ├── train.py         # Training pipeline
│   └── predict.py       # Inference functions
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── Dockerfile           # Docker configuration
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
Run the training pipeline:
```bash
python -m src.train
```

### API Server
Start the FastAPI server:
```bash
uvicorn src.app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Prediction
Make predictions programmatically:
```python
from src.predict import predict_from_json

result = predict_from_json({
    'age': 54, 
    'cholesterol': 233,
    'sex': 'M'
})
print(result)
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

## Model Performance

Metrics are saved to `reports/metrics.json` after training.
Confusion matrix and other visualizations are saved to `reports/figures/`.

## Features

- **Data Preprocessing**: Automated handling of numeric and categorical features
- **Model Training**: RandomForest classifier with pipeline integration
- **REST API**: FastAPI endpoint for real-time predictions
- **Testing**: Unit tests for preprocessing and inference
