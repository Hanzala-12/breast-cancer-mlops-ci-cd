# MLflow Complete Workflow & CI/CD Auto Retraining Project

## Part 1: MLflow Workflow
- **Dataset**: Breast Cancer Classification (from `sklearn.datasets`).
- **Models**: Random Forest and Logistic Regression.
- **MLflow Tracking**: Training script `src/train.py` logs metrics (`accuracy`, `precision`, `recall`, `f1_score`) and parameters. Models are automatically registered.
- **Model Registry**: Models are pushed to the registry, and the best model is identified.

## Part 2: GitHub CI/CD Workflow
- The `.github/workflows/ci_cd.yaml` executes on push.
- It sets up Python, installs `requirements.txt`, runs `src/train.py` (which produces `models/best_model.pkl`), and then tests it with pytest (`tests/test_train.py`).
- A deploy job conditionally pushes the best model to Hugging Face Hub (needs `HF_TOKEN` secret to work).

## Part 3: Auto Retraining Module
- **Trigger**: The `.github/workflows/retrain.yaml` uses a `cron` (Weekly on Sundays) to auto-retrain the model.
- **Pipeline Integration**: Re-runs training code on current/new data, tests strictly ensuring it doesn't break, and finally automatically transitions to the new release mechanism (deploy).
- **Selection Logic**: The train script natively selects the best performing approach internally before deployment (Logistic vs RF).

## Local Execution
1. Create venv: `python -m venv venv` and activate.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train: `python src/train.py`.
4. Test: `pytest tests/`.
5. Run MLflow UI: `mlflow ui`.