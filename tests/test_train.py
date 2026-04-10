import pytest
import os
import joblib

def test_best_model_exists():
    assert os.path.exists("models/best_model.pkl"), "Best model artifact was not generated."

def test_model_type():
    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        assert model is not None
        assert type(model).__name__ in ["RandomForestClassifier", "LogisticRegression"]
