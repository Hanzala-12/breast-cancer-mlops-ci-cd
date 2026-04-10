import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
import os
import joblib

def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    with mlflow.start_run(run_name=model_name) as run:
        if params:
            mlflow.log_params(params)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        
        signature = infer_signature(X_train, preds)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="BreastCancerModel"
        )
        print(f"{model_name} logged. Accuracy: {acc}")
        
        return acc, model, model_info.model_uri, run.info.run_id, model_name

def main():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Disable autologging to prevent duplicate metrics and ensure clean logging
    mlflow.autolog(disable=True) 

    X_train, X_test, y_train, y_test = load_data()
    
    models_run = []

    # Model 1: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
    rf = RandomForestClassifier(**rf_params)
    models_run.append(train_and_log_model(rf, "RandomForest", X_train, X_test, y_train, y_test, rf_params))
    
    # Model 2: Logistic Regression
    lr_params = {"max_iter": 1000, "random_state": 42}
    lr = LogisticRegression(**lr_params)
    models_run.append(train_and_log_model(lr, "LogisticRegression", X_train, X_test, y_train, y_test, lr_params))

    # Model 3: Gradient Boosting
    gb_params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
    gb = GradientBoostingClassifier(**gb_params)
    models_run.append(train_and_log_model(gb, "GradientBoosting", X_train, X_test, y_train, y_test, gb_params))
    
    # Sort and pick the best model from THIS retraining cycle
    models_run.sort(key=lambda x: x[0], reverse=True)
    best_new_acc, best_new_model, best_new_uri, best_new_run_id, best_new_name = models_run[0]
    print(f"\nBest model from current run: {best_new_name} with Accuracy: {best_new_acc}")

    client = MlflowClient()
    model_name = "BreastCancerModel"
    promote_to_production = True

    try:
        # Get the current "Production" model to compare
        latest_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if latest_versions:
            prod_version = latest_versions[0]
            prod_run_id = prod_version.run_id
            prod_run = client.get_run(prod_run_id)
            prod_acc = prod_run.data.metrics.get("accuracy", 0.0)

            print(f"Current Production Model Accuracy: {prod_acc}")
            # Retraining Trigger Logic: Only replace if performance improves
            if best_new_acc > prod_acc:
                print("New model outperformed the production model. Promoting...")
            else:
                print("New model did not outperform the production model. Keeping existing model.")
                promote_to_production = False
    except Exception as e:
        print("No existing production model found or registry not yet initialized. Deploying as first production model.")

    if promote_to_production:
        print("Saving new Best Model...")
        joblib.dump(best_new_model, "models/best_model.pkl")

        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = next((v for v in versions if v.run_id == best_new_run_id), None)

        if target_version:
            # First transition to 'Staging'
            client.transition_model_version_stage(
                name=model_name,
                version=target_version.version,
                stage="Staging"
            )
            # Then transition to 'Production'
            client.transition_model_version_stage(
                name=model_name,
                version=target_version.version,
                stage="Production"
            )
            print(f"Model version {target_version.version} successfully promoted to Production.")

if __name__ == "__main__":
    main()
