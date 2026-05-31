import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns   
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
import shutil
import optuna # Import Optuna untuk hyperparameter tuning

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Create a new MLflow Experiment
mlflow.set_experiment("Modelling Weather Prediction Tuning")

# Load dataset
df = pd.read_csv(r"weather_preprocessing.csv", sep = ",")

X = df.drop(columns=["rain_next_6h"])
y = df["rain_next_6h"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
input_example = x_train[0:5]

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    score = cross_val_score(model, x_train, y_train, n_jobs=-1, cv=3)
    return score.mean()

with mlflow.start_run(run_name="Tuning_RandomForest_with_Optuna", nested=False) as run:
    print("Memulai Hyperparameter Tuning dengan Optuna...")
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=15)

    print("\nPencarian selesai! Parameter terbaik:", study.best_params)

    best_model = RandomForestClassifier(**study.best_params, random_state=69420)
    best_model.fit(x_train, y_train)

    y_test_pred = best_model.predict(x_test)
    y_train_pred = best_model.predict(x_train)

    print("Melakukan Manual Logging Parameter dan Metrik ke MLflow...")

    # Log Parameter Terbaik
    mlflow.log_params(study.best_params)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average="weighted")
    rec = recall_score(y_test, y_test_pred, average="weighted")
    f1 = f1_score(y_test, y_test_pred, average="weighted")

    # Log Metrik
    mlflow.log_metrics(
        {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    )

    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="best_tuning_model")

    if os.path.exists("best_tuning_model"):
        shutil.rmtree("best_tuning_model")
    
    mlflow.sklearn.save_model(
        best_model,
        "best_tuning_model",
        input_example=input_example,
    )

    mlflow.log_artifacts("best_tuning_model", artifact_path="best_tuning_model")

    # Log Model Machine Learning
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm_test, annot=True, fmt="d")
    plt.title("Test Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    cm_train = confusion_matrix(y_train, y_train_pred)
    plt.figure()
    sns.heatmap(cm_train, annot=True, fmt="d")
    plt.title("Training Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    plt.close()

    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(x_test)[:, 1]

        # ROC
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve")
        plt.savefig("training_roc_curve.png")
        mlflow.log_artifact("training_roc_curve.png")
        plt.close()

        # Precision-Recall
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title("Precision Recall Curve")
        plt.savefig("training_precision_recall_curve.png")
        mlflow.log_artifact("training_precision_recall_curve.png")
        plt.close()

    # Log classification report
    report = classification_report(y_test, y_test_pred)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Laporan Klasifikasi Hasil Optuna Tuning:\n\n")
        f.write(f"Parameter Terbaik: {study.best_params}\n\n")
        f.write(report)

    mlflow.log_artifact(report_path)
