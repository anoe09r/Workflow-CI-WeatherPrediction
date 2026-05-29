import matplotlib
matplotlib.use("Agg")
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Modelling Weather Prediction")

mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv(r"D:\ANUGRAH\2026\Continuous Learning\Dicoding\IDCamp2025\MSML\weather_preprocessing.csv", sep = ",")

X = df.drop(columns=["rain_next_6h"])
y = df["rain_next_6h"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="RandomForest_Model") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    y_prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction)

    print(f"Akurasi Model: {accuracy:.4f}")
    print("\nLaporan Klasifikasi:\n", report)