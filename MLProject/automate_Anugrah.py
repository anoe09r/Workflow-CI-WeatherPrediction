import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("weather_raw.csv")

print("Duplikasi:", data.duplicated().sum())
print("Missing values:\n", data.isna().sum())

data = data.dropna().reset_index(drop=True)

z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)
print(f"Jumlah outlier yang dihapus: {outliers.sum()}")
data = data[~outliers].reset_index(drop=True)

# Encoding
label_encoder = LabelEncoder()
data["rain_next_6h"] = label_encoder.fit_transform(data["rain_next_6h"])

features = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'rain', 'rain_sum_next_6h']
X = data[features]
y = data["rain_next_6h"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data["rain_next_6h"] = y.values

processed_data.to_csv("MLProject/weather_preprocessing.csv", index=False)
