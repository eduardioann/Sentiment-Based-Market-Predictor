import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from model import add_technical_indicators, add_sentiment_features
import joblib
import pickle

df = pd.read_csv('data/processed/tesla_lagged.csv')
df['date'] = pd.to_datetime(df['date'])

df = add_technical_indicators(df)
df = add_sentiment_features(df)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

for col in feature_columns:
    if col not in df.columns:
        df[col] = 0

df = df[feature_columns]

sequence_length = 10

missing = [col for col in feature_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing features in new data: {missing}")

date_col = 'date'
numerical_features = [col for col in feature_columns if col != date_col]

df[date_col] = df[date_col].astype(np.int64) // 10**9

try:
    scaler = joblib.load('model/feature_scaler.joblib')
    print("Loaded saved scaler from training.")
except FileNotFoundError:
    raise FileNotFoundError("Scalerul salvat nu a fost găsit! Rulează antrenarea modelului pentru a genera 'feature_scaler.joblib'.")

scaled_features = scaler.transform(df[feature_columns])

X_new = np.array([scaled_features[-sequence_length:]])
print("X_new shape:", X_new.shape)

models = []
for fold in range(1, 6):
    model_path = f'best_model_fold_{fold}.h5'
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    models.append(model)

predictions = []
for model in models:
    pred = model.predict(X_new)
    predictions.append(pred)

ensemble_prediction = np.mean(predictions, axis=0)

price_5d_pred = np.clip(ensemble_prediction[0][0][0], -0.15, 0.15)
price_7d_pred = np.clip(ensemble_prediction[1][0][0], -0.15, 0.15)
direction_pred = ensemble_prediction[2][0][0]
direction_7d_pred = ensemble_prediction[3][0][0]

current_price = df['Close'].iloc[-1]

predicted_price_5d = current_price * (1 + price_5d_pred)
predicted_price_7d = current_price * (1 + price_7d_pred)

print("\nPrediction Results:")
print(f"Current Price: ${current_price:.2f}")
print("\n5-Day Prediction:")
print(f"Predicted Price Change: {price_5d_pred:.2%}")
print(f"Predicted Price: ${predicted_price_5d:.2f}")
print(f"Absolute Change: ${predicted_price_5d - current_price:.2f}")

print("\n7-Day Prediction:")
print(f"Predicted Price Change: {price_7d_pred:.2%}")
print(f"Predicted Price: ${predicted_price_7d:.2f}")
print(f"Absolute Change: ${predicted_price_7d - current_price:.2f}")

print("\nDirection Prediction (5 days):")
print(f"Probability of Price Increase in 5 days: {direction_pred:.2%}")
print(f"Predicted Direction in 5 days: {'Up' if direction_pred > 0.5 else 'Down'}")

print("\nDirection Prediction (7 days):")
print(f"Probability of Price Increase in 7 days: {direction_7d_pred:.2%}")
print(f"Predicted Direction in 7 days: {'Up' if direction_7d_pred > 0.5 else 'Down'}")

if abs(price_5d_pred) > 0.05 or abs(price_7d_pred) > 0.05:
    print("\n⚠️ Warning: These predictions show significant price movements!")
    print("Please verify the model's performance and consider these predictions with caution!")