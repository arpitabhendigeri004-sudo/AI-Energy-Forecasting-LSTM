import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

from src.preprocess import load_data, scale_data, create_sequences
from src.train import build_model, train_model
from src.predict import predict

# ================= STEP 1: LOAD DATA =================
data = load_data()
print("✅ Data loaded")

# ================= STEP 2: SCALE DATA =================
scaled_data, scaler = scale_data(data)

# ================= STEP 3: CREATE SEQUENCES =================
X, y = create_sequences(scaled_data)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# ================= STEP 4: SPLIT DATA =================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ================= STEP 5: BUILD MODEL =================
model = build_model((X_train.shape[1], 1))

# ================= STEP 6: TRAIN MODEL =================
print("🚀 Training started...")
train_model(model, X_train, y_train)

# ================= STEP 7: PREDICT =================
predictions = predict(model, X_test)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# ================= STEP 8: EVALUATION =================
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📊 Model Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# ================= STEP 9: SAVE MODEL =================
print("\n💾 Saving model...")

os.makedirs("models", exist_ok=True)

model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved!")

# ================= STEP 10: SAVE GRAPH =================
os.makedirs("images", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Energy Forecasting")

plt.savefig("images/final_output.png")   # 🔥 NO plt.show()

print("📈 Graph saved in images folder")

print("\n🎉 EVERYTHING COMPLETED SUCCESSFULLY!")