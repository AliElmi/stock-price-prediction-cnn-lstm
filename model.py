import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten

# ---------------------------------------------------------
# Create required folders
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
data = pd.read_csv("data/AMZN.csv")
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
ma_day = [10, 50, 100]
for ma in ma_day:
    data[f"MA_{ma}"] = data["Close"].rolling(ma).mean()

data["Daily_Return"] = data["Close"].pct_change()
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------
# 3. Prepare Data for Model
# ---------------------------------------------------------
df = data.copy()
X, Y = [], []
window_size = 100

for i in range(1, len(df) - window_size - 1):
    first = df.iloc[i, 2]
    temp = [(df.iloc[i + j, 2] - first) / first for j in range(window_size)]
    target = (df.iloc[i + window_size, 2] - first) / first
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array([target]).reshape(1, 1))

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(train_X).reshape(-1, 1, 100, 1)
test_X = np.array(test_X).reshape(-1, 1, 100, 1)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

# ---------------------------------------------------------
# 4. Build Model (CNN + BiLSTM)
# ---------------------------------------------------------
model = tf.keras.Sequential()

model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation="relu", input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation="relu")))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation="relu")))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))

model.add(Dense(1, activation="linear"))

model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])

# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=10,
    batch_size=40,
    shuffle=True,
    verbose=1
)

# ---------------------------------------------------------
# 6. Save Model
# ---------------------------------------------------------
model.save("saved_model/model.h5")
print("Model saved successfully.")

# ---------------------------------------------------------
# 7. Predictions on Test Set
# ---------------------------------------------------------
predictions = model.predict(test_X).reshape(-1)
actual = np.array(test_Y).reshape(-1)

# Save predictions to CSV
df_out = pd.DataFrame({"Actual": actual, "Predicted": predictions})
df_out.to_csv("results/predictions.csv", index=False)

# ---------------------------------------------------------
# 8. Plot Loss Curve
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("results/loss_curve.png")
plt.close()

# ---------------------------------------------------------
# 9. Plot Actual vs Predicted
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(actual[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.title("Actual vs Predicted (First 200 Samples)")
plt.xlabel("Sample")
plt.ylabel("Normalized Price Change")
plt.legend()
plt.savefig("results/actual_vs_predicted.png")
plt.close()

# ---------------------------------------------------------
# 10. Zoom-in Plot
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(actual[50:120], label="Actual")
plt.plot(predictions[50:120], label="Predicted")
plt.title("Zoomed Actual vs Predicted (Samples 50–120)")
plt.xlabel("Sample")
plt.ylabel("Normalized Price Change")
plt.legend()
plt.savefig("results/zoom_plot.png")
plt.close()

print("All results saved in /results/")
