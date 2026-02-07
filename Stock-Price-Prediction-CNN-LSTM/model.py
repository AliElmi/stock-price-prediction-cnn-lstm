#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv('data/AMZN.csv')
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
ma_day = [10, 50, 100]
for ma in ma_day:
    data[f"MA_{ma}"] = data['Close'].rolling(ma).mean()

data['Daily_Return'] = data['Close'].pct_change()
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# -----------------------------
# 3. Prepare Data for Model
# -----------------------------
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

# -----------------------------
# 4. Build Model (CNN + BiLSTM)
# -----------------------------
model = tf.keras.Sequential()

model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=10,
    batch_size=40,
    shuffle=True,
    verbose=1
)

# -----------------------------
# 6. Save Model
# -----------------------------
model.save("model.h5")
print("Model saved as model.h5")
