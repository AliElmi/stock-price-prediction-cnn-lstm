# 📈 Stock Price Prediction using CNN + BiLSTM

A deep learning model for forecasting stock prices using a hybrid architecture that combines Convolutional Neural Networks (CNN) and Bidirectional LSTM layers.  
This project uses historical Amazon (AMZN) stock data to predict future price movements.

---

## 📌 Overview

This project demonstrates a complete pipeline for stock price prediction:

- Data preprocessing and feature engineering  
- Sliding window sequence generation  
- A hybrid **CNN + BiLSTM** deep learning model  
- Model training and evaluation  
- Saving the trained model for future inference  

The full exploratory analysis, visualizations, and evaluation plots are available in the Jupyter Notebook.

---

## 📂 Project Structure

project/
│── notebook.ipynb          # Full research notebook with EDA & visualizations
│── model.py                # Clean executable training script
│── data/
│     └── AMZN.csv          # Dataset
│── saved_model/
│     └── model.h5          # Trained model
│── requirements.txt        # Dependencies
│── README.md               # Project documentation



---

## 🧠 Model Architecture

The model consists of:

### **1. TimeDistributed CNN layers**
- Extract local temporal patterns  
- Capture short-term fluctuations in stock prices  

### **2. MaxPooling layers**
- Downsample extracted features  
- Reduce noise and dimensionality  

### **3. Bidirectional LSTM layers**
- Learn long-term dependencies  
- Capture forward and backward temporal relationships  

### **4. Dense output layer**
- Predicts the normalized next-day price movement  

---

## 🚀 How to Run

### **1. Install dependencies**

pip install -r requirements.txt



2. Run the training script

python model.py


3. Output

    Trained model saved as:
saved_model/model.h5
    Training logs printed in the console



----
📊 Evaluation

The model is evaluated using:

    Mean Squared Error (MSE)

    Mean Absolute Error (MAE)

    R² Score

    Explained Variance

Visualizations of training curves and prediction results are available in notebook.ipynb.


---
📘 Notes

    The .ipynb file contains the full exploratory data analysis (EDA), visualizations, and evaluation plots.

    The model.py file is a clean, production-ready version of the training pipeline.

    The dataset (AMZN.csv) must be placed inside the data/ directory.


---
📬 Contact

If you have questions or want to discuss the project, feel free to reach out.

---

## 📄 License

MIT License

