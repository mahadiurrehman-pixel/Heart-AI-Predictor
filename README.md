# ❤️ Heart-AI-Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KNN-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> A Machine Learning based Heart Disease Risk Prediction System that analyzes clinical patient data and predicts heart disease risk using a K-Nearest Neighbors (KNN) classifier with a Streamlit web interface.

---

## 🚀 Overview

Heart-AI-Predictor is an end-to-end Machine Learning project that demonstrates the complete ML workflow:

```
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis (EDA)
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Streamlit Deployment
```

---

## 📸 Application Preview

The application allows users to enter patient health information and receive an ML-based risk prediction.

Features:

- Patient input interface
- Real-time prediction
- Risk probability estimation
- Interactive Streamlit dashboard

---

# ✨ Features

✅ Machine Learning based prediction  
✅ KNN Classification Algorithm  
✅ Feature preprocessing pipeline  
✅ StandardScaler integration  
✅ One-hot encoding support  
✅ Interactive Streamlit web application  
✅ Real-time heart disease risk analysis  

---

# 🧠 Machine Learning Model

| Component | Details |
|---|---|
| Algorithm | K-Nearest Neighbors (KNN) |
| Problem Type | Binary Classification |
| Target Variable | HeartDisease |
| Output | 0 = Low Risk, 1 = High Risk |
| Feature Processing | One-Hot Encoding |
| Scaling | StandardScaler |

---

# 📊 Dataset

Dataset used:

**Heart Failure Prediction Dataset**

Source:
Kaggle Heart Disease Dataset

The dataset contains clinical attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

---

# 🩺 Input Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Gender |
| Chest Pain Type | Type of chest pain |
| Resting BP | Resting blood pressure |
| Cholesterol | Cholesterol level |
| Fasting BS | Blood sugar level |
| Resting ECG | ECG result |
| Max HR | Maximum heart rate |
| Exercise Angina | Exercise induced angina |
| Oldpeak | ST depression value |
| ST Slope | Slope of ST segment |

---

# 🛠️ Tech Stack

```
Python
│
├── Pandas          → Data Processing
├── NumPy           → Numerical Operations
├── Scikit-learn    → Machine Learning
├── Joblib          → Model Saving
└── Streamlit       → Web Application
```

---

# 📂 Project Structure

```
Heart-AI-Predictor/

│
├── app.py                 # Streamlit application
│
├── KNN_heart.pkl    # Trained KNN model
├── scaler.pkl       # Feature scaler
├── columns.pkl      # Feature columns
│
├── requirements.txt       # Dependencies
│
├── README.md              # Documentation
│
└── notebook/
    └── cleaning_model.ipynb     # Model training notebook
```

---

# ⚙️ Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Heart-AI-Predictor.git

cd CardioRisk-ML
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Application

```bash
streamlit run app.py
```

Application will open:

```
http://localhost:8501
```

---

# 📦 Requirements

`requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

# 🏋️ Model Training

The model training pipeline includes:

- Data preprocessing
- Encoding categorical variables
- Splitting training/testing data
- Feature scaling
- KNN model training
- Model evaluation
- Saving trained files using Joblib

Example:

```python
model.fit(X_train_scaled, y_train)

joblib.dump(model, "knn_heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(columns, "heart_columns.pkl")
```

---

# 📈 Model Evaluation

Evaluation metrics used:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# ⚠️ Disclaimer

This project is created for **educational purposes only**.

It is not a replacement for professional medical advice, diagnosis, or treatment.

Always consult qualified healthcare professionals for medical decisions.

---

# 👨‍💻 Developer

## Mahadi 🚀

Built with ❤️ using:

- Python
- Scikit-learn
- Streamlit
- Machine Learning

---

# 📄 License

MIT License

Free to use, modify, and distribute.

---

⭐ If you like this project, consider giving it a star!
