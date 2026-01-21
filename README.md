# Credit Card Fraud Detection using Data Science

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.  
Due to the highly imbalanced nature of fraud datasets, special techniques like SMOTE and anomaly detection are used.

## 🔍 Problem Statement
Credit card fraud causes significant financial losses. The goal is to build a model that accurately detects fraudulent transactions while minimizing false positives.

## 📊 Dataset
- Source: Kaggle Credit Card Fraud Dataset
- Transactions: 284,807
- Fraud cases: 492
- Features: PCA-transformed + Amount, Time

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit (optional)

## 🧠 Models Used
- Logistic Regression
- Random Forest
- XGBoost

## 📈 Evaluation Metrics
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/model_training.py

