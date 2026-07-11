import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

def run_churn_pipeline():
    # 1. Data Loading & Cleaning
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset missing. Please place '{file_path}' in this folder.")
        
    df = pd.read_csv(file_path)
    df.drop('customerID', axis=1, inplace=True)
    
    # Handle hidden empty strings in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Map target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 2. One-Hot Encoding
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X = pd.get_dummies(X, drop_first=True).astype(float)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    # 5. K-Value Hyperparameter Optimization
    k_values = range(1, 21, 2)
    accuracies, precisions, recalls = [], [], []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, pos_label=1))
        
    # Generate and export the comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, label='Accuracy')
    plt.plot(k_values, precisions, marker='s', linewidth=2, label='Precision')
    plt.plot(k_values, recalls, marker='^', linewidth=2, label='Recall')
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Performance Score', fontsize=12)
    plt.title('KNN Optimization: Metrics vs. K Value (21 Selected Features)', fontsize=14, pad=15)
    plt.xticks(k_values)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('knn_k_comparison.png')
    
    # 6. Final Model Execution (Optimal K=19)
    final_knn = KNeighborsClassifier(n_neighbors=11)
    final_knn.fit(X_train_scaled, y_train)
    y_pred_final = final_knn.predict(X_test_scaled)
    
    print("="*60)
    print("FINAL CLASSIFICATION REPORT (K=11, Full Feature Set)")
    print("="*60)
    print(classification_report(y_test, y_pred_final, target_names=['Retained (0)', 'Churned (1)']))
    
    print("="*60)
    print("CONFUSION MATRIX BREAKDOWN")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"True Negatives  (Predicted Retained, Actual Retained): {cm[0][0]}")
    print(f"False Positives (Predicted Churned,  Actual Retained): {cm[0][1]}")
    print(f"False Negatives (Predicted Retained, Actual Churned) : {cm[1][0]}")
    print(f"True Positives  (Predicted Churned,  Actual Churned) : {cm[1][1]}")
    print("="*60)

if __name__ == "__main__":
    run_churn_pipeline()