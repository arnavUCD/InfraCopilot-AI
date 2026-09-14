"""Train predictive maintenance model for EV charger failure detection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import joblib
import os

from data_generator import generate_charger_fleet


def train_model(fleet: pd.DataFrame, output_dir: str = '../backend/models_v5'):
    """
    Train ensemble model with SMOTE for class imbalance.
    
    Args:
        fleet: DataFrame with charger features and failure labels
        output_dir: Directory to save trained model and scaler
    
    Returns:
        Trained model and scaler
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature selection
    feature_columns = [
        'age_months', 'utilization_cycles', 'temp_variance',
        'voltage_stability', 'model_year'
    ]
    X = fleet[feature_columns]
    y = fleet['failure']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to training data
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_smote, y_train_smote)
    
    # Get Random Forest predictions for Logistic Regression
    rf_proba = rf_model.predict_proba(X_train_smote)[:, 1]
    
    # Train Logistic Regression on RF outputs
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(rf_proba.reshape(-1, 1), y_train_smote)
    
    # Evaluate on test set
    rf_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    lr_test_proba = lr_model.predict_proba(rf_test_proba.reshape(-1, 1))[:, 1]
    
    roc_auc = roc_auc_score(y_test, lr_test_proba)
    pr_auc = average_precision_score(y_test, lr_test_proba)
    
    print(f"\nModel Performance:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    # Predictions with threshold tuning
    threshold = 0.35
    predictions = (lr_test_proba > threshold).astype(int)
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_test, predictions))
    
    # Save models
    joblib.dump(rf_model, f'{output_dir}/rf_model.pkl')
    joblib.dump(lr_model, f'{output_dir}/lr_model.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    print(f"\nModels saved to {output_dir}/")
    
    return rf_model, lr_model, scaler


if __name__ == '__main__':
    print("Generating synthetic fleet...")
    fleet = generate_charger_fleet(n_chargers=50000)
    
    print("Training model...")
    train_model(fleet)
