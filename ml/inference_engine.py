"""Inference engine for real-time charger failure predictions."""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
import os


class InferenceEngine:
    """
    Load trained models and perform predictions on charger data.
    """
    
    def __init__(self, model_dir: str = '../backend/models_v5'):
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
        self.lr_model = joblib.load(f'{model_dir}/lr_model.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.threshold = 0.35
    
    def predict_batch(self, chargers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict failure probability for a batch of chargers.
        
        Args:
            chargers_df: DataFrame with charger features
        
        Returns:
            DataFrame with predictions and risk levels
        """
        feature_columns = [
            'age_months', 'utilization_cycles', 'temp_variance',
            'voltage_stability', 'model_year'
        ]
        
        X = chargers_df[feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Random Forest -> Logistic Regression ensemble
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        predictions = self.lr_model.predict_proba(rf_proba.reshape(-1, 1))[:, 1]
        
        # Determine risk level
        risk_levels = []
        for prob in predictions:
            if prob > 0.7:
                risk_levels.append('CRITICAL')
            elif prob > 0.5:
                risk_levels.append('WARNING')
            else:
                risk_levels.append('SAFE')
        
        # Add results to dataframe
        results = chargers_df.copy()
        results['failure_probability'] = predictions
        results['risk_level'] = risk_levels
        results['estimated_savings'] = (predictions * 50000).round(0)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from Random Forest.
        
        Returns:
            Dictionary of feature names and importance scores
        """
        feature_names = [
            'age_months', 'utilization_cycles', 'temp_variance',
            'voltage_stability', 'model_year'
        ]
        importances = self.rf_model.feature_importances_
        return dict(zip(feature_names, importances))


if __name__ == '__main__':
    from data_generator import generate_charger_fleet
    
    print("Loading inference engine...")
    engine = InferenceEngine()
    
    print("Generating test chargers...")
    test_chargers = generate_charger_fleet(n_chargers=100)
    
    print("Running predictions...")
    predictions = engine.predict_batch(test_chargers)
    
    print(f"\nPredictions (first 10):")
    print(predictions[['charger_id', 'failure_probability', 'risk_level', 'estimated_savings']].head(10))
    
    print(f"\nFeature Importance:")
    importance = engine.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
