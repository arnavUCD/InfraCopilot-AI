"""Generate synthetic EV charger fleet data for training and inference."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_charger_fleet(n_chargers: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic charger fleet data.
    
    Args:
        n_chargers: Number of chargers to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with charger features and failure labels
    """
    np.random.seed(random_state)
    
    # Generate features
    age_months = np.random.randint(6, 121, n_chargers)
    utilization_cycles = np.random.randint(100, 5001, n_chargers)
    temp_variance = np.abs(np.random.normal(5, 2, n_chargers))
    voltage_stability = np.random.uniform(0.90, 1.0, n_chargers)
    region_code = np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_chargers)
    model_year = np.random.randint(2019, 2025, n_chargers)
    
    # Generate failure label based on feature interactions
    # Higher risk with: high age, high utilization, high temp variance, low voltage stability
    failure_score = (
        (age_months / 120) * 0.35 +
        (utilization_cycles / 5000) * 0.30 +
        (temp_variance / 10) * 0.25 +
        (1 - voltage_stability) * 10 * 0.10
    )
    
    # Add some random noise and threshold at ~3% failure rate
    noise = np.random.normal(0, 0.15, n_chargers)
    failure_probability = 1 / (1 + np.exp(-(failure_score + noise - 1.2)))
    failures = (failure_probability > 0.3).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'charger_id': [f'CHARGER_{i:05d}' for i in range(n_chargers)],
        'age_months': age_months,
        'utilization_cycles': utilization_cycles,
        'temp_variance': temp_variance,
        'voltage_stability': voltage_stability,
        'region_code': region_code,
        'model_year': model_year,
        'failure_probability': failure_probability,
        'failure': failures
    })
    
    return df


if __name__ == '__main__':
    # Generate sample fleet
    fleet = generate_charger_fleet(n_chargers=50000)
    print(f"Generated fleet of {len(fleet)} chargers")
    print(f"Failure rate: {fleet['failure'].mean():.2%}")
    print(f"\nFeature summary:\n{fleet.describe()}")
