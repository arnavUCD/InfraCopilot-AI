"""FastAPI backend for InfraCopilot AI."""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.data_generator import generate_charger_fleet
from ml.inference_engine import InferenceEngine

app = FastAPI(title="InfraCopilot AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
engine = InferenceEngine(model_dir=os.path.join(os.path.dirname(__file__), 'models_v5'))

# Generate fleet once at startup
fleet_df = generate_charger_fleet(n_chargers=50000)
predictions_df = engine.predict_batch(fleet_df)


class ChargerResponse(BaseModel):
    charger_id: str
    age_months: int
    utilization_cycles: int
    temp_variance: float
    voltage_stability: float
    failure_probability: float
    risk_level: str
    estimated_savings: float


class FleetStatsResponse(BaseModel):
    total_chargers: int
    critical_count: int
    warning_count: int
    safe_count: int
    total_estimated_savings: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/fleet", response_model=FleetStatsResponse)
async def get_fleet_stats():
    """Get aggregate fleet statistics."""
    critical = (predictions_df['risk_level'] == 'CRITICAL').sum()
    warning = (predictions_df['risk_level'] == 'WARNING').sum()
    safe = (predictions_df['risk_level'] == 'SAFE').sum()
    total_savings = predictions_df['estimated_savings'].sum()
    
    return FleetStatsResponse(
        total_chargers=len(predictions_df),
        critical_count=int(critical),
        warning_count=int(warning),
        safe_count=int(safe),
        total_estimated_savings=float(total_savings)
    )


@app.get("/api/chargers")
async def get_chargers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    sort: str = Query('-failure_probability')
):
    """Get paginated charger predictions."""
    df = predictions_df.copy()
    
    # Sort
    sort_col = sort.lstrip('-')
    sort_asc = not sort.startswith('-')
    df = df.sort_values(by=sort_col, ascending=sort_asc)
    
    # Paginate
    df = df.iloc[skip:skip+limit]
    
    return {
        "chargers": df[['charger_id', 'failure_probability', 'risk_level', 'estimated_savings']].to_dict(orient='records'),
        "total": len(predictions_df),
        "skip": skip,
        "limit": limit
    }


@app.get("/api/charger/{charger_id}", response_model=ChargerResponse)
async def get_charger_detail(charger_id: str):
    """Get detailed prediction for a specific charger."""
    charger = predictions_df[predictions_df['charger_id'] == charger_id].iloc[0]
    
    return ChargerResponse(
        charger_id=charger['charger_id'],
        age_months=int(charger['age_months']),
        utilization_cycles=int(charger['utilization_cycles']),
        temp_variance=float(charger['temp_variance']),
        voltage_stability=float(charger['voltage_stability']),
        failure_probability=float(charger['failure_probability']),
        risk_level=charger['risk_level'],
        estimated_savings=float(charger['estimated_savings'])
    )


@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance scores from the trained model."""
    importance = engine.get_feature_importance()
    return {feature: float(score) for feature, score in importance.items()}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
