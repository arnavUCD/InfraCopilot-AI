# 🚀 InfraCopilot AI
### Predictive Maintenance & Incident Response for EV Charging Networks

**InfraCopilot AI** is a full-stack machine learning platform for **predictive maintenance** and **intelligent incident response** in EV charging infrastructure. Built end-to-end for the **Data Pigeon AI Incident Response Challenge** at **SacHacks 2025**.

---

## 🎯 Problem Statement

EV charging networks face critical operational challenges:
- **Unexpected failures** cause extended downtime
- **Reactive maintenance** is expensive and inefficient
- **Poor prioritization** of repairs across large fleets
- **Lost revenue** from offline chargers
- **No predictive insights** — systems fail without warning

### The Data
- **50,000+ chargers** across the network
- **~3% failure rate**, but concentrated in predictable failure modes
- **Rich telemetry**: utilization, temperature, voltage stability, age, location
- **Massive cost impact**: $50K per charger downtime, thousands of daily users affected

---

## 🧠 Solution: Proactive Maintenance via ML

InfraCopilot predicts charger failures **before they occur** by:

1. **Failure Prediction** — Identify high-risk units using Random Forest + Logistic Regression
2. **Root Cause Analysis** — Explain *why* a charger will fail (feature importance)
3. **Prioritized Recommendations** — Rank repairs by impact and urgency
4. **Cost Quantification** — Show potential savings per action
5. **Interactive Dashboard** — Fleet overview + detailed charger diagnostics

### Results
- **ROC-AUC: 0.99** | **PR-AUC: 0.81** | **Recall: 90%+** (catches failures)
- **Cost Avoidance: $300K+** from prevented downtime
- **False Alarm Rate: ~49%** precision (acceptable for maintenance prioritization)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Next.js + TypeScript + TailwindCSS)             │
│  ├─ Fleet Dashboard (KPIs, risk distribution)             │
│  ├─ Charger Table (search, filter, sort, paginate)        │
│  └─ Copilot Detail View (predictions + recommendations)   │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/JSON
┌──────────────────────▼──────────────────────────────────────┐
│  Backend (FastAPI + Python)                               │
│  ├─ /predict (batch charger inference)                    │
│  ├─ /charger/:id (detailed diagnostics)                   │
│  └─ /recommendations (maintenance ranking)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  ML Layer (scikit-learn + Pandas)                          │
│  ├─ Data Generator (simulate 50K chargers)                │
│  ├─ Model Training (Random Forest + Logistic Regression)  │
│  └─ Inference Engine (real-time predictions)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Machine Learning
- **Python 3.10+**
- **scikit-learn** (Random Forest, Logistic Regression, metrics)
- **pandas** (data manipulation)
- **imbalanced-learn** (SMOTE for class imbalance)
- **numpy** (numerical computing)

### Backend
- **FastAPI** (async REST API)
- **Uvicorn** (ASGI server)
- **Pydantic** (data validation)

### Frontend
- **Next.js 14** (React framework, App Router)
- **TypeScript** (type safety)
- **TailwindCSS v4** (styling)
- **Recharts** (data visualization)

---

## 📊 Model Overview

### Training Pipeline

```python
# Generate synthetic fleet data
from ml.data_generator import generate_charger_fleet
fleet = generate_charger_fleet(n_chargers=50000)

# Train model with class imbalance handling
from ml.train_model import train_model
model, scaler = train_model(fleet)

# Inference on new chargers
from ml.inference_engine import predict_failures
risks = predict_failures(fleet, model, scaler)
```

### Features
1. **Age** (months in service)
2. **Utilization** (daily charge cycles)
3. **Temperature** (operating temp variance)
4. **Voltage Stability** (supply consistency)
5. **Geographic Region** (climate/maintenance patterns)
6. **Model Year** (hardware generation)

### Class Distribution
- **Healthy** (97%): No failure within 30 days
- **At-Risk** (3%): Failure likely within 30 days

**Mitigation**: SMOTE oversampling + cost-aware threshold tuning

---

## 📈 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|------------------|
| **ROC-AUC** | 0.99 | Excellent discrimination |
| **PR-AUC** | 0.81 | Strong positive predictive value |
| **Recall** | 90%+ | Catches most failures |
| **Precision** | 49% | ~1 in 2 predictions correct |
| **False Positive Cost** | Low | Safe to over-predict |

**Trade-off Rationale**: Precision is lower because:
- **Cost of missing a failure** (lost revenue, user impact) >> **Cost of false alarm** (preventive maintenance)
- Maintenance teams prefer proactive inspections over reactive emergency response

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip / npm

### Setup

```bash
# Clone repository
git clone https://github.com/arnavUCD/InfraCopilot-AI.git
cd InfraCopilot-AI

# ML Setup
cd ml
pip install -r requirements.txt
python train_model.py  # Train on simulated data

# Backend Setup
cd ../backend
pip install -r requirements.txt
uvicorn main:app --reload  # Start FastAPI server on localhost:8000

# Frontend Setup
cd ../frontend
npm install
npm run dev  # Start Next.js dev server on localhost:3000
```

### First Run

1. **Visit** http://localhost:3000
2. **Browse Fleet Dashboard** — see risk distribution across 50K chargers
3. **Search a Charger ID** (e.g., "CHARGER_00042")
4. **View Predictions**:
   - Failure probability
   - Root cause features
   - Recommended actions
   - Potential cost savings

---

## 📂 Project Structure

```
InfraCopilot-AI/
├── ml/                          # Machine Learning Layer
│   ├── data_generator.py        # Generate synthetic fleet (50K chargers)
│   ├── train_model.py           # Training pipeline with SMOTE & cost-aware tuning
│   ├── inference_engine.py      # Real-time prediction on new chargers
│   └── requirements.txt
│
├── backend/                     # FastAPI Backend
│   ├── main.py                  # REST API endpoints
│   ├── models_v5/               # Serialized trained models
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   ├── outputs/                 # Prediction cache
│   └── requirements.txt
│
├── frontend/                    # Next.js Frontend
│   ├── app/                     # App Router pages
│   │   ├── page.tsx             # Dashboard home
│   │   └── charger/[id]/page.tsx # Charger detail view
│   ├── components/
│   │   ├── FleetDashboard.tsx
│   │   ├── ChargerTable.tsx
│   │   └── CopilotView.tsx
│   ├── lib/
│   │   └── api.ts               # API client
│   └── package.json
│
└── README.md
```

---

## 🔍 Key Features

### 📊 Fleet Dashboard
- **Total Chargers**: 50,000
- **Risk Distribution Pie Chart**: Critical / Warning / Safe
- **Total Projected Savings**: Cost avoidance from proactive repairs
- **Real-time Updates**: Refreshes every 30 seconds

### 📋 Charger Table
- **Search**: Filter by charger ID
- **Columns**: ID, Status, Failure Probability, Savings, Last Updated
- **Sort**: By probability, savings, or risk level
- **Pagination**: Handles 50K+ rows efficiently
- **Inline Status Badge**: Color-coded risk levels

### 🤖 Copilot Detail View (Per Charger)

```json
{
  "charger_id": "CHARGER_00042",
  "failure_probability": 0.87,
  "risk_level": "CRITICAL",
  "root_causes": {
    "high_utilization": 0.34,
    "temperature_variance": 0.28,
    "age": 0.25,
    "voltage_instability": 0.13
  },
  "recommended_action": "Replace power supply unit and thermal paste",
  "time_to_failure_days": 7,
  "estimated_savings": "$47,500",
  "priority_rank": 3
}
```

---

## 🔬 Model Details

### Algorithm: Random Forest + Logistic Regression Ensemble

**Why this combination?**
1. **Random Forest** captures non-linear patterns in charger degradation
2. **Logistic Regression** on top provides calibrated probabilities
3. **SMOTE** handles severe class imbalance (97% vs 3%)
4. **Cost-aware thresholding** optimizes for recall (catch failures)

### Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

SMOTE(
    k_neighbors=5,
    random_state=42
)

# Threshold tuned for 90%+ recall
threshold = 0.35  # Default confidence threshold
```

---

## 🎓 Training Data

**Generated Synthetically** to simulate real-world charger behavior:

```python
# Example charger feature distribution
Charger(
    age_months=randint(6, 120),           # 6 months to 10 years
    utilization_cycles=randint(100, 5000),  # Daily usage
    temp_variance=gauss(5, 2),            # °C std deviation
    voltage_stability=uniform(0.95, 1.0), # % consistency
    region_code=choice(['CA', 'TX', 'NY']),
    model_year=randint(2019, 2024)
)
```

**Failure modes simulated**:
- Thermal degradation (high temp variance → power supply failure)
- Electrical wear (high utilization → component fatigue)
- Age-related (>5 years → increased risk)

---

## 📈 Example Predictions

### Scenario 1: New Charger (Low Risk)
```
Charger ID: CHARGER_01000
Age: 2 months | Utilization: 400 cycles/day | Temp: 2.1°C variance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure Probability: 5%
Risk Level: ✅ SAFE
Recommended Action: Routine maintenance (quarterly)
Estimated Savings: $500 (preventive care)
```

### Scenario 2: Aging Charger (High Risk)
```
Charger ID: CHARGER_04521
Age: 7 years | Utilization: 4,800 cycles/day | Temp: 8.4°C variance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure Probability: 88%
Risk Level: 🔴 CRITICAL
Root Causes:
  • High Utilization (34%)
  • Temperature Variance (28%)
  • Age (25%)
  • Voltage Instability (13%)

Recommended Action: URGENT - Replace power supply and thermistor
Time to Failure: 6 days
Estimated Savings: $52,000 (avoid emergency downtime)
Priority Rank: 2 / 50000
```

---

## 🔐 API Endpoints

### GET `/api/fleet`
Returns aggregate fleet statistics.

```bash
curl http://localhost:8000/api/fleet
```

**Response**:
```json
{
  "total_chargers": 50000,
  "critical_count": 487,
  "warning_count": 1203,
  "safe_count": 48310,
  "total_estimated_savings": 12450000
}
```

### GET `/api/chargers`
Returns paginated charger list with predictions.

```bash
curl "http://localhost:8000/api/chargers?skip=0&limit=50&sort=-failure_probability"
```

### GET `/api/charger/{charger_id}`
Fetch detailed prediction for a specific charger.

```bash
curl http://localhost:8000/api/charger/CHARGER_00042
```

---

## 🛠 Development Guide

### Adding New Features

1. **New Feature in ML Model**
   ```python
   # In ml/data_generator.py
   charger['new_metric'] = some_value
   
   # In ml/train_model.py
   FEATURE_COLUMNS = [..., 'new_metric']
   ```

2. **Expose via API**
   ```python
   # In backend/main.py
   @app.get("/api/charger/{charger_id}")
   async def get_charger(charger_id: str):
       # Include new_metric in response
   ```

3. **Display in Frontend**
   ```tsx
   // In frontend/components/CopilotView.tsx
   <div>New Metric: {charger.new_metric}</div>
   ```

### Testing

```bash
# Test ML model
python ml/train_model.py --test

# Test API
pytest backend/tests/

# Test Frontend
cd frontend && npm test
```

---

## 📊 Results & Impact

### By the Numbers
- **50,000 chargers** monitored
- **90%+ recall** on failure prediction
- **$300K+ annual savings** from prevented downtime
- **6-30 day early warning** before failure
- **49% false alarm rate** (acceptable for proactive maintenance)

### Use Cases
1. **Fleet Operators**: Prioritize repair teams and supply orders
2. **Network Planners**: Identify geographic or model-year patterns
3. **Finance**: Quantify maintenance ROI and avoid emergency costs
4. **Customer Service**: Proactive notifications about charger maintenance

---

## 🚧 Future Enhancements

- [ ] Real hardware telemetry integration (replace synthetic data)
- [ ] Time-series LSTM for sequential failure pattern detection
- [ ] Multi-model ensemble (XGBoost, LightGBM)
- [ ] Automated repair scheduling API
- [ ] Mobile app for field technicians
- [ ] Anomaly detection for unexpected failure modes
- [ ] Cost optimization solver (repair vs. replacement decisions)

---

## 📜 License

MIT License © 2026 Arnav Sharma

---

## 🤝 Questions?

For questions or feature requests, open an issue on GitHub.

**Built for SacHacks 2025** — Data Pigeon AI Incident Response Challenge
