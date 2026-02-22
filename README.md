🚀 InfraCopilot AI
Intelligent Predictive Maintenance & Incident Response for EV Charging Networks

InfraCopilot AI is a full-stack predictive maintenance platform designed to improve reliability in mission-critical EV charging infrastructure.

Built for the Data Pigeon AI Incident Response Challenge, the system uses machine learning to proactively detect charger failures, generate actionable maintenance recommendations, and quantify real business impact through cost savings.

🎯 Problem

EV charging networks face:

Unexpected downtime

Expensive reactive maintenance

Lost revenue from failed chargers

Poor prioritization of field technicians

Traditional systems detect failures after they happen.

InfraCopilot detects them before they happen.

🧠 Solution

InfraCopilot AI combines:

⚡ Predictive failure modeling

📊 Fleet-wide risk scoring

🤖 Copilot-style maintenance recommendations

💰 Cost-optimized decision thresholds

🌐 Real-time web dashboard

The system scores every charger in a fleet and identifies:

🔴 Critical units needing immediate intervention

🟡 Warning units that should be scheduled soon

🟢 Healthy chargers

Each flagged charger includes:

Failure probability

Top contributing risk factors

Recommended action

Estimated downtime savings

🏗️ Architecture
ML Pipeline (Python, scikit-learn)
        ↓
Model + Fleet Recommendations (CSV/JSON)
        ↓
FastAPI Backend
        ↓
Next.js Frontend Dashboard
Stack

Machine Learning

Python

scikit-learn

SMOTE (class imbalance handling)

Logistic Regression

Random Forest

Cost-weighted threshold optimization

Precision-Recall analysis

Business impact modeling

Backend

FastAPI

Pandas

Uvicorn

Frontend

Next.js (App Router)

TypeScript

TailwindCSS

API proxy routing

📊 Model Highlights

50,000 simulated chargers

3% failure rate

ROC-AUC: 0.99

PR-AUC: 0.81

Failure Recall: 90%+

Precision: ~49%

Cost savings: $300K+ on test fleet

Real-time fleet scoring in <3 seconds

Threshold is optimized using:

Minimize (False Negatives × $1200) + (False Positives × $50)

This directly aligns model performance with real business impact.

🔍 Core Features
1️⃣ Fleet Dashboard

Risk distribution

Critical / Warning / Safe breakdown

Total projected savings

2️⃣ Fleet Table

Pagination & filtering

Search by charger ID

Sort by risk or savings

Action recommendations inline

3️⃣ Charger Copilot View

Risk probability

Confidence level

Root cause breakdown

Maintenance instructions

Time-to-failure estimate

💰 Business Impact

Without predictive model:

All failures occur → full downtime cost

With InfraCopilot:

High-risk chargers proactively repaired

Missed failures minimized

Downtime reduced significantly

Example test results:

$360,000 potential downtime

Reduced to ~$47,500

$312,500 savings

🧪 How to Run Locally
1️⃣ Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export FLEET_CSV_PATH=outputs/fleet_recommendations.csv
uvicorn backend.main:app --reload --port 8000

Test:

http://127.0.0.1:8000/health
2️⃣ Frontend
cd frontend
npm install
npm run dev

Visit:

http://localhost:3000
📁 Repository Structure
infra-copilot/
├── backend/
│   ├── main.py
│   ├── models_v5/
│   ├── outputs/
│   └── requirements.txt
├── ml/
│   ├── data_generator.py
│   ├── train_model.py
│   ├── inference_engine.py
│   └── notebooks/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
└── README.md
🌍 Why This Matters

Infrastructure downtime is expensive.

EV adoption depends on reliability.

InfraCopilot moves the system from:

Reactive maintenance
to
Proactive intelligence

🏆 Judging Alignment

Innovation

Cost-aware threshold optimization

Copilot-style explanation layer

Business-aligned ML metrics

Impact

Quantified dollar savings

Direct downtime reduction

Feasibility

FastAPI production-ready backend

Scalable fleet scoring

Modular ML pipeline

Technical Depth

Imbalanced classification

PR-AUC optimization

Calibration

Cross-validation

Model comparison
