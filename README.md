# 🚀 InfraCopilot AI  
### Predictive Maintenance & Incident Response for EV Charging Networks  

InfraCopilot AI is a full-stack platform that uses machine learning to **predict failures in EV charging infrastructure**, generate **copilot-style maintenance recommendations**, and **reduce costly downtime**.

Built for the **Data Pigeon AI Incident Response Challenge** at SacHacks.

---

## 🎯 Problem

EV charging networks face:
- Unexpected failures → downtime  
- Expensive reactive maintenance  
- Poor prioritization of repairs  
- Lost revenue from non-functional chargers  

Most systems react **after failures occur**.

---

## 🧠 Solution

InfraCopilot enables **proactive maintenance** by:

- Predicting failure probability for each charger  
- Identifying high-risk units in advance  
- Explaining *why* a failure may happen  
- Recommending actionable fixes  
- Quantifying potential cost savings  

---

---

## ⚙️ Tech Stack

### Machine Learning
- Python
- scikit-learn
- SMOTE (handling class imbalance)
- Logistic Regression & Random Forest
- Precision-Recall optimization
- Cost-aware threshold tuning

### Backend
- FastAPI
- Pandas
- Uvicorn

### Frontend
- Next.js (App Router)
- TypeScript
- TailwindCSS

---

## 📊 Model Performance

- Dataset: 50,000 simulated chargers  
- Failure rate: ~3%  
- ROC-AUC: **0.99**  
- PR-AUC: **0.81**  
- Recall (failures caught): **90%+**  
- Precision: ~49%  
- Cost savings: **$300K+**


This ensures the model prioritizes **catching failures over avoiding false alarms**.

---

## 🔍 Features

### 📊 Fleet Dashboard
- Total chargers
- Risk distribution (Critical / Warning / Safe)
- Total projected savings

### 📋 Fleet Table
- Search by charger ID
- Filter by risk level
- Sort by savings or risk
- Pagination (handles 50K+ rows)

### 🤖 Copilot View
For each charger:
- Failure probability
- Root cause analysis
- Recommended action
- Time-to-failure estimate
- Estimated savings

---

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
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
└── README.md


