# ANVAYA
### AI-Powered Pre-Delinquency Early Warning System for Retail Banking

<p align="center">
  Predict financial stress 14–30 days before EMI default using Explainable AI, Behavioral Analytics, Real-Time Streaming, and Agentic Interventions.
</p>

---

## Overview

ANVAYA is an enterprise-grade pre-delinquency early warning system designed for retail banking institutions. The system detects early financial stress signals before a customer misses their EMI payment, enabling banks to shift from reactive collections to proactive customer assistance.

Unlike traditional credit scoring systems that rely on historical credit bureau data, ANVAYA continuously analyzes behavioral patterns, repayment habits, cash flow irregularities, and transaction-level signals to identify customers at risk.

The platform combines Machine Learning, Explainable AI, Real-Time Streaming, and AI Agents to deliver actionable interventions before delinquency occurs.

---

## Problem Statement

Traditional credit risk systems are reactive.

Banks generally contact customers only after they miss EMI payments, resulting in:

- Higher recovery costs
- Customer dissatisfaction
- Increased defaults
- Delayed risk detection
- Poor visibility into changing financial behavior

Credit scores such as CIBIL or FICO are updated periodically and often fail to detect sudden cash-flow deterioration.

ANVAYA addresses this challenge by identifying early stress indicators 2–4 weeks before default.

---

## Solution

ANVAYA performs:

- Early financial stress detection.
- Personalized behavioral analysis.
- Explainable risk scoring.
- Real-time transaction monitoring.
- Automated intervention recommendations.
- AI-generated relationship manager summaries.
- Customer communication drafting.

---

# Key Features

### 1. Personal Rhythm Engine

Instead of comparing customers to population averages, ANVAYA compares customers against their own historical behavior.

This allows accurate assessment for:
- Gig workers
- Freelancers
- Farmers
- Contractors
- Salaried employees

---

### 2. Dual-Track Risk Engine

- XGBoost → Fast screening.
- LightGBM → Deep analysis.
- Meta Logistic Regression → Final prediction.
- Isotonic Regression → Probability calibration.

---

### 3. SHAP Explainability

Provides:
- Top 3 risk drivers.
- Contribution percentages.
- Plain-English explanations.
- Regulatory transparency.

---

### 4. Kafka Streaming

Real-time transaction processing using Apache Kafka.

Features:
- Event ingestion.
- Streaming risk scoring.
- Alert generation.
- Risk updates.

---

### 5. Stress Contagion Detector

Detects employer-level financial stress.

If multiple employees from the same company enter high-risk zones, the system escalates connected customers automatically.

---

### 6. LangGraph AI Agents

#### Agent 1: Risk Analyzer
- Generates RM summaries.

#### Agent 2: Action Agent
- Drafts personalized interventions.
- Creates customer messages.

---

# System Architecture

```text
Customer Transactions
         │
         ▼
Apache Kafka Streaming
         │
         ▼
Personal Rhythm Engine
         │
         ▼
WOE Transformation
         │
         ▼
XGBoost + LightGBM
         │
         ▼
Probability Calibration
         │
         ▼
SHAP Explainability
         │
         ▼
Intervention Engine
         │
         ▼
LangGraph Agents
         │
         ▼
FastAPI Backend
         │
         ▼
React Dashboard
```

---

# Tech Stack

| Category | Technologies |
|---------|-------------|
| Language | Python 3.10 |
| Machine Learning | XGBoost, LightGBM, Scikit-Learn |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Streaming | Apache Kafka |
| AI Agents | LangGraph, LangChain |
| Backend | FastAPI |
| Frontend | React 19, Vite |
| Visualization | Recharts |
| Styling | Tailwind CSS |
| Containerization | Docker |
| APIs | FastAPI REST |

---

# 13 Risk Features

| Feature | Description |
|--------|-------------|
| F1 | EMI to Income Ratio |
| F2 | Savings Drawdown |
| F3 | Income Arrival Irregularity |
| F4 | Spending Pattern Shift |
| F5 | Auto Debit Failure Count |
| F6 | Lending App Usage |
| F7 | Cash Hoarding Ratio |
| F8 | Stress Velocity |
| F9 | Payment Timing Entropy |
| F10 | Peer Cohort Stress |
| F11 | Overdraft Frequency |
| F12 | Cross Loan Consistency |
| F13 | Secondary Income Stability |

---

# Risk Bands

| Band | Probability of Default |
|------|-----------------------|
| GREEN | < 5% |
| YELLOW | 5–15% |
| HIGH | 15–25% |
| RED | > 25% |

---

# Model Performance

| Metric | Score |
|-------|--------|
| AUC-ROC | 0.9391 |
| KS Statistic | 74.18 |
| Gini Coefficient | 0.8782 |
| Precision | 0.7299 |
| Recall | 0.4739 |
| F1 Score | 0.5746 |

---

# Project Structure

```text
ANVAYA/
│
├── data/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineering.py
│   ├── personal_rhythm_engine.py
│   ├── woe_transformer.py
│   ├── model_trainer.py
│   ├── shap_explainer.py
│   ├── kafka_consumer.py
│   ├── kafka_producer.py
│   ├── stress_contagion.py
│   ├── intervention_engine.py
│   └── langgraph_agents.py
│
├── dashboard/
│   ├── frontend/
│   └── backend/
│
└── docker-compose.yml
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/ANVAYA.git
cd ANVAYA
```

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Start Kafka

```bash
docker-compose up -d
```

---

# Train Models

```bash
python src/main.py
```

---

# Start Streaming Engine

Terminal 1:

```bash
python src/kafka_consumer.py
```

Terminal 2:

```bash
python src/kafka_producer.py
```

---

# Start Backend

```bash
cd dashboard/backend
uvicorn main:app --reload --port 8000
```

---

# Start Frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

Visit:

```text
http://localhost:5173
```

---

# Dashboard Modules

- Portfolio Overview
- Customer Profile
- Live Risk Scoring
- Alerts Feed
- Intervention Tracker

---

# Business Impact

For a bank with 100,000 customers:

- Early risk detection.
- Reduced delinquency rates.
- Lower collection costs.
- Improved customer retention.
- Faster intervention decisions.

---

# Future Improvements

- Core Banking System Integration
- Continuous Model Retraining
- Cloud Deployment
- Kubernetes Support
- Conversational RM Assistant
- Multi-Tenant Architecture

---

# Resume Highlights

✔ Built a real-time banking risk prediction platform.

✔ Processed 50,000+ customer records.

✔ Developed 13 domain-specific risk features.

✔ Implemented explainable AI using SHAP.

✔ Integrated Kafka event streaming.

✔ Developed LangGraph multi-agent workflows.

✔ Built React + FastAPI production dashboard.

---

## License

This project is licensed under the MIT License.

---

### ANVAYA — Predicting financial stress before delinquency occurs.
