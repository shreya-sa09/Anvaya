from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# P1: Windows Path Escape Fix
BASE = r"C:\Project\My_Anvaya\data\processed/"

# P2: Load all CSVs once at startup (module-level cached dataframes)
try:
    PRED_DF = pd.read_csv(BASE + "model_predictions.csv")
    PRED_DF['SK_ID_CURR'] = PRED_DF['SK_ID_CURR'].astype(int)
except Exception as e:
    print(f"Error loading model_predictions.csv: {e}")
    PRED_DF = pd.DataFrame()

try:
    SHAP_DF = pd.read_csv(BASE + "shap_explanations.csv")
    SHAP_DF['SK_ID_CURR'] = SHAP_DF['SK_ID_CURR'].astype(int)
except Exception as e:
    print(f"Error loading shap_explanations.csv: {e}")
    SHAP_DF = pd.DataFrame()

try:
    AGENT_DF = pd.read_csv(BASE + "agent_outputs.csv")
    AGENT_DF['SK_ID_CURR'] = AGENT_DF['SK_ID_CURR'].astype(int)
except Exception as e:
    print(f"Error loading agent_outputs.csv: {e}")
    AGENT_DF = pd.DataFrame()

try:
    FEAT_DF = pd.read_csv(BASE + "features_final.csv")
    FEAT_DF['SK_ID_CURR'] = FEAT_DF['SK_ID_CURR'].astype(int)
except Exception as e:
    print(f"Error loading features_final.csv: {e}")
    FEAT_DF = pd.DataFrame()

try:
    MASTER_DF = pd.read_csv(BASE + "master.csv")
    # Cast to float first, then int to handle any decimal customer IDs
    MASTER_DF['SK_ID_CURR'] = MASTER_DF['SK_ID_CURR'].astype(float).astype(int)
except Exception as e:
    print(f"Error loading master.csv: {e}")
    MASTER_DF = pd.DataFrame()


# P16: Preserve all original risk bands: RED, HIGH, YELLOW, GREEN. Do not map HIGH to YELLOW.
def remap_band(band):
    return str(band).strip().upper()


@app.get("/portfolio")
def get_portfolio():
    if PRED_DF.empty:
        return {"total": 0, "critical": 0, "high": 0, "yellow": 0, "green": 0, "flagged": 0, "avg_pd": 0.0}

    pred = PRED_DF.copy()
    pred['risk_band'] = pred['risk_band'].apply(remap_band)

    total = len(pred)
    critical = len(pred[pred['risk_band'] == 'RED'])
    high = len(pred[pred['risk_band'] == 'HIGH'])
    yellow = len(pred[pred['risk_band'] == 'YELLOW'])
    green = len(pred[pred['risk_band'] == 'GREEN'])
    avg_pd = round(pred['final_pd'].mean() * 100, 1)

    return {
        "total": total,
        "critical": critical,
        "high": high,
        "yellow": yellow,
        "green": green,
        # P10: flagged today is RED + HIGH + YELLOW band count
        "flagged": critical + high + yellow,
        "avg_pd": avg_pd
    }


@app.get("/customers")
def get_customers(band: str = "All", limit: int = 50):
    if PRED_DF.empty:
        return []

    pred = PRED_DF.copy()
    shap = SHAP_DF.copy()

    pred['risk_band'] = pred['risk_band'].apply(remap_band)
    shap['risk_band'] = shap['risk_band'].apply(remap_band)

    if band != "All":
        pred = pred[pred['risk_band'] == band.upper()]
        order = {'RED': 0, 'HIGH': 1, 'YELLOW': 2, 'GREEN': 3}
        pred['_s'] = pred['risk_band'].map(order)
        pred = pred.sort_values('_s').drop(columns='_s')
    else:
        pred = pred.sort_values('SK_ID_CURR')

    pred = pred.head(limit)

    merged = pred.merge(
        shap[['SK_ID_CURR', 'top_driver_1_label']],
        on='SK_ID_CURR',
        how='left'
    )

    merged['top_driver_1_label'] = merged['top_driver_1_label'].fillna('-')

    return merged[[
        'SK_ID_CURR', 'risk_band',
        'final_pd', 'top_driver_1_label'
    ]].to_dict(orient='records')


@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    if PRED_DF.empty:
        return {"error": "Customer not found"}

    pred = PRED_DF.copy()
    shap = SHAP_DF.copy()

    pred['risk_band'] = pred['risk_band'].apply(remap_band)
    shap['risk_band'] = shap['risk_band'].apply(remap_band)

    pred_row = pred[pred['SK_ID_CURR'] == customer_id]
    if pred_row.empty:
        return {"error": "Customer not found"}

    result = pred_row.iloc[0].to_dict()

    shap_row = shap[shap['SK_ID_CURR'] == customer_id]
    if not shap_row.empty:
        result.update(shap_row.iloc[0].to_dict())

    # AGENT Message outputs check
    if not AGENT_DF.empty:
        agent_row = AGENT_DF[AGENT_DF['SK_ID_CURR'] == customer_id]
        if not agent_row.empty:
            result['customer_message'] = agent_row['customer_message'].iloc[0]

    # P3: Dynamic 90-day Trend usingxgb_score, lgb_score, and final_pd
    xgb = float(result.get('xgb_score', 0.0))
    lgb = float(result.get('lgb_score', 0.0))
    fpd = float(result.get('final_pd', 0.0))
    result['pd_trend_90d'] = [
        {"day": "90d Ago", "pd": round(xgb * 0.8 * 100, 1)},
        {"day": "60d Ago", "pd": round(lgb * 0.9 * 100, 1)},
        {"day": "30d Ago", "pd": round(((xgb + lgb) / 2) * 100, 1)},
        {"day": "Current", "pd": round(fpd * 100, 1)}
    ]

    # P4: Payment History (6m bar chart) derived from F9 timing entropy
    entropy = 0.0
    if not FEAT_DF.empty:
        feat_row = FEAT_DF[FEAT_DF['SK_ID_CURR'] == customer_id]
        if not feat_row.empty:
            if 'F9_payment_timing_entropy_WOE' in feat_row.columns:
                entropy = float(feat_row['F9_payment_timing_entropy_WOE'].iloc[0])

    entropy_shifted = max(0.0, entropy + 2.0)
    relative_months = ["5m Ago", "4m Ago", "3m Ago", "2m Ago", "1m Ago", "Current"]
    multipliers = [1.2, 1.5, 1.8, 2.2, 2.6, 3.0]
    result['payment_history_6m'] = [
        {"month": relative_months[i], "delay_days": max(0, int(entropy_shifted * multipliers[i] + fpd * 10.0))}
        for i in range(6)
    ]

    # P5: Dynamic Rhythm Deviation (baseline vs current for top 3 SHAP)
    rhythm = []
    for i in range(1, 4):
        label = result.get(f'top_driver_{i}_label')
        pct = result.get(f'top_driver_{i}_shap_contribution_pct')
        if label and pd.notna(label) and label != '-' and pct and pd.notna(pct):
            curr = float(pct)
            base = max(5.0, round(curr * (1.0 - fpd), 1))
            rhythm.append({
                "label": label,
                "base": base,
                "current": curr
            })
    result['rhythm_deviation'] = rhythm

    # P15: Dynamic Stress Velocity direction
    if fpd > 0.20:
        result['stress_velocity'] = 'steep'
    elif fpd > 0.10:
        result['stress_velocity'] = 'gradual'
    else:
        result['stress_velocity'] = 'flat'

    # Retrieve matching organization type from master if available for cohort details
    result['cohort'] = "General Retail Banking Cohort"
    if not MASTER_DF.empty:
        master_row = MASTER_DF[MASTER_DF['SK_ID_CURR'] == customer_id]
        if not master_row.empty:
            org = master_row['ORGANIZATION_TYPE'].iloc[0]
            if pd.notna(org):
                result['cohort'] = f"Cohort: Employer Group #{int(org)}"

    return result


@app.get("/alerts")
def get_alerts():
    if SHAP_DF.empty:
        return []

    shap = SHAP_DF.copy()
    shap['risk_band'] = shap['risk_band'].apply(remap_band)

    # Return alerts in RED or HIGH risk bands
    flagged = shap[shap['risk_band'].isin(['RED', 'HIGH'])].copy()
    flagged = flagged.sort_values('SK_ID_CURR').head(100)

    return flagged[[
        'SK_ID_CURR', 'risk_band', 'final_pd',
        'top_driver_1_label',
        'top_driver_1_explanation',
        'recommended_intervention',
        'case_summary'
    ]].to_dict(orient='records')


@app.get("/interventions")
def get_interventions():
    if SHAP_DF.empty:
        return []

    shap = SHAP_DF.copy()
    shap['risk_band'] = shap['risk_band'].apply(remap_band)

    if not AGENT_DF.empty:
        merged = shap.merge(
            AGENT_DF[['SK_ID_CURR', 'customer_message']],
            on='SK_ID_CURR',
            how='left'
        )
    else:
        merged = shap.copy()
        merged['customer_message'] = ''

    merged['customer_message'] = merged['customer_message'].fillna('')

    return merged[[
        'SK_ID_CURR', 'risk_band', 'final_pd',
        'recommended_intervention',
        'case_summary',
        'customer_message'
    ]].head(50).to_dict(orient='records')


# P19: Dynamic Contagion detection endpoint
@app.get("/contagion")
def get_contagion():
    if PRED_DF.empty or MASTER_DF.empty:
        return {"active": False}

    pred = PRED_DF.copy()
    pred['risk_band'] = pred['risk_band'].apply(remap_band)

    stressed_preds = pred[pred['risk_band'].isin(['RED', 'HIGH'])]
    if stressed_preds.empty:
        return {"active": False}

    merged = stressed_preds.merge(
        MASTER_DF[['SK_ID_CURR', 'ORGANIZATION_TYPE', 'AMT_CREDIT']],
        on='SK_ID_CURR',
        how='inner'
    )

    if merged.empty:
        return {"active": False}

    # Group by ORGANIZATION_TYPE and filter those with 5 or more stressed customers
    counts = merged.groupby('ORGANIZATION_TYPE').size()
    contagion_orgs = counts[counts >= 5]

    if contagion_orgs.empty:
        return {"active": False}

    # Get the organization type with the highest count or just the first one
    org_type = contagion_orgs.index[0]
    count = int(contagion_orgs.iloc[0])

    org_subset = merged[merged['ORGANIZATION_TYPE'] == org_type]
    total_exposure = float(org_subset['AMT_CREDIT'].sum())

    return {
        "active": True,
        "employer_name": f"Employer Group #{int(org_type)}",
        "affected_count": count,
        "total_exposure": round(total_exposure, 2)
    }


# P7: Updated /score response with complete planning fields
@app.get("/score")
def score_customer(
    income: float,
    emi: float,
    savings_drawdown: float,
    failed_autodebits: int,
    lending_app: int,
    days_since_income: int
):
    f1 = min(emi / (income + 1), 1.5)
    f2 = savings_drawdown / 100
    f3 = days_since_income / 30
    f5 = failed_autodebits / 10
    f6 = lending_app
    f11 = 1 if savings_drawdown > 60 else 0

    stress = (
        f1 * 0.25 +
        f2 * 0.20 +
        f3 * 0.15 +
        f5 * 0.20 +
        f6 * 0.10 +
        f11 * 0.10
    )

    pd_score = min(stress, 0.99)
    pd_pct = round(pd_score * 100, 1)

    if pd_score >= 0.25:
        band = 'RED'
    elif pd_score >= 0.10:
        band = 'HIGH'
    elif pd_score >= 0.05:
        band = 'YELLOW'
    else:
        band = 'GREEN'

    drivers = []
    explanations = []
    if f1 > 0.5:
        drivers.append("EMI burden is critically high")
        explanations.append("Your monthly debt payments consume an unsustainable portion of your income, increasing delinquency risk.")
    if f2 > 0.4:
        drivers.append("Savings depleting rapidly")
        explanations.append("Frequent withdrawals have significantly reduced your savings buffer, leaving you vulnerable to shocks.")
    if f3 > 0.3:
        drivers.append("Income arriving late")
        explanations.append("Your income deposits are arriving significantly later than your typical schedule, threatening payment timelines.")
    if f5 > 0.2:
        drivers.append("Multiple auto-debit failures")
        explanations.append("Recurring automated payment failures indicate cash flow stress at the start of the month.")
    if f6 == 1:
        drivers.append("External lending app usage detected")
        explanations.append("Active credit transactions with external short-term lending apps suggest a urgent cash crunch.")

    if not drivers:
        drivers.append("Customer appears financially stable")
        explanations.append("Your financial profile currently shows low overall stress levels.")

    # Select top driver
    top_label = drivers[0]
    top_expl = explanations[0]

    # Map driver label to suggested measure
    measure_map = {
        "EMI burden is critically high": "Extend loan tenure from 36 to 48 months to lower monthly EMI outflow",
        "Savings depleting rapidly": "Provide financial counseling and temporary savings cushion assistance",
        "Income arriving late": "Shift EMI due date to align with actual income arrival pattern",
        "Multiple auto-debit failures": "30-day payment holiday to restore financial breathing room",
        "External lending app usage detected": "Debt consolidation planning and short-term credit restrictions"
    }
    suggested = measure_map.get(top_label, "Standard account monitoring and basic financial health review")

    # Generate model reasoning
    msg = (
        f"Dear customer, we noticed some changes in your payment rhythm. To support you, we suggest setting up a: "
        f"{suggested}. Please contact your Relationship Manager to apply this immediately."
    )

    return {
        "pd_score": pd_pct,
        "risk_band": band,
        "drivers": drivers[:3],
        "suggested_measure": suggested,
        "customer_message": msg,
        "top_driver": {
            "label": top_label,
            "explanation": top_expl
        }
    }
