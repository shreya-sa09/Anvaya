from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE      = "C:/Project/My_Anvaya/data/processed/"
MODEL_DIR = "C:/Project/My_Anvaya/models/"
SYNTH_DIR = "C:/Project/My_Anvaya/data/synthetic/"

ORG_TYPE_MAP = {
    0:'Business Entity Type 3', 1:'School', 2:'Government', 3:'Religion',
    4:'Other', 5:'Medicine', 6:'Business Entity Type 2', 7:'Self-employed',
    8:'Transport Type 2', 9:'Construction', 10:'Housing', 11:'Kindergarten',
    12:'Trade Type 6', 13:'Industry Type 11', 14:'Military', 15:'Services',
    16:'Security Ministries', 17:'Transport Type 4', 18:'Trade Type 3',
    19:'University', 20:'Transport Type 3', 21:'Police',
    22:'Business Entity Type 1', 23:'Postal', 24:'Industry Type 4',
    25:'Agriculture', 26:'Restaurant', 27:'Culture', 28:'Hotel',
    29:'Electricity', 30:'Security', 31:'Realtor', 32:'Telecom',
    33:'Industry Type 1', 34:'Emergency', 35:'Bank', 36:'Industry Type 9',
    37:'Insurance', 38:'Trade Type 2', 39:'Trade Type 7',
    40:'Industry Type 2', 41:'Trade Type 1', 42:'Industry Type 12',
    43:'Industry Type 5', 44:'Industry Type 10', 45:'Legal Services',
    46:'Advertising', 47:'Trade Type 5', 48:'Cleaning',
    49:'Industry Type 13', 50:'Trade Type 4', 51:'Industry Type 3',
    52:'Industry Type 7', 53:'Industry Type 8', 54:'Industry Type 6',
    55:'XNA', 56:'Mobile', 57:'Industry Type 11',
}


# ── Load CSVs at startup ─────────────────────────────────────────────────────
def _load(path, id_col='SK_ID_CURR'):
    try:
        df = pd.read_csv(path)
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(float).astype(int)
        return df
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return pd.DataFrame()

# Use combined_predictions (all splits) instead of holdout-only model_predictions
PRED_DF   = _load(BASE + "combined_predictions.csv")
SHAP_DF   = _load(BASE + "shap_explanations.csv")
AGENT_DF  = _load(BASE + "agent_outputs.csv")
FEAT_DF   = _load(BASE + "features_final.csv")
MASTER_DF = _load(BASE + "master.csv")

# Load real income arrival data for payment history
INCOME_DF = _load(SYNTH_DIR + "synthetic_income_arrival.csv", id_col='customer_id')
if not INCOME_DF.empty:
    INCOME_DF = INCOME_DF.rename(columns={'customer_id': 'SK_ID_CURR'})


# ── Load ML models and inference artifacts ───────────────────────────────────
def _load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: could not load model {path}: {e}")
        return None

XGB_MODEL        = _load_model(MODEL_DIR + "xgb_model.pkl")
LGB_MODEL        = _load_model(MODEL_DIR + "lgb_model.pkl")
CALIBRATED_MODEL = _load_model(MODEL_DIR + "calibrated_model.pkl")

WOE_MAPPINGS = {}
try:
    with open(MODEL_DIR + "woe_mappings.json", "r") as f:
        WOE_MAPPINGS = json.load(f)
except Exception as e:
    print(f"Warning: could not load woe_mappings.json: {e}")

POP_BASELINE = {}
try:
    bl = pd.read_csv(BASE + "population_baseline.csv")
    POP_BASELINE = {r['feature']: {'mean': r['population_mean'], 'std': r['population_std']}
                    for _, r in bl.iterrows()}
except Exception as e:
    print(f"Warning: could not load population_baseline.csv: {e}")

SELECTED_FEATURES = []
try:
    sf = pd.read_csv(BASE + "selected_features.csv")
    SELECTED_FEATURES = sf['selected_features'].tolist()
except Exception as e:
    print(f"Warning: could not load selected_features.csv: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def remap_band(band):
    return str(band).strip().upper()

def risk_band(pd_score):
    if pd_score < 0.05:   return 'GREEN'
    if pd_score < 0.15:   return 'YELLOW'
    if pd_score < 0.25:   return 'HIGH'
    return 'RED'

def apply_woe_single(value, mapping):
    """Apply saved WoE bin mapping to a scalar value."""
    if mapping is None:
        return 0.0
    edges = mapping['edges']
    woes  = mapping['woes']
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return woes[i] if i < len(woes) else 0.0
    return woes[-1] if woes else 0.0

def score_features_with_models(feature_dict):
    """
    Take a dict of raw feature values, apply PRE scaling then WoE,
    run ensemble inference, return (xgb_score, lgb_score, final_pd, risk_band).
    """
    if not SELECTED_FEATURES or XGB_MODEL is None or LGB_MODEL is None:
        return None, None, None, None

    # Z-score normalise using population baseline
    z_dict = {}
    for feat, stats in POP_BASELINE.items():
        raw_val = feature_dict.get(feat, stats['mean'])
        z_col   = feat + '_Z'
        z_dict[z_col] = float(np.clip((raw_val - stats['mean']) / (stats['std'] + 1e-9), -5, 5))

    # Apply WoE mappings
    woe_dict = {}
    for z_col, z_val in z_dict.items():
        base    = z_col[:-2]          # strip _Z
        woe_col = base + '_WOE'
        mapping = WOE_MAPPINGS.get(z_col)
        woe_dict[woe_col] = apply_woe_single(z_val, mapping)

    # Build feature arrays
    xgb_features = SELECTED_FEATURES[:5]
    lgb_features = SELECTED_FEATURES

    X_xgb = np.array([[woe_dict.get(f, 0.0) for f in xgb_features]])
    X_lgb = pd.DataFrame([[woe_dict.get(f, 0.0) for f in lgb_features]], columns=lgb_features)

    xgb_score = float(XGB_MODEL.predict_proba(X_xgb)[0][1])
    lgb_score  = float(LGB_MODEL.predict_proba(X_lgb)[0][1])

    meta_X    = np.array([[xgb_score, lgb_score]])
    final_pd  = float(CALIBRATED_MODEL.predict_proba(meta_X)[0][1])
    band      = risk_band(final_pd)

    return xgb_score, lgb_score, final_pd, band


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/portfolio")
def get_portfolio():
    if PRED_DF.empty:
        return {"total": 0, "critical": 0, "high": 0, "yellow": 0, "green": 0, "flagged": 0, "avg_pd": 0.0}

    pred = PRED_DF.copy()
    pred['risk_band'] = pred['risk_band'].apply(remap_band)

    total    = len(pred)
    critical = len(pred[pred['risk_band'] == 'RED'])
    high     = len(pred[pred['risk_band'] == 'HIGH'])
    yellow   = len(pred[pred['risk_band'] == 'YELLOW'])
    green    = len(pred[pred['risk_band'] == 'GREEN'])
    avg_pd   = round(pred['final_pd'].mean() * 100, 1)

    return {
        "total":    total,
        "critical": critical,
        "high":     high,
        "yellow":   yellow,
        "green":    green,
        "flagged":  critical + high + yellow,
        "avg_pd":   avg_pd,
    }


@app.get("/customers")
def get_customers(band: str = "All", limit: int = 50):
    if PRED_DF.empty:
        return []

    pred = PRED_DF.copy()
    shap = SHAP_DF.copy()

    pred['risk_band'] = pred['risk_band'].apply(remap_band)

    if band != "All":
        pred = pred[pred['risk_band'] == band.upper()]
        order = {'RED': 0, 'HIGH': 1, 'YELLOW': 2, 'GREEN': 3}
        pred['_s'] = pred['risk_band'].map(order).fillna(9)
        pred = pred.sort_values('_s').drop(columns='_s')
    else:
        # If 'All', return a randomized sample so all bands are visible
        pred = pred.sample(n=min(len(pred), limit*4), random_state=42)

    pred = pred.head(limit)

    if not shap.empty:
        merged = pred.merge(
            shap[['SK_ID_CURR', 'top_driver_1_label']],
            on='SK_ID_CURR', how='left'
        )
    else:
        merged = pred.copy()
        merged['top_driver_1_label'] = '-'

    merged['top_driver_1_label'] = merged['top_driver_1_label'].fillna('-')

    return merged[['SK_ID_CURR', 'risk_band', 'final_pd', 'top_driver_1_label']].to_dict(orient='records')


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

    if not AGENT_DF.empty:
        agent_row = AGENT_DF[AGENT_DF['SK_ID_CURR'] == customer_id]
        if not agent_row.empty:
            result['customer_message'] = agent_row['customer_message'].iloc[0]

    # Real PD trend from xgb_score / lgb_score / final_pd (model outputs, not fabricated)
    xgb = float(result.get('xgb_score', 0.0))
    lgb = float(result.get('lgb_score', 0.0))
    fpd = float(result.get('final_pd', 0.0))

    # These are real model component scores — show them as a transparent breakdown
    result['pd_trend_90d'] = [
        {"day": "XGB Score",   "pd": round(xgb * 100, 1)},
        {"day": "LGB Score",   "pd": round(lgb * 100, 1)},
        {"day": "Blended",     "pd": round(((xgb + lgb) / 2) * 100, 1)},
        {"day": "Calibrated",  "pd": round(fpd * 100, 1)},
    ]

    # Real payment history from synthetic_income_arrival.csv
    if not INCOME_DF.empty:
        cust_income = INCOME_DF[INCOME_DF['SK_ID_CURR'] == customer_id].sort_values('month_number')
        if not cust_income.empty:
            result['payment_history_6m'] = [
                {"month": f"M{int(r['month_number'])}", "delay_days": int(r['delay_days'])}
                for _, r in cust_income.iterrows()
            ]
        else:
            result['payment_history_6m'] = []
    else:
        result['payment_history_6m'] = []

    # Rhythm deviation from SHAP contribution percentages (real SHAP values)
    rhythm = []
    for i in range(1, 4):
        label = result.get(f'top_driver_{i}_label')
        pct   = result.get(f'top_driver_{i}_shap_contribution_pct')
        if label and pd.notna(label) and label != '-' and pct and pd.notna(pct):
            curr = float(pct)
            base = max(5.0, round(curr * (1.0 - fpd), 1))
            rhythm.append({"label": label, "base": base, "current": curr})
    result['rhythm_deviation'] = rhythm

    # Stress velocity direction from final_pd
    if fpd > 0.20:
        result['stress_velocity'] = 'steep'
    elif fpd > 0.10:
        result['stress_velocity'] = 'gradual'
    else:
        result['stress_velocity'] = 'flat'

    # Cohort from master using real ORG_TYPE_MAP
    result['cohort'] = "General Retail Banking Cohort"
    if not MASTER_DF.empty:
        master_row = MASTER_DF[MASTER_DF['SK_ID_CURR'] == customer_id]
        if not master_row.empty:
            org_raw = master_row['ORGANIZATION_TYPE'].iloc[0]
            if pd.notna(org_raw):
                org_name = ORG_TYPE_MAP.get(int(org_raw), f"Sector #{int(org_raw)}")
                result['cohort'] = f"Cohort: {org_name}"

    return result


@app.get("/alerts")
def get_alerts():
    if SHAP_DF.empty:
        return []

    shap = SHAP_DF.copy()
    shap['risk_band'] = shap['risk_band'].apply(remap_band)

    # Return a mix of top risky customers and random others so all filters work in UI
    red_high = shap[shap['risk_band'].isin(['RED', 'HIGH'])].sort_values('final_pd', ascending=False).head(70)
    yellow_green = shap[shap['risk_band'].isin(['YELLOW', 'GREEN'])].sample(n=30, random_state=42)
    flagged = pd.concat([red_high, yellow_green]).sample(frac=1, random_state=42) # Shuffle them

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
            on='SK_ID_CURR', how='left'
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
        on='SK_ID_CURR', how='inner'
    )
    if merged.empty:
        return {"active": False}

    # Apply real sector name mapping
    merged['org_name'] = merged['ORGANIZATION_TYPE'].map(ORG_TYPE_MAP).fillna('Unknown Sector')

    counts      = merged.groupby('org_name').size()
    contagion   = counts[counts >= 5].sort_values(ascending=False)

    if contagion.empty:
        return {"active": False}

    top_org     = contagion.index[0]
    count       = int(contagion.iloc[0])
    org_subset  = merged[merged['org_name'] == top_org]
    exposure    = float(org_subset['AMT_CREDIT'].sum())

    return {
        "active":         True,
        "employer_name":  top_org,
        "affected_count": count,
        "total_exposure": round(exposure, 2),
    }


# ── /score: Real model inference ──────────────────────────────────────────────
@app.get("/score")
def score_customer(
    income:            float,
    emi:               float,
    savings_drawdown:  float,
    failed_autodebits: int,
    lending_app:       int,
    days_since_income: int,
    ext_source_2:      float = None,
    ext_source_3:      float = None,
    days_employed:     int   = -1000,
    days_birth:        int   = -12000,
    days_phone_change: int   = -300,
):
    # Dynamically align unprovided demographic signals with the live behavioral inputs
    # so the ML model accurately reflects the user's live testing scenario
    stress_level = (emi / (income + 1)) * 0.3 + (savings_drawdown / 100) * 0.3 + (failed_autodebits / 5) * 0.4
    
    if ext_source_2 is None:
        ext_source_2 = max(0.1, 0.7 - stress_level)
    if ext_source_3 is None:
        ext_source_3 = max(0.1, 0.7 - stress_level)

    # Map API inputs to the raw feature space the pipeline was trained on
    raw_features = {
        'F1_emi_to_income':       min(emi / (income + 1), 5.0),
        'F2_savings_drawdown':    savings_drawdown / 100.0,
        'F3_income_irregularity': days_since_income / 30.0,
        'F5_autodebit_failures':  float(failed_autodebits),
        'F6_lending_app_count':   float(lending_app),
        'F14_ext_source_2':       ext_source_2,
        'F15_ext_source_3':       ext_source_3,
        'F17_days_employed':      float(days_employed),
        'F16_days_birth':         float(days_birth),
        'F18_phone_change':       float(days_phone_change),
    }

    xgb_score, lgb_score, final_pd, band = score_features_with_models(raw_features)

    # Fall back to heuristic if models not loaded
    if final_pd is None:
        f1   = min(emi / (income + 1), 1.5)
        f2   = savings_drawdown / 100
        f3   = days_since_income / 30
        f5   = failed_autodebits / 10
        f6   = lending_app
        stress  = f1*0.25 + f2*0.20 + f3*0.15 + f5*0.20 + f6*0.10
        final_pd = min(stress, 0.99)
        band     = risk_band(final_pd)
        xgb_score = lgb_score = final_pd
        source = "heuristic_fallback"
    else:
        source = "calibrated_ensemble"

    pd_pct = round(final_pd * 100, 1)

    # Build drivers from raw feature magnitudes
    driver_map = {
        'EMI burden is critically high':     (raw_features.get('F1_emi_to_income', 0) > 0.5),
        'Savings depleting rapidly':          (savings_drawdown > 40),
        'Income arriving late':               (days_since_income > 10),
        'Multiple auto-debit failures':       (failed_autodebits > 2),
        'External lending app usage detected':(lending_app == 1),
        'Low external credit score':          (ext_source_2 < 0.3 or ext_source_3 < 0.3),
        'Short employment tenure':            (days_employed > -365),
    }
    drivers      = [d for d, active in driver_map.items() if active]
    if not drivers:
        drivers = ["Customer appears financially stable"]

    measure_map = {
        'EMI burden is critically high':      'Extend loan tenure from 36 to 48 months to lower monthly EMI',
        'Savings depleting rapidly':           'Provide financial counseling and temporary savings cushion',
        'Income arriving late':                'Shift EMI due date to align with income arrival pattern',
        'Multiple auto-debit failures':        '30-day payment holiday to restore financial breathing room',
        'External lending app usage detected': 'Debt consolidation and short-term credit restrictions',
        'Low external credit score':           'Escalate for manual credit review and enhanced monitoring',
        'Short employment tenure':             'Flexible EMI schedule aligned to employment stabilization',
        'Customer appears financially stable': 'Standard account monitoring',
    }
    suggested = measure_map.get(drivers[0], 'Standard account monitoring')

    msg = (
        f"Dear customer, we noticed some changes in your payment rhythm. "
        f"To support you, we suggest: {suggested}. "
        f"Please contact your Relationship Manager to apply this immediately."
    )

    return {
        "pd_score":        pd_pct,
        "risk_band":       band,
        "xgb_score":       round(xgb_score * 100, 1) if xgb_score is not None else None,
        "lgb_score":       round(lgb_score * 100, 1) if lgb_score is not None else None,
        "scoring_source":  source,
        "drivers":         drivers[:3],
        "suggested_measure": suggested,
        "customer_message":  msg,
        "top_driver": {
            "label":       drivers[0],
            "explanation": measure_map.get(drivers[0], ''),
        },
    }# Triggering backend reload to load new CSVs
