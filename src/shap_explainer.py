import os
import pandas as pd
import numpy as np
import joblib
import shap

PROCESSED_PATH = "../data/processed/"
MODELS_PATH = "../models/"

FEATURE_LABELS = {
    'F2_savings_drawdown_WOE':       'Savings Depletion',
    'F3_income_irregularity_WOE':    'Income Arrival Irregularity',
    'F7_cash_hoarding_WOE':          'Cash Hoarding Behavior',
    'F8_stress_velocity_WOE':        'Stress Accelerating Rapidly',
    'F9_payment_timing_entropy_WOE': 'Erratic Payment Timing',
    'F10_cohort_stress_WOE':         'Peer Group Under Stress',
    'F11_overdraft_freq_WOE':        'Frequent Overdrafts',
}

EXPLANATION_TEMPLATES = {
    'F2_savings_drawdown_WOE':
        'Savings balance has declined significantly over the last 60 days, '
        'leaving very little financial buffer.',
    'F3_income_irregularity_WOE':
        'Income is arriving later than this customer\'s own historical pattern, '
        'suggesting salary delays or irregular gig payouts.',
    'F7_cash_hoarding_WOE':
        'ATM cash withdrawals have spiked above this customer\'s normal level, '
        'suggesting digital credit channels may be exhausted.',
    'F8_stress_velocity_WOE':
        'Financial stress is accelerating rapidly compared to earlier months, '
        'not just present but getting worse quickly.',
    'F9_payment_timing_entropy_WOE':
        'Payment dates have become increasingly erratic, '
        'suggesting reactive rather than planned cash management.',
    'F10_cohort_stress_WOE':
        'Customers in the same income and employment group are also showing '
        'elevated stress, pointing to a possible external economic pressure.',
    'F11_overdraft_freq_WOE':
        'Account balance has gone negative on multiple days this period, '
        'indicating the financial buffer is fully exhausted.',
}

INTERVENTION_MAP = {
    'F2_savings_drawdown_WOE':       'Emergency credit line or temporary overdraft facility',
    'F3_income_irregularity_WOE':    'Shift EMI due date to align with actual income arrival pattern',
    'F7_cash_hoarding_WOE':          '30-day payment holiday to restore financial breathing room',
    'F8_stress_velocity_WOE':        'Immediate RM outreach — stress accelerating beyond automated intervention',
    'F9_payment_timing_entropy_WOE': 'Shift EMI due date and set up flexible payment reminder',
    'F10_cohort_stress_WOE':         'Monitor closely — external pressure, check employer stress signals',
    'F11_overdraft_freq_WOE':        '30-day payment holiday and emergency credit line',
}


def load_dataframe(filename, required_columns=None):
    path = os.path.join(PROCESSED_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{filename} not found in {PROCESSED_PATH}.')
    df = pd.read_csv(path)
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise KeyError(
                f'Missing columns {missing} in {filename}. '
                'Check upstream pipeline output.'
            )
    return df


def load_model():
    path = os.path.join(MODELS_PATH, 'lgb_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError('lgb_model.pkl not found in models/. Run model_trainer.py first.')
    return joblib.load(path)


def build_explanation_row(customer_id, band, pd_score, feature_names, shap_row):
    top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
    top_drivers = []
    for rank, idx in enumerate(top_idx, start=1):
        feature = feature_names[idx]
        label = FEATURE_LABELS.get(feature, feature)
        explanation = EXPLANATION_TEMPLATES.get(feature, 'No explanation available.')
        contribution = float(abs(shap_row[idx]) / (np.abs(shap_row).sum() + 1e-9))
        top_drivers.append({
            'feature': feature,
            'label': label,
            'explanation': explanation,
            'contribution_pct': contribution * 100,
        })

    recommended = INTERVENTION_MAP.get(top_drivers[0]['feature'], 'Review account for tailored intervention')
    case_summary = (
        f"This customer is flagged as {band} risk with a "
        f"{pd_score * 100:.0f}% probability of default. "
        f"Primary driver: {top_drivers[0]['explanation']} "
        f"Secondary factor: {top_drivers[1]['explanation']} "
        f"Additional signal: {top_drivers[2]['explanation']} "
        f"Recommended action: {recommended}."
    )

    return {
        'SK_ID_CURR': customer_id,
        'risk_band': band,
        'final_pd': pd_score,
        'top_driver_1_feature': top_drivers[0]['feature'],
        'top_driver_1_label': top_drivers[0]['label'],
        'top_driver_1_explanation': top_drivers[0]['explanation'],
        'top_driver_1_shap_contribution_pct': round(top_drivers[0]['contribution_pct'], 2),
        'top_driver_2_feature': top_drivers[1]['feature'],
        'top_driver_2_label': top_drivers[1]['label'],
        'top_driver_2_explanation': top_drivers[1]['explanation'],
        'top_driver_2_shap_contribution_pct': round(top_drivers[1]['contribution_pct'], 2),
        'top_driver_3_feature': top_drivers[2]['feature'],
        'top_driver_3_label': top_drivers[2]['label'],
        'top_driver_3_explanation': top_drivers[2]['explanation'],
        'top_driver_3_shap_contribution_pct': round(top_drivers[2]['contribution_pct'], 2),
        'recommended_intervention': recommended,
        'case_summary': case_summary,
    }


def run():
    print('=== SHAP Explainer ===')
    features_df = load_dataframe('features_final.csv', ['SK_ID_CURR', 'TARGET'])
    preds_df = load_dataframe('model_predictions.csv', ['SK_ID_CURR', 'final_pd', 'risk_band'])
    model = load_model()
    selected = load_dataframe('selected_features.csv', ['selected_features'])
    feature_cols = selected['selected_features'].tolist()

    flagged = preds_df[preds_df['risk_band'].isin(['YELLOW', 'HIGH', 'RED'])].copy()
    print(f"Loaded {len(flagged)} flagged customers from model_predictions.csv")

    flagged = flagged.merge(features_df[['SK_ID_CURR'] + feature_cols], on='SK_ID_CURR', how='left')
    missing_features = [c for c in feature_cols if c not in flagged.columns]
    if missing_features:
        raise KeyError(
            f'Missing feature columns in merged dataframe: {missing_features}'
        )

    X_flagged = flagged[feature_cols].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_flagged)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print('Computing SHAP values and building explanations...')
    explanations = []
    for idx, row in flagged.iterrows():
        shap_row = shap_values[idx]
        explanations.append(
            build_explanation_row(
                int(row['SK_ID_CURR']),
                row['risk_band'],
                float(row['final_pd']),
                feature_cols,
                shap_row,
            )
        )

    explanation_df = pd.DataFrame(explanations)
    explanation_df.to_csv(os.path.join(PROCESSED_PATH, 'shap_explanations.csv'), index=False)
    print('Saved shap_explanations.csv')
    print(explanation_df.head(3).to_string(index=False))

    band_counts = explanation_df['risk_band'].value_counts().reindex(['YELLOW', 'HIGH', 'RED'], fill_value=0)
    print('\nExplanations generated per band:')
    print(band_counts.to_string())

    for band in ['YELLOW', 'HIGH', 'RED']:
        sample = explanation_df[explanation_df['risk_band'] == band].head(1)
        if not sample.empty:
            row = sample.iloc[0]
            print(f"\nSample {band} explanation:")
            print(f"  Customer {row['SK_ID_CURR']} — {row['risk_band']} — PD {row['final_pd']:.2%}")
            print(f"  Primary: {row['top_driver_1_label']}")
            print(f"  Secondary: {row['top_driver_2_label']}")
            print(f"  Third: {row['top_driver_3_label']}")
            print(f"  Intervention: {row['recommended_intervention']}")
            print(f"  Summary: {row['case_summary']}")

    print('\nSHAP explainer complete.')


if __name__ == '__main__':
    run()
