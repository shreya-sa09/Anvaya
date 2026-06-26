import pandas as pd

PROCESSED_PATH = '../data/processed/'

AVG_LOAN_AMOUNT = 500000
AVG_INCOME = 150000


def calculate_intervention(top_driver_feature,
                           top_driver_label,
                           final_pd,
                           customer_id):

    intervention = {
        'customer_id': customer_id,
        'final_pd': final_pd,
        'top_driver': top_driver_label,
        'intervention_type': None,
        'intervention_detail': None,
        'rupee_impact': None,
        'urgency': None
    }

    f = top_driver_feature

    if 'F3' in f or 'income_irregularity' in f:
        intervention['intervention_type'] = 'EMI_DATE_SHIFT'
        intervention['intervention_detail'] = (
            'Shift EMI due date to 7th of month '
            'when salary credits are most consistent'
        )
        intervention['rupee_impact'] = (
            'Reduces missed payment risk. '
            'No change in EMI amount.'
        )
        intervention['urgency'] = 'MEDIUM'

    elif 'F7' in f or 'cash_hoarding' in f:
        current_emi = round(AVG_LOAN_AMOUNT / 36, 0)
        new_emi = round(AVG_LOAN_AMOUNT / 48, 0)
        saving = current_emi - new_emi
        intervention['intervention_type'] = 'TENURE_EXTENSION'
        intervention['intervention_detail'] = (
            f'Extend loan tenure from 36 to 48 months. '
            f'New EMI: Rs {new_emi:,.0f}/month'
        )
        intervention['rupee_impact'] = (
            f'Customer saves Rs {saving:,.0f}/month '
            f'in cash outflow immediately'
        )
        intervention['urgency'] = 'HIGH'

    elif 'F2' in f or 'savings_drawdown' in f:
        credit_limit = round(AVG_INCOME * 0.5, 0)
        intervention['intervention_type'] = 'EMERGENCY_CREDIT_LINE'
        intervention['intervention_detail'] = (
            f'Offer emergency overdraft up to '
            f'Rs {credit_limit:,.0f} at 0% for 30 days'
        )
        intervention['rupee_impact'] = (
            f'Provides Rs {credit_limit:,.0f} buffer '
            f'to prevent missed EMI'
        )
        intervention['urgency'] = 'HIGH'

    elif 'F8' in f or 'stress_velocity' in f:
        intervention['intervention_type'] = 'RM_ESCALATION'
        intervention['intervention_detail'] = (
            'Stress accelerating rapidly. '
            'Assign Relationship Manager immediately. '
            'Do not send automated message.'
        )
        intervention['rupee_impact'] = (
            'Early RM contact reduces default '
            'probability by estimated 30-40%'
        )
        intervention['urgency'] = 'CRITICAL'

    elif 'F9' in f or 'payment_timing' in f:
        intervention['intervention_type'] = 'PAYMENT_HOLIDAY'
        intervention['intervention_detail'] = (
            '30-day payment holiday to restore '
            'financial breathing room'
        )
        intervention['rupee_impact'] = (
            f'Defers Rs {round(AVG_LOAN_AMOUNT/36,0):,.0f} '
            f'this month. Added to end of tenure.'
        )
        intervention['urgency'] = 'MEDIUM'

    elif 'F11' in f or 'overdraft' in f:
        intervention['intervention_type'] = 'LOAN_RESTRUCTURE'
        intervention['intervention_detail'] = (
            'Restructure loan with 3-month moratorium '
            'on principal repayment'
        )
        intervention['rupee_impact'] = (
            'Reduces immediate cash pressure. '
            'Interest continues to accrue.'
        )
        intervention['urgency'] = 'HIGH'

    elif 'F10' in f or 'cohort_stress' in f:
        intervention['intervention_type'] = 'COHORT_RELIEF'
        intervention['intervention_detail'] = (
            'Sector-wide stress detected. '
            'Enroll in cohort relief program.'
        )
        intervention['rupee_impact'] = (
            'EMI reduced by 20% for 3 months '
            f'= Rs {round(AVG_LOAN_AMOUNT/36*0.2,0):,.0f}/month saving'
        )
        intervention['urgency'] = 'MEDIUM'

    else:
        intervention['intervention_type'] = 'MANUAL_REVIEW'
        intervention['intervention_detail'] = (
            'Flag for manual review by credit team'
        )
        intervention['rupee_impact'] = 'To be determined'
        intervention['urgency'] = 'LOW'

    return intervention


def run_intervention_engine():
    print("="*55)
    print("INTERVENTION ENGINE")
    print("="*55)

    shap_df = pd.read_csv(PROCESSED_PATH + 'shap_explanations.csv')
    flagged = shap_df[shap_df['risk_band'].isin(['YELLOW', 'RED'])].copy()

    print(f"Flagged customers: {len(flagged)}")
    print()

    results = []
    for _, row in flagged.iterrows():
        result = calculate_intervention(
            top_driver_feature=row['top_driver_1_feature'],
            top_driver_label=row['top_driver_1_label'],
            final_pd=row['final_pd'],
            customer_id=row['SK_ID_CURR']
        )
        result['risk_band'] = row['risk_band']
        result['case_summary'] = row['case_summary']
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_PATH + 'interventions.csv', index=False)

    print("Intervention type distribution:")
    print(results_df['intervention_type'].value_counts().to_string())
    print()
    print("Urgency distribution:")
    print(results_df['urgency'].value_counts().to_string())
    print()

    for band in ['RED', 'YELLOW']:
        subset = results_df[results_df['risk_band'] == band]
        if len(subset) == 0:
            continue
        sample = subset.iloc[0]
        print(f"--- {band} SAMPLE ---")
        print(f"  Customer     : {sample['customer_id']}")
        print(f"  Top driver   : {sample['top_driver']}")
        print(f"  Intervention : {sample['intervention_type']}")
        print(f"  Detail       : {sample['intervention_detail']}")
        print(f"  Rupee impact : {sample['rupee_impact']}")
        print(f"  Urgency      : {sample['urgency']}")
        print()

    print(f"Saved interventions.csv — {len(results_df)} rows")
    print("="*55)


if __name__ == "__main__":
    run_intervention_engine()