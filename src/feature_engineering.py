import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

PROCESSED_PATH = "../data/processed/"
SYNTHETIC_PATH = "../data/synthetic/"

def load_all_data():
    print("Loading master + synthetic files...")
    master  = pd.read_csv(PROCESSED_PATH + "master.csv")
    balance = pd.read_csv(SYNTHETIC_PATH + "synthetic_daily_balance.csv")
    income  = pd.read_csv(SYNTHETIC_PATH + "synthetic_income_arrival.csv")
    txn     = pd.read_csv(SYNTHETIC_PATH + "synthetic_transactions.csv")
    print(f"  master: {master.shape}")
    print(f"  balance: {balance.shape}")
    print(f"  income: {income.shape}")
    print(f"  txn: {txn.shape}")
    return master, balance, income, txn


# F1 — EMI to Income Ratio
def compute_f1(master):
    master['F1_emi_to_income'] = (
        master['AMT_ANNUITY'] / (master['AMT_INCOME_TOTAL'] / 12)
    ).clip(0, 5)
    return master


# F2 — Savings Drawdown Percentage
def compute_f2(master, balance):
    early = balance[balance['date'] <= 60].groupby('customer_id')['closing_balance'].mean().rename('bal_early')
    late  = balance[balance['date'] >= 120].groupby('customer_id')['closing_balance'].mean().rename('bal_late')
    df = pd.concat([early, late], axis=1).reset_index()
    df['F2_savings_drawdown'] = (
        (df['bal_early'] - df['bal_late']) / (df['bal_early'].abs() + 1)
    ).clip(0, 1)
    master = master.merge(
        df[['customer_id', 'F2_savings_drawdown']],
        left_on='SK_ID_CURR', right_on='customer_id', how='left'
    ).drop(columns='customer_id')
    master['F2_savings_drawdown'] = master['F2_savings_drawdown'].fillna(
        master['F2_savings_drawdown'].median()
    )
    return master


# F3 — Income Arrival Irregularity
def compute_f3(income):
    agg = income.groupby('customer_id').agg(
        mean_delay=('delay_days', 'mean'),
        std_delay=('delay_days', 'std'),
    ).reset_index()
    agg['std_delay'] = agg['std_delay'].fillna(0)
    agg['F3_income_irregularity'] = (
        agg['mean_delay'] * 0.5 + agg['std_delay'] * 0.5
    ).clip(0, 30)
    return agg[['customer_id', 'F3_income_irregularity']]


# F4 — Spending Pattern Shift
def compute_f4(txn):
    categories = ['groceries', 'rent_utilities', 'medical', 'entertainment',
                  'atm_cash', 'dining_travel', 'online_shopping']

    def spend_vector(df):
        total = df['amount'].sum()
        if total == 0:
            return [0.0] * len(categories)
        return [df[df['category'] == c]['amount'].sum() / total for c in categories]

    rows = []
    for cid, grp in txn.groupby('customer_id'):
        early = grp[grp['date'] <= 90]
        late  = grp[grp['date'] > 90]
        ve = spend_vector(early)
        vl = spend_vector(late)
        try:
            shift = cosine(ve, vl) if sum(ve) > 0 and sum(vl) > 0 else 0.0
        except:
            shift = 0.0
        rows.append({'customer_id': cid, 'F4_spend_shift': round(shift, 4)})
    return pd.DataFrame(rows)


# F5 — Auto-Debit Failure Count
def compute_f5(master, balance):
    balance = balance.copy()
    balance['month_number'] = ((balance['date'] - 1) // 30) + 1
    window = balance[balance['date'].between(25, 30)]
    min_balance = window.groupby(['customer_id', 'month_number'])['closing_balance'].min().reset_index()

    emi = master[['SK_ID_CURR', 'AMT_ANNUITY']].copy()
    emi['customer_id'] = emi['SK_ID_CURR']
    emi['monthly_emi'] = emi['AMT_ANNUITY'] / 12

    min_balance = min_balance.merge(emi[['customer_id', 'monthly_emi']], on='customer_id', how='left')
    min_balance['failure'] = (min_balance['closing_balance'] < min_balance['monthly_emi']).astype(int)

    agg = min_balance.groupby('customer_id')['failure'].sum().reset_index()
    agg.columns = ['customer_id', 'F5_autodebit_failures']
    return agg


# F6 — Lending App Transaction Count
def compute_f6(txn):
    agg = txn.copy()
    agg['amount_int'] = agg['amount'].round().astype(int)
    agg['atm_cash_multiple_500'] = (
        (agg['category'] == 'atm_cash') & (agg['amount_int'] % 500 == 0)
    ).astype(int)
    result = agg.groupby('customer_id')['atm_cash_multiple_500'].sum().reset_index()
    result.columns = ['customer_id', 'F6_lending_app_count']
    return result


# F7 — Cash Hoarding Ratio
def compute_f7(txn):
    total = txn.groupby('customer_id')['amount'].sum().rename('total_spend')
    atm   = txn[txn['is_atm_withdrawal'] == 1].groupby('customer_id')['amount'].sum().rename('atm_spend')
    df = pd.concat([total, atm], axis=1).reset_index()
    df['F7_cash_hoarding'] = (df['atm_spend'] / (df['total_spend'] + 1)).clip(0, 1).fillna(0)
    return df[['customer_id', 'F7_cash_hoarding']]


# F8 — Stress Velocity
def compute_f8(balance):
    rows = []
    for cid, grp in balance.groupby('customer_id'):
        grp = grp.sort_values('date')
        if len(grp) >= 2:
            slope = np.polyfit(grp['date'].astype(float), grp['closing_balance'].astype(float), 1)[0]
        else:
            slope = 0.0
        rows.append({'customer_id': cid, 'F8_stress_velocity': round(slope, 4)})
    return pd.DataFrame(rows)


# F9 — Payment Timing Entropy
def compute_f9(income):
    agg = income.groupby('customer_id')['actual_date'].std().reset_index()
    agg.columns = ['customer_id', 'F9_payment_timing_entropy']
    agg['F9_payment_timing_entropy'] = agg['F9_payment_timing_entropy'].fillna(0)
    return agg


# F10 — Peer Cohort Stress Index
def compute_f10(balance):
    avg_bal    = balance.groupby('customer_id')['closing_balance'].mean().reset_index()
    avg_bal.columns = ['customer_id', 'avg_balance']
    pop_median = avg_bal['avg_balance'].median()
    avg_bal['F10_cohort_stress'] = (
        (pop_median - avg_bal['avg_balance']) / (abs(pop_median) + 1)
    ).clip(0, 5)
    return avg_bal[['customer_id', 'F10_cohort_stress']]


# F11 — Overdraft Frequency
def compute_f11(balance):
    agg = balance.groupby('customer_id')['is_negative'].sum().reset_index()
    agg.columns = ['customer_id', 'F11_overdraft_freq']
    return agg


# F12 — Cross Loan Payment Consistency
def compute_f12(master):
    if 'inst_payment_delay_std' in master.columns:
        master['F12_cross_loan_consistency'] = master['inst_payment_delay_std'].fillna(0).clip(0, 100)
    else:
        master['F12_cross_loan_consistency'] = 0.0
    return master


# F13 — Secondary Income Index
def compute_f13(income):
    customer_mean = income.groupby('customer_id')['amount'].transform('mean')
    income = income.copy()
    income['is_secondary'] = (income['amount'] > customer_mean * 1.2).astype(int)
    agg = income.groupby('customer_id').agg(
        total_credits=('amount', 'count'),
        secondary_credits=('is_secondary', 'sum')
    ).reset_index()
    agg['F13_secondary_income'] = (
        agg['secondary_credits'] / (agg['total_credits'] + 1)
    ).clip(0, 1)
    return agg[['customer_id', 'F13_secondary_income']]


def compute_f14_to_f18(master):
    master['F14_ext_source_2'] = master['EXT_SOURCE_2']
    master['F15_ext_source_3'] = master['EXT_SOURCE_3']
    master['F16_days_birth'] = master['DAYS_BIRTH']
    master['F17_days_employed'] = master['DAYS_EMPLOYED']
    master['F18_phone_change'] = master['DAYS_LAST_PHONE_CHANGE']
    return master


def build_features(master, balance, income, txn):
    print("\nBuilding 13 features + 5 demographics...")

    master = compute_f1(master);           print("  F1 done — EMI to Income Ratio")
    master = compute_f2(master, balance);  print("  F2 done — Savings Drawdown")
    master = compute_f12(master);          print("  F12 done — Cross Loan Consistency")
    master = compute_f14_to_f18(master);   print("  F14-F18 done — Demographics & Bureau")

    f3  = compute_f3(income);   print("  F3 done — Income Irregularity")
    f4  = compute_f4(txn);      print("  F4 done — Spend Shift")
    f5  = compute_f5(master, balance);   print("  F5 done — AutoDebit Failures")
    f6  = compute_f6(txn);      print("  F6 done — Lending App Count")
    f7  = compute_f7(txn);      print("  F7 done — Cash Hoarding")
    f8  = compute_f8(balance);  print("  F8 done — Stress Velocity")
    f9  = compute_f9(income);   print("  F9 done — Payment Timing Entropy")
    f10 = compute_f10(balance); print("  F10 done — Cohort Stress")
    f11 = compute_f11(balance); print("  F11 done — Overdraft Frequency")
    f13 = compute_f13(income);  print("  F13 done — Secondary Income")

    for df in [f3, f4, f5, f6, f7, f8, f9, f10, f11, f13]:
        master = master.merge(df, left_on='SK_ID_CURR', right_on='customer_id', how='left').drop(
            columns='customer_id', errors='ignore'
        )

    # Fill any remaining nulls with median
    f_cols = [
        'F1_emi_to_income',
        'F2_savings_drawdown',
        'F3_income_irregularity',
        'F4_spend_shift',
        'F5_autodebit_failures',
        'F6_lending_app_count',
        'F7_cash_hoarding',
        'F8_stress_velocity',
        'F9_payment_timing_entropy',
        'F10_cohort_stress',
        'F11_overdraft_freq',
        'F12_cross_loan_consistency',
        'F13_secondary_income',
        'F14_ext_source_2',
        'F15_ext_source_3',
        'F16_days_birth',
        'F17_days_employed',
        'F18_phone_change'
    ]
    for col in f_cols:
        master[col] = master[col].fillna(master[col].median())

    print(f"\n  Feature matrix shape: {master.shape}")
    print(f"  Features built: {f_cols}")
    return master


if __name__ == "__main__":
    master, balance, income, txn = load_all_data()
    master = build_features(master, balance, income, txn)

    feature_cols = ['SK_ID_CURR', 'TARGET'] + [c for c in master.columns if c.startswith('F')]
    features_df  = master[feature_cols]
    features_df.to_csv(PROCESSED_PATH + "features.csv", index=False)
    print(f"\nSaved features.csv — shape: {features_df.shape}")
    print("\nFeature engineering complete.")