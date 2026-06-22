import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

PROCESSED_PATH = "../data/processed/"

def clean_dataframe(df, name):
    print(f"\nCleaning {name}...")

    # Drop columns with more than 40% nulls
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > 0.4].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns with >40% nulls")

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df)} duplicate rows")

    # Clip outliers using 1st and 99th percentile
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(Q1, Q3)

    print(f"  Final shape: {df.shape}")
    return df


def encode_categoricals(app):
    print("\nEncoding categorical columns...")
    cat_cols = app.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        app[col] = le.fit_transform(app[col].astype(str))
    print(f"  Encoded {len(cat_cols)} columns")
    return app


def fix_anomalies(app):
    print("\nFixing known anomalies...")
    if 'DAYS_EMPLOYED' in app.columns:
        app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].replace(365243, np.nan)
        app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].fillna(app['DAYS_EMPLOYED'].median())
    return app


def aggregate_installments(inst):
    print("\nAggregating installments...")
    inst['payment_delay'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['is_late'] = (inst['payment_delay'] > 0).astype(int)
    inst['is_underpaid'] = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype(int)

    agg = inst.groupby('SK_ID_CURR').agg(
        inst_payment_delay_mean=('payment_delay', 'mean'),
        inst_payment_delay_std=('payment_delay', 'std'),
        inst_late_payment_count=('is_late', 'sum'),
        inst_underpaid_count=('is_underpaid', 'sum'),
        inst_total_payments=('AMT_PAYMENT', 'count'),
        inst_num_loans=('SK_ID_PREV', 'nunique')
    ).reset_index()

    print(f"  Installments aggregated: {agg.shape}")
    return agg


def aggregate_bureau(bur):
    print("\nAggregating bureau...")
    agg = bur.groupby('SK_ID_CURR').agg(
        bur_total_credits=('SK_ID_BUREAU', 'count'),
        bur_days_overdue_mean=('CREDIT_DAY_OVERDUE', 'mean'),
        bur_days_overdue_max=('CREDIT_DAY_OVERDUE', 'max'),
        bur_amt_credit_sum=('AMT_CREDIT_SUM', 'sum'),
        bur_num_loans=('SK_ID_BUREAU', 'nunique')
    ).reset_index()

    print(f"  Bureau aggregated: {agg.shape}")
    return agg


def aggregate_credit_card(cc):
    print("\nAggregating credit card...")
    agg = cc.groupby('SK_ID_CURR').agg(
        cc_balance_mean=('AMT_BALANCE', 'mean'),
        cc_balance_max=('AMT_BALANCE', 'max'),
        cc_atm_drawings_mean=('AMT_DRAWINGS_ATM_CURRENT', 'mean'),
        cc_total_drawings_mean=('AMT_DRAWINGS_CURRENT', 'mean'),
    ).reset_index()

    print(f"  Credit card aggregated: {agg.shape}")
    return agg


def build_master(app, inst, bur, cc):
    print("\nBuilding master dataframe...")

    inst_agg = aggregate_installments(inst)
    bur_agg = aggregate_bureau(bur)
    cc_agg = aggregate_credit_card(cc)

    master = app.merge(inst_agg, on='SK_ID_CURR', how='left')
    master = master.merge(bur_agg, on='SK_ID_CURR', how='left')
    master = master.merge(cc_agg, on='SK_ID_CURR', how='left')

    master = master.fillna(master.median(numeric_only=True))

    print(f"  Master dataframe shape: {master.shape}")
    return master


def preprocess(app, inst, bur, cc):
    app  = clean_dataframe(app, "application_train")
    inst = clean_dataframe(inst, "installments")
    bur  = clean_dataframe(bur, "bureau")
    cc   = clean_dataframe(cc, "credit_card_balance")

    app = fix_anomalies(app)
    app = encode_categoricals(app)

    master = build_master(app, inst, bur, cc)
    master.to_csv(PROCESSED_PATH + "master.csv", index=False)
    print("\nSaved master.csv")
    return master


if __name__ == "__main__":
    from data_loader import load_raw_data, filter_customers
    app, inst, bur, cc = load_raw_data()
    app, inst, bur, cc = filter_customers(app, inst, bur, cc)
    master = preprocess(app, inst, bur, cc)
    print("\nPreprocessing complete.")