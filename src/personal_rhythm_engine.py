import pandas as pd
import numpy as np
import os

PROCESSED_PATH = "../data/processed/"
SYNTHETIC_BALANCE_PATH = "../data/synthetic/synthetic_daily_balance.csv"

FEATURE_COLS = [
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

BALANCE_METRICS = [
    'mean_balance',
    'std_balance',
    'mean_daily_change',
    'negative_day_rate'
]


def load_balance_data():
    return pd.read_csv(SYNTHETIC_BALANCE_PATH)


def compute_window_metrics(df):
    if df.empty:
        return pd.Series({
            'mean_balance': 0.0,
            'std_balance': 0.0,
            'mean_daily_change': 0.0,
            'negative_day_rate': 0.0,
            'daily_change_std': 0.0
        })

    df = df.sort_values('date')
    mean_balance = df['closing_balance'].mean()
    std_balance = df['closing_balance'].std()
    daily_change = df['closing_balance'].diff().fillna(0.0)
    mean_daily_change = daily_change.mean()
    negative_day_rate = (df['closing_balance'] < 0).mean()
    daily_change_std = daily_change.std()

    return pd.Series({
        'mean_balance': mean_balance,
        'std_balance': std_balance,
        'mean_daily_change': mean_daily_change,
        'negative_day_rate': negative_day_rate,
        'daily_change_std': daily_change_std
    })


def compute_personal_baseline_zscores(balance_df):
    print("Computing personal baseline metrics from daily balance...")
    baseline = balance_df[balance_df['date'] <= 90]
    observation = balance_df[balance_df['date'] > 90]

    baseline_stats = baseline.groupby('customer_id').apply(compute_window_metrics).reset_index()
    observation_stats = observation.groupby('customer_id').apply(compute_window_metrics).reset_index()

    metrics = observation_stats.merge(
        baseline_stats,
        on='customer_id',
        how='left',
        suffixes=('_obs', '_base')
    ).fillna(0.0)

    metrics['mean_balance_Z'] = (
        metrics['mean_balance_obs'] - metrics['mean_balance_base']
    ) / (metrics['std_balance_base'] + 0.001)

    metrics['std_balance_Z'] = (
        metrics['std_balance_obs'] - metrics['std_balance_base']
    ) / (metrics['std_balance_base'] + 0.001)

    metrics['mean_daily_change_Z'] = (
        metrics['mean_daily_change_obs'] - metrics['mean_daily_change_base']
    ) / (metrics['daily_change_std_base'] + 0.001)

    metrics['negative_day_rate_Z'] = (
        metrics['negative_day_rate_obs'] - metrics['negative_day_rate_base']
    ) / (metrics['negative_day_rate_base'] + 0.001)

    z_scores = metrics[['customer_id', 'mean_balance_Z', 'std_balance_Z',
                        'mean_daily_change_Z', 'negative_day_rate_Z']].copy()
    z_scores = z_scores.rename(columns={'customer_id': 'SK_ID_CURR'})
    return z_scores


def compute_population_stats(features_df):
    stats = {}
    for col in FEATURE_COLS:
        if col not in features_df.columns:
            continue
        mean = features_df[col].median()  # Using median as robust mean estimator
        std = features_df[col].std()
        stats[col] = {
            'mean': float(mean),
            'std': float(std if std > 0 else 1.0)
        }
    return stats


def normalize_original_features(features_df, pop_stats):
    df = features_df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        mean = pop_stats[col]['mean']
        std = pop_stats[col]['std']
        z_col = col + '_Z'
        df[z_col] = ((df[col] - mean) / std).clip(-5, 5).round(4)
    return df


def run_personal_rhythm_engine_leakage_free(train_df, val_df, holdout_df):
    print("Running leakage-free Personal Rhythm Engine...")
    balance_df = load_balance_data()
    personal_z = compute_personal_baseline_zscores(balance_df)

    # Compute population statistics on the train split only to prevent leakage
    pop_stats = compute_population_stats(train_df)

    # Normalize each split using the train statistics
    train_norm = normalize_original_features(train_df, pop_stats)
    val_norm = normalize_original_features(val_df, pop_stats)
    holdout_norm = normalize_original_features(holdout_df, pop_stats)

    # Merge customer-specific Z-scores (no leakage since they are computed customer-by-customer)
    train_result = train_norm.merge(personal_z, on='SK_ID_CURR', how='left')
    val_result = val_norm.merge(personal_z, on='SK_ID_CURR', how='left')
    holdout_result = holdout_norm.merge(personal_z, on='SK_ID_CURR', how='left')

    z_cols = ['mean_balance_Z', 'std_balance_Z', 'mean_daily_change_Z', 'negative_day_rate_Z']
    for df in [train_result, val_result, holdout_result]:
        df[z_cols] = df[z_cols].fillna(0.0)

    return train_result, val_result, holdout_result, pop_stats


def run_personal_rhythm_engine(features_df):
    # Backward compatibility wrapper
    print("WARNING: run_personal_rhythm_engine is deprecated and contains scaling leakage. Use run_personal_rhythm_engine_leakage_free instead.")
    balance_df = load_balance_data()
    personal_z = compute_personal_baseline_zscores(balance_df)
    pop_stats = compute_population_stats(features_df)
    base_normalized = normalize_original_features(features_df, pop_stats)
    result = base_normalized.merge(personal_z, on='SK_ID_CURR', how='left')
    result[['mean_balance_Z', 'std_balance_Z', 'mean_daily_change_Z', 'negative_day_rate_Z']] = (
        result[['mean_balance_Z', 'std_balance_Z', 'mean_daily_change_Z', 'negative_day_rate_Z']]
        .fillna(0.0)
    )
    return result, pop_stats


def save_population_baseline(pop_stats):
    rows = []
    for feature, vals in pop_stats.items():
        rows.append({
            'feature': feature,
            'population_mean': vals['mean'],
            'population_std': vals['std']
        })
    pd.DataFrame(rows).to_csv(PROCESSED_PATH + 'population_baseline.csv', index=False)


if __name__ == "__main__":
    print("Loading features.csv...")
    features_df = pd.read_csv(PROCESSED_PATH + "features.csv")
    print(f"  Loaded: {features_df.shape}")

    # For standalone runs, we split locally to demonstrate leakage-free logic
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(features_df, test_size=0.3, random_state=42, stratify=features_df['TARGET'])
    val_df, holdout_df = train_test_split(val_df, test_size=0.5, random_state=42, stratify=val_df['TARGET'])

    train_res, val_res, holdout_res, pop_stats = run_personal_rhythm_engine_leakage_free(train_df, val_df, holdout_df)

    # Save
    train_res.to_csv(PROCESSED_PATH + "train_features_normalized.csv", index=False)
    val_res.to_csv(PROCESSED_PATH + "val_features_normalized.csv", index=False)
    holdout_res.to_csv(PROCESSED_PATH + "holdout_features_normalized.csv", index=False)

    save_population_baseline(pop_stats)
    print("Saved baseline and normalized files.")
