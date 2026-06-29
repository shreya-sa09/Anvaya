import os
import re
import pandas as pd
import numpy as np

PROCESSED_PATH = "../data/processed/"
KEEP_WOE_COLS = [
    'F3_income_irregularity_WOE',
    'F7_cash_hoarding_WOE',
    'F8_stress_velocity_WOE',
    'F2_savings_drawdown_WOE',
    'F9_payment_timing_entropy_WOE',
    'F11_overdraft_freq_WOE',
    'F10_cohort_stress_WOE',
    'mean_balance_WOE',
    'std_balance_WOE',
    'mean_daily_change_WOE',
    'negative_day_rate_WOE',
    'F14_ext_source_2_WOE',
    'F15_ext_source_3_WOE',
    'F16_days_birth_WOE',
    'F17_days_employed_WOE',
    'F18_phone_change_WOE'
]


def load_woe_data(split_name):
    path = os.path.join(PROCESSED_PATH, f'{split_name}_features_woe.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{split_name}_features_woe.csv not found in {PROCESSED_PATH}. '
            'Run woe_transformer.py first.'
        )
    df = pd.read_csv(path)
    return df


def show_woe_columns(df):
    woe_cols = [c for c in df.columns if 'WOE' in c]
    print(f"\nWOE columns found: {woe_cols}")
    return woe_cols


def match_expected_columns(df):
    available = []
    missing = []
    mapped = {}

    for expected in KEEP_WOE_COLS:
        if expected in df.columns:
            available.append(expected)
            continue

        expected_base = expected.replace('_WOE', '')
        candidates = [
            c for c in df.columns
            if expected_base in c and c.endswith('WOE')
        ]

        if len(candidates) == 1:
            available.append(candidates[0])
            mapped[expected] = candidates[0]
        elif len(candidates) > 1:
            exact = [c for c in candidates if c == expected_base + '_WOE']
            chosen = exact[0] if exact else candidates[0]
            available.append(chosen)
            mapped[expected] = chosen
        else:
            missing.append(expected)

    return available, missing, mapped


def drop_correlated(df, feature_cols, threshold=0.85):
    print(f"\nChecking correlations (threshold={threshold}) on training split...")
    if len(feature_cols) <= 1:
        print("  Zero or one feature available, no correlation drop needed.")
        return feature_cols

    corr = df[feature_cols].corr().abs()
    pairs = []
    for i, col_i in enumerate(feature_cols):
        for j, col_j in enumerate(feature_cols[i + 1 :], start=i + 1):
            pairs.append((corr.iloc[i, j], col_i, col_j))

    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    dropped = set()

    for value, col_i, col_j in pairs:
        if value < threshold:
            break
        if col_i in dropped or col_j in dropped:
            continue
        dropped.add(col_j)
        print(
            f"  Dropping {col_j} (corr={value:.3f} with {col_i})"
        )

    if not dropped:
        print("  No highly correlated pairs found")

    final = [c for c in feature_cols if c not in dropped]
    print(f"  Features after correlation check: {len(final)}")
    return final


def deduplicate_customers(df):
    print("Fixing duplicate customer rows and SK_ID_CURR type...")

    required = ['SK_ID_CURR', 'TARGET']
    for col in required:
        if col not in df.columns:
            raise KeyError(
                f'Missing required column {col} in features_woe.csv'
            )

    # Normalize SK_ID_CURR to integer IDs before deduplication.
    df['SK_ID_CURR'] = pd.to_numeric(df['SK_ID_CURR'], errors='coerce').round().astype('Int64')
    if df['SK_ID_CURR'].isnull().any():
        raise ValueError('SK_ID_CURR contains non-numeric values after coercion.')

    df = df.sort_values('TARGET', ascending=False)
    before = df.shape[0]
    df = df.drop_duplicates(subset='SK_ID_CURR', keep='first')
    after = df.shape[0]
    df = df.reset_index(drop=True)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

    print(f"  Rows before dedup: {before}")
    print(f"  Rows after dedup:  {after}")
    return df


def build_final_matrix(df, final_cols):
    keep = ['SK_ID_CURR', 'TARGET'] + final_cols
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in final matrix: {missing}. "
            "Check features_woe.csv and selected feature list."
        )

    final_df = df[keep].copy().fillna(0)
    print(f"  Matrix shape: {final_df.shape}")
    return final_df


def run_feature_selection_leakage_free():
    print('=== Leakage-Free Feature Selector ===')
    
    # Load splits
    train_raw = load_woe_data('train')
    val_raw = load_woe_data('val')
    holdout_raw = load_woe_data('holdout')

    show_woe_columns(train_raw)
    available, missing, mapped = match_expected_columns(train_raw)

    print('\nExpected WOE columns:')
    for col in KEEP_WOE_COLS:
        present = 'FOUND' if (col in available or col in mapped) else 'MISSING'
        mapped_name = f" -> {mapped[col]}" if col in mapped else ''
        print(f"  {col}: {present}{mapped_name}")

    if missing:
        print('\nWARNING: Some expected WOE columns are missing. Proceeding with available columns.')
        print(f"  Missing columns: {missing}")

    if not available:
        raise ValueError(
            'No expected WOE features were found in train_features_woe.csv.'
        )

    # Perform correlation drop on the training split only!
    final_cols = drop_correlated(train_raw, available)

    # Deduplicate each split
    train_dedup = deduplicate_customers(train_raw)
    val_dedup = deduplicate_customers(val_raw)
    holdout_dedup = deduplicate_customers(holdout_raw)

    # Build final matrices using selected features
    train_final = build_final_matrix(train_dedup, final_cols)
    val_final = build_final_matrix(val_dedup, final_cols)
    holdout_final = build_final_matrix(holdout_dedup, final_cols)

    # Save outputs
    train_final.to_csv(os.path.join(PROCESSED_PATH, 'train_features_final.csv'), index=False)
    val_final.to_csv(os.path.join(PROCESSED_PATH, 'val_features_final.csv'), index=False)
    holdout_final.to_csv(os.path.join(PROCESSED_PATH, 'holdout_features_final.csv'), index=False)
    print('Saved train/val/holdout final feature matrices.')

    # Save backward compatibility placeholder file features_final.csv (points to holdout split or full)
    # The API will read features_final.csv for customer ID lookup, so we can save a combined version there
    combined_final = pd.concat([train_final, val_final, holdout_final], axis=0).drop_duplicates(subset='SK_ID_CURR')
    combined_final.to_csv(os.path.join(PROCESSED_PATH, 'features_final.csv'), index=False)
    print('Saved combined features_final.csv for API lookup.')

    selected_df = pd.DataFrame({'selected_features': final_cols})
    selected_df.to_csv(os.path.join(PROCESSED_PATH, 'selected_features.csv'), index=False)
    print('Saved selected_features.csv')

    print('\nSelected features:')
    for feature in final_cols:
        print(f"    {feature}")
    print('\nFeature selector complete.')


def run():
    run_feature_selection_leakage_free()


if __name__ == '__main__':
    run()
