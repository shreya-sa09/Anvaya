import pandas as pd
import numpy as np
import os
import json

PROCESSED_PATH = "../data/processed/"
MODELS_PATH = "../models/"

Z_COLS = [
    'F1_emi_to_income_Z',
    'F2_savings_drawdown_Z',
    'F3_income_irregularity_Z',
    'F4_spend_shift_Z',
    'F5_autodebit_failures_Z',
    'F6_lending_app_count_Z',
    'F7_cash_hoarding_Z',
    'F8_stress_velocity_Z',
    'F9_payment_timing_entropy_Z',
    'F10_cohort_stress_Z',
    'F11_overdraft_freq_Z',
    'F12_cross_loan_consistency_Z',
    'F13_secondary_income_Z',
    'F14_ext_source_2_Z',
    'F15_ext_source_3_Z',
    'F16_days_birth_Z',
    'F17_days_employed_Z',
    'F18_phone_change_Z',
    'mean_balance_Z',
    'std_balance_Z',
    'mean_daily_change_Z',
    'negative_day_rate_Z'
]


def compute_woe_iv(df, feature, target, bins=10):
    """
    Bins a feature into quantile buckets, then computes
    WOE and IV for each bin.
    Returns: mapping dictionary (edges, woes), scalar IV value
    """
    temp = df[[feature, target]].copy().dropna()

    # Bin into quantiles — use fewer bins if low variance
    try:
        temp['bin'] = pd.qcut(temp[feature], q=bins, duplicates='drop')
    except Exception:
        try:
            temp['bin'] = pd.cut(temp[feature], bins=5, duplicates='drop')
        except Exception:
            return None, 0.0

    total_events     = temp[target].sum()
    total_nonevents  = (temp[target] == 0).sum()

    if total_events == 0 or total_nonevents == 0:
        return None, 0.0

    grouped = temp.groupby('bin', observed=True)[target].agg(
        events=lambda x: x.sum(),
        nonevents=lambda x: (x == 0).sum()
    ).reset_index()

    grouped['dist_events']    = grouped['events']    / total_events
    grouped['dist_nonevents'] = grouped['nonevents'] / total_nonevents

    # Replace zeros to avoid log(0)
    grouped['dist_events']    = grouped['dist_events'].replace(0, 0.0001)
    grouped['dist_nonevents'] = grouped['dist_nonevents'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_nonevents'])
    grouped['iv']  = (grouped['dist_events'] - grouped['dist_nonevents']) * grouped['woe']

    iv = grouped['iv'].sum()
    
    # Extract sorted boundaries
    grouped['left'] = grouped['bin'].apply(lambda x: float(x.left))
    grouped = grouped.sort_values('left')
    
    # We construct boundaries. To cover all values from -inf to inf:
    edges = [-999999999.0]
    for idx, row in grouped.iterrows():
        edges.append(float(row['bin'].right))
    edges[-1] = 999999999.0
    
    woes = [float(w) for w in grouped['woe'].tolist()]
    
    mapping = {
        'edges': edges,
        'woes': woes
    }

    return mapping, round(iv, 4)


def apply_woe_mapping(df, feature, mapping):
    """
    Maps each value in the feature column to its WOE value
    using the pre-computed bin boundaries.
    """
    woe_col = feature.replace('_Z', '_WOE')
    if mapping is None:
        df[woe_col] = 0.0
        return df

    edges = mapping['edges']
    woes = mapping['woes']

    # Use pd.cut to assign categories (indices of woes)
    binned = pd.cut(df[feature], bins=edges, labels=False, include_lowest=True)
    
    # Fill nan with default woe (e.g. 0.0)
    df[woe_col] = binned.map(lambda idx: woes[int(idx)] if not pd.isna(idx) else 0.0)
    return df


def train_woe_mappings(train_df, target_col='TARGET'):
    print("Training WoE Mappings on train split...")
    mappings = {}
    iv_summary = []

    for col in Z_COLS:
        if col not in train_df.columns:
            continue

        mapping, iv = compute_woe_iv(train_df, col, target_col)
        if mapping is None:
            print(f"  {col}: skipped (no variance)")
            continue

        mappings[col] = mapping
        strength = (
            "STRONG"   if iv >= 0.3  else
            "MEDIUM"   if iv >= 0.1  else
            "WEAK"     if iv >= 0.02 else
            "USELESS"
        )
        iv_summary.append({'feature': col, 'iv': iv, 'strength': strength})
        print(f"  {col}: IV={iv:.4f}  [{strength}]")

    iv_df = pd.DataFrame(iv_summary).sort_values('iv', ascending=False)
    return mappings, iv_df


def save_woe_mappings(mappings, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved WoE mappings to {filepath}")


def load_woe_mappings(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def apply_all_woe_mappings(df, mappings):
    df = df.copy()
    for col, mapping in mappings.items():
        df = apply_woe_mapping(df, col, mapping)
    return df


def run_woe_transformation(df, target_col='TARGET'):
    # Backward compatibility function (will train and apply on same df, causing leakage - for demo/compat only)
    print("WARNING: run_woe_transformation contains target leakage. Use train_woe_mappings instead.")
    mappings, iv_df = train_woe_mappings(df, target_col)
    transformed_df = apply_all_woe_mappings(df, mappings)
    
    # Return in original signature format: df, iv_df, bin_stats
    # Let's map bin_stats to a dummy dictionary for backward compatibility
    return transformed_df, iv_df, mappings


if __name__ == "__main__":
    print("Running WoE standalone test...")
    df = pd.read_csv(PROCESSED_PATH + "features_normalized.csv")
    transformed, iv_df, mappings = run_woe_transformation(df)
    transformed.to_csv(PROCESSED_PATH + "features_woe.csv", index=False)
    iv_df.to_csv(PROCESSED_PATH + "iv_summary.csv", index=False)
    save_woe_mappings(mappings, MODELS_PATH + "woe_mappings.json")
    print("WoE Transformation complete.")