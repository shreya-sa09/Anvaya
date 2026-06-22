import pandas as pd
import numpy as np

PROCESSED_PATH = "../data/processed/"

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
    'F13_secondary_income_Z'
]


def compute_woe_iv(df, feature, target, bins=10):
    """
    Bins a feature into quantile buckets, then computes
    WOE and IV for each bin.
    Returns: bin-level stats dataframe, scalar IV value
    """
    temp = df[[feature, target]].copy().dropna()

    # Bin into quantiles — use fewer bins if low variance
    try:
        temp['bin'] = pd.qcut(temp[feature], q=bins, duplicates='drop')
    except Exception:
        temp['bin'] = pd.cut(temp[feature], bins=5, duplicates='drop')

    total_events     = temp[target].sum()
    total_nonevents  = (temp[target] == 0).sum()

    if total_events == 0 or total_nonevents == 0:
        return pd.DataFrame(), 0.0

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
    grouped['feature'] = feature

    return grouped, round(iv, 4)


def apply_woe_transform(df, feature, bin_stats):
    """
    Maps each value in the feature column to its WOE value
    using the bin boundaries learned during compute_woe_iv.
    """
    woe_col = feature.replace('_Z', '_WOE')
    temp = df[[feature]].copy()

    # Reconstruct bin intervals from the groupby output
    bin_stats_sorted = bin_stats.sort_values('bin')
    bins_list = bin_stats_sorted['bin'].tolist()
    woe_list  = bin_stats_sorted['woe'].tolist()

    def lookup_woe(val):
        for interval, woe in zip(bins_list, woe_list):
            try:
                if val in interval:
                    return woe
            except TypeError:
                pass
        # Fallback: nearest bin
        return woe_list[0]

    df[woe_col] = df[feature].apply(lookup_woe)
    return df


def run_woe_transformation(df, target_col='TARGET'):
    print("Running WOE Transformation...")
    print(f"  Input shape: {df.shape}")

    iv_summary = []
    all_bin_stats = {}

    for col in Z_COLS:
        if col not in df.columns:
            print(f"  Skipping {col} — not in dataframe")
            continue

        bin_stats, iv = compute_woe_iv(df, col, target_col)

        if bin_stats.empty:
            print(f"  {col}: skipped (no variance)")
            continue

        all_bin_stats[col] = bin_stats
        df = apply_woe_transform(df, col, bin_stats)

        strength = (
            "STRONG"   if iv >= 0.3  else
            "MEDIUM"   if iv >= 0.1  else
            "WEAK"     if iv >= 0.02 else
            "USELESS"
        )
        iv_summary.append({'feature': col, 'iv': iv, 'strength': strength})
        print(f"  {col}: IV={iv:.4f}  [{strength}]")

    iv_df = pd.DataFrame(iv_summary).sort_values('iv', ascending=False)

    print(f"\n  WOE columns added: {len(iv_summary)}")
    print(f"\n  Features to KEEP (IV >= 0.02):")
    keep = iv_df[iv_df['iv'] >= 0.02]
    print(keep.to_string(index=False))

    print(f"\n  Features to DROP (IV < 0.02):")
    drop = iv_df[iv_df['iv'] < 0.02]
    print(drop.to_string(index=False))

    return df, iv_df, all_bin_stats


if __name__ == "__main__":
    print("Loading features_normalized.csv...")
    df = pd.read_csv(PROCESSED_PATH + "features_normalized.csv")
    print(f"  Loaded: {df.shape}")

    df, iv_df, bin_stats = run_woe_transformation(df)

    # Save full output with WOE columns added
    df.to_csv(PROCESSED_PATH + "features_woe.csv", index=False)
    print(f"\nSaved features_woe.csv — shape: {df.shape}")

    # Save IV summary
    iv_df.to_csv(PROCESSED_PATH + "iv_summary.csv", index=False)
    print("Saved iv_summary.csv")

    print("\nWOE Transformation complete.")