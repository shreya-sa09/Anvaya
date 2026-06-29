import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')

from data_loader import load_raw_data, filter_customers
from preprocessor import preprocess
import synthetic_generator
import feature_engineering
import personal_rhythm_engine
import woe_transformer
import feature_selector
import model_trainer
import shap_explainer

print('=== ANVAYA — Leakage-Free Full Pipeline ===')

print('\n[1/8] Loading raw data...')
app, inst, bur, cc = load_raw_data()
app, inst, bur, cc = filter_customers(app, inst, bur, cc)

print('\n[2/8] Preprocessing...')
master = preprocess(app, inst, bur, cc)

print('\n[3/8] Checking/Generating synthetic data...')
synthetic_files = [
    '../data/synthetic/synthetic_daily_balance.csv',
    '../data/synthetic/synthetic_income_arrival.csv',
    '../data/synthetic/synthetic_transactions.csv'
]
if all(os.path.exists(f) for f in synthetic_files):
    print("  Preserving existing synthetic dataset (skipping generation).")
else:
    print("  Generating synthetic data...")
    synthetic_generator.generate_synthetic_data(master)

print('\n[4/8] Building features...')
master, balance, income, txn = feature_engineering.load_all_data()
features_df = feature_engineering.build_features(
    master, balance, income, txn
)

# Deduplicate by customer ID early to avoid overlaps
features_df = features_df.drop_duplicates(subset='SK_ID_CURR').reset_index(drop=True)

feature_cols = ['SK_ID_CURR', 'TARGET'] + [
    c for c in features_df.columns if c.startswith('F')
]
features_df = features_df[feature_cols]
features_df.to_csv('../data/processed/features.csv', index=False)
print(f"  Total customers: {features_df.shape[0]}")

# Perform stratified train/validation/holdout split early to prevent leakage
train_df, temp_df = train_test_split(
    features_df,
    test_size=0.30,
    random_state=42,
    stratify=features_df['TARGET']
)
val_df, holdout_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df['TARGET']
)

print('\n[5/8] Running Leakage-Free Personal Rhythm Engine (Scaling)...')
train_norm, val_norm, holdout_norm, pop_stats = personal_rhythm_engine.run_personal_rhythm_engine_leakage_free(
    train_df, val_df, holdout_df
)

train_norm.to_csv('../data/processed/train_features_normalized.csv', index=False)
val_norm.to_csv('../data/processed/val_features_normalized.csv', index=False)
holdout_norm.to_csv('../data/processed/holdout_features_normalized.csv', index=False)
personal_rhythm_engine.save_population_baseline(pop_stats)

# Save combined normalized features for backward compatibility
combined_norm = pd.concat([train_norm, val_norm, holdout_norm], axis=0)
combined_norm.to_csv('../data/processed/features_normalized.csv', index=False)

print('\n[6/8] Leakage-Free WOE Transformation...')
mappings, iv_df = woe_transformer.train_woe_mappings(train_norm, target_col='TARGET')
woe_transformer.save_woe_mappings(mappings, '../models/woe_mappings.json')

train_woe = woe_transformer.apply_all_woe_mappings(train_norm, mappings)
val_woe = woe_transformer.apply_all_woe_mappings(val_norm, mappings)
holdout_woe = woe_transformer.apply_all_woe_mappings(holdout_norm, mappings)

train_woe.to_csv('../data/processed/train_features_woe.csv', index=False)
val_woe.to_csv('../data/processed/val_features_woe.csv', index=False)
holdout_woe.to_csv('../data/processed/holdout_features_woe.csv', index=False)
iv_df.to_csv('../data/processed/iv_summary.csv', index=False)

# Save combined WOE features for backward compatibility
combined_woe = pd.concat([train_woe, val_woe, holdout_woe], axis=0)
combined_woe.to_csv('../data/processed/features_woe.csv', index=False)

print('\n[7/8] Leakage-Free Feature Selection...')
feature_selector.run()

print('\n[8/8] Model Training + SHAP Explanations...')
model_trainer.run()
shap_explainer.run()

print('\n=== ANVAYA Pipeline Complete ===')