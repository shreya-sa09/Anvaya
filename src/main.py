import sys
import pandas as pd

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

print('=== ANVAYA — Full Pipeline ===')

print('\n[1/8] Loading raw data...')
app, inst, bur, cc = load_raw_data()
app, inst, bur, cc = filter_customers(app, inst, bur, cc)

print('\n[2/8] Preprocessing...')
master = preprocess(app, inst, bur, cc)

print('\n[3/8] Generating synthetic data...')
synthetic_generator.generate_synthetic_data(master)

print('\n[4/8] Building features...')
master, balance, income, txn = feature_engineering.load_all_data()
features_df = feature_engineering.build_features(
    master, balance, income, txn
)
feature_cols = ['SK_ID_CURR', 'TARGET'] + [
    c for c in features_df.columns if c.startswith('F')
]
features_df[feature_cols].to_csv(
    '../data/processed/features.csv', index=False
)

print('\n[5/8] Running Personal Rhythm Engine...')
normalized_df, pop_stats = personal_rhythm_engine.run_personal_rhythm_engine(
    features_df
)
normalized_df.to_csv(
    '../data/processed/features_normalized.csv', index=False
)
personal_rhythm_engine.save_population_baseline(pop_stats)

print('\n[6/8] WOE Transformation...')
features_norm = pd.read_csv(
    '../data/processed/features_normalized.csv'
)
woe_df, iv_df, bin_stats = woe_transformer.run_woe_transformation(
    features_norm, target_col='TARGET'
)
woe_df.to_csv('../data/processed/features_woe.csv', index=False)
iv_df.to_csv('../data/processed/iv_summary.csv', index=False)

print('\n[7/8] Feature Selection...')
feature_selector.run()

print('\n[8/8] Model Training + SHAP...')
model_trainer.run()
shap_explainer.run()

print('\n=== ANVAYA Pipeline Complete ===')