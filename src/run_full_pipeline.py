import sys
sys.path.insert(0, '.')
import pandas as pd
from data_loader import load_raw_data, filter_customers
from preprocessor import preprocess
import synthetic_generator
import feature_engineering
import personal_rhythm_engine
import woe_transformer

print('Loading raw data...')
app, inst, bur, cc = load_raw_data()
app, inst, bur, cc = filter_customers(app, inst, bur, cc)
master = preprocess(app, inst, bur, cc)

print('Generating synthetic data...')
synthetic_generator.generate_synthetic_data(master)

print('Building features...')
master, balance, income, txn = feature_engineering.load_all_data()
features_df = feature_engineering.build_features(master, balance, income, txn)
feature_cols = ['SK_ID_CURR', 'TARGET'] + [c for c in features_df.columns if c.startswith('F')]
features_df[feature_cols].to_csv('../data/processed/features.csv', index=False)
print('Saved features.csv')

print('Running personal rhythm engine...')
normalized_df, pop_stats = personal_rhythm_engine.run_personal_rhythm_engine(features_df)
normalized_df.to_csv('../data/processed/features_normalized.csv', index=False)
personal_rhythm_engine.save_population_baseline(pop_stats)
print('Saved features_normalized.csv and population_baseline.csv')

print('Computing WOE/IV scores...')
features_norm = pd.read_csv('../data/processed/features_normalized.csv')
_, iv_df, _ = woe_transformer.run_woe_transformation(features_norm, target_col='TARGET')
print(iv_df[['feature', 'iv']].to_string(index=False))
