
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

PROCESSED_PATH = "../data/processed/"
MODELS_PATH = "../models/"

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 11,
    'random_state': 42,
    'eval_metric': 'auc',
    'use_label_encoder': False,
}

LGB_PARAMS = {
    'n_estimators': 150,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1,
}


def load_features():
    path = os.path.join(PROCESSED_PATH, 'features_final.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'features_final.csv not found in {PROCESSED_PATH}. Run feature_selector.py first.'
        )
    df = pd.read_csv(path)
    return df


def load_selected_features():
    path = os.path.join(PROCESSED_PATH, 'selected_features.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'selected_features.csv not found in {PROCESSED_PATH}. Run feature_selector.py first.'
        )
    sel = pd.read_csv(path)
    if 'selected_features' not in sel.columns:
        raise ValueError('selected_features.csv must contain a selected_features column.')
    return sel['selected_features'].tolist()


def stratified_splits(df, feature_cols):
    print('Splitting data into train/validation/holdout...')

    if 'TARGET' not in df.columns:
        raise KeyError('TARGET column missing from features_final.csv')
    if 'SK_ID_CURR' not in df.columns:
        raise KeyError('SK_ID_CURR column missing from features_final.csv')

    X = df[feature_cols].fillna(0)
    y = df['TARGET']

    X_temp, X_hold, y_temp, y_hold, id_temp, id_hold = train_test_split(
        X, y, df['SK_ID_CURR'],
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_temp, y_temp, id_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=y_temp,
    )

    print(f"  Train size: {X_train.shape[0]}  default rate: {y_train.mean():.4f}")
    print(f"  Val size:   {X_val.shape[0]}  default rate: {y_val.mean():.4f}")
    print(f"  Holdout size: {X_hold.shape[0]}  default rate: {y_hold.mean():.4f}")

    return (
        X_train, X_val, X_hold,
        y_train, y_val, y_hold,
        id_train, id_val, id_hold,
    )


def train_xgb(X_train, y_train, xgb_features):
    print('\nTraining XGBoost model...')
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train[xgb_features], y_train)
    return model


def train_lgb(X_train, y_train, lgb_features):
    print('\nTraining LightGBM model...')
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train[lgb_features], y_train)
    return model


def apply_gating(xgb_model, lgb_model, X, xgb_features, lgb_features):
    print('\nApplying dual-track gating...')
    xgb_score = xgb_model.predict_proba(X[xgb_features])[:, 1]
    lgb_score = lgb_model.predict_proba(X[lgb_features])[:, 1]
    final_score = xgb_score.copy()
    uncertain = (xgb_score >= 0.15) & (xgb_score <= 0.25)
    final_score[uncertain] = lgb_score[uncertain]
    print(f"  Routed {int(uncertain.sum())} customers to LightGBM in uncertain zone")
    return xgb_score, lgb_score, final_score


def train_meta_model(xgb_score, lgb_score, y_val):
    print('\nTraining meta logistic regression...')
    meta_X = np.vstack([xgb_score, lgb_score]).T
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X, y_val)
    return meta_model


def calibrate_meta_model(meta_model, xgb_score, lgb_score, y_val):
    print('\nCalibrating final probability with isotonic regression...')
    meta_X = np.vstack([xgb_score, lgb_score]).T
    calibrated = CalibratedClassifierCV(meta_model, method='isotonic', cv=3)
    calibrated.fit(meta_X, y_val)
    return calibrated


def compute_ks(y_true, y_prob):
    from scipy.stats import ks_2samp
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(pos, neg)
    return ks_stat * 100


def compute_gini(auc):
    return 2 * auc - 1


def risk_band(pd_score):
    if pd_score < 0.05:
        return 'GREEN'
    if pd_score < 0.15:
        return 'YELLOW'
    if pd_score < 0.25:
        return 'HIGH'
    return 'RED'


def evaluate_holdout(y_true, y_pred_prob):
    auc = roc_auc_score(y_true, y_pred_prob)
    ks = compute_ks(y_true, y_pred_prob)
    gini = compute_gini(auc)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print('\nFinal holdout evaluation:')
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  KS Statistic     : {ks:.2f}")
    print(f"  Gini Coefficient : {gini:.4f}")
    print(f"  Precision @0.5   : {precision:.4f}")
    print(f"  Recall @0.5      : {recall:.4f}")
    print(f"  F1 Score @0.5    : {f1:.4f}")
    print(f"  Confusion matrix :\n{cm}")
    return auc, ks, gini, precision, recall, f1, cm


def ensure_models_folder():
    os.makedirs(MODELS_PATH, exist_ok=True)


def save_models(xgb_model, lgb_model, meta_model, calibrated_model):
    ensure_models_folder()
    joblib.dump(xgb_model, os.path.join(MODELS_PATH, 'xgb_model.pkl'))
    joblib.dump(lgb_model, os.path.join(MODELS_PATH, 'lgb_model.pkl'))
    joblib.dump(meta_model, os.path.join(MODELS_PATH, 'meta_model.pkl'))
    joblib.dump(calibrated_model, os.path.join(MODELS_PATH, 'calibrated_model.pkl'))
    print('\nSaved models to models/')


def save_predictions(df, filename):
    path = os.path.join(PROCESSED_PATH, filename)
    df.to_csv(path, index=False)
    print(f'Saved {filename} — shape: {df.shape}')


def run():
    print('=== Model Trainer ===')
    df = load_features()
    print(f"Loaded features_final.csv — shape: {df.shape}")
    feature_cols = load_selected_features()
    print(f"Loaded selected_features.csv ({len(feature_cols)} features)")

    xgb_features = feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
    lgb_features = feature_cols
    print(f"  XGBoost features ({len(xgb_features)}): {xgb_features}")
    print(f"  LightGBM features ({len(lgb_features)}): {lgb_features}")

    X_train, X_val, X_hold, y_train, y_val, y_hold, id_train, id_val, id_hold = stratified_splits(df, feature_cols)

    xgb_model = train_xgb(X_train, y_train, xgb_features)
    lgb_model = train_lgb(X_train, y_train, lgb_features)

    print('\nScoring validation set...')
    xgb_val, lgb_val, _ = apply_gating(xgb_model, lgb_model, X_val, xgb_features, lgb_features)
    meta_model = train_meta_model(xgb_val, lgb_val, y_val)
    calibrated_model = calibrate_meta_model(meta_model, xgb_val, lgb_val, y_val)

    print('\nScoring holdout set...')
    xgb_hold = xgb_model.predict_proba(X_hold[xgb_features])[:, 1]
    lgb_hold = lgb_model.predict_proba(X_hold[lgb_features])[:, 1]
    meta_hold_X = np.vstack([xgb_hold, lgb_hold]).T
    final_hold = calibrated_model.predict_proba(meta_hold_X)[:, 1]

    auc, ks, gini, precision, recall, f1, cm = evaluate_holdout(y_hold, final_hold)

    band_series = pd.Series(final_hold).apply(risk_band)
    distribution = band_series.value_counts().reindex(['GREEN', 'YELLOW', 'HIGH', 'RED'], fill_value=0)
    print('\nRisk band distribution (holdout):')
    print(distribution.to_string())

    train_results = pd.DataFrame({
        'SK_ID_CURR': id_train.values,
        'TARGET': y_train.values,
        'xgb_score': xgb_model.predict_proba(X_train[xgb_features])[:, 1],
        'lgb_score': lgb_model.predict_proba(X_train[lgb_features])[:, 1],
    })
    train_results['final_pd'] = pd.Series(train_results['xgb_score']).where(
        (train_results['xgb_score'] < 0.15) | (train_results['xgb_score'] > 0.25),
        train_results['lgb_score']
    )
    train_results['risk_band'] = train_results['final_pd'].apply(risk_band)

    holdout_results = pd.DataFrame({
        'SK_ID_CURR': id_hold.values,
        'TARGET': y_hold.values,
        'xgb_score': xgb_hold,
        'lgb_score': lgb_hold,
        'final_pd': final_hold,
        'risk_band': band_series.values,
    })

    save_predictions(train_results, 'train_predictions.csv')
    save_predictions(holdout_results, 'model_predictions.csv')
    save_models(xgb_model, lgb_model, meta_model, calibrated_model)

    print('\nModel trainer complete.')
    print(f"  Holdout AUC: {auc:.4f}")
    print(f"  Holdout KS: {ks:.2f}")
    print(f"  Holdout Gini: {gini:.4f}")
    print(f"  Holdout Precision: {precision:.4f}")
    print(f"  Holdout Recall: {recall:.4f}")
    print(f"  Holdout F1: {f1:.4f}")


if __name__ == '__main__':
    run()

