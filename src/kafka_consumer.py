import json
import time
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC  = 'anvaya-transactions'
OUTPUT_TOPIC = 'anvaya-risk-scores'
PROCESSED_PATH = '../data/processed/'
MODELS_PATH    = '../models/'


def load_selected_features():
    """Always load feature list from selected_features.csv — never hardcode."""
    sf  = pd.read_csv(PROCESSED_PATH + 'selected_features.csv')
    all_features = sf['selected_features'].tolist()
    xgb_features = all_features[:5]
    lgb_features = all_features
    return xgb_features, lgb_features


def load_models():
    xgb  = joblib.load(MODELS_PATH + 'xgb_model.pkl')
    lgb  = joblib.load(MODELS_PATH + 'lgb_model.pkl')
    meta = joblib.load(MODELS_PATH + 'calibrated_model.pkl')
    print("Models loaded successfully")
    return xgb, lgb, meta


def load_woe_mappings():
    with open(MODELS_PATH + 'woe_mappings.json', 'r') as f:
        return json.load(f)


def load_population_baseline():
    bl = pd.read_csv(PROCESSED_PATH + 'population_baseline.csv')
    return {r['feature']: {'mean': r['population_mean'], 'std': r['population_std']}
            for _, r in bl.iterrows()}


def apply_woe_single(value, mapping):
    if mapping is None:
        return 0.0
    edges = mapping['edges']
    woes  = mapping['woes']
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return woes[i] if i < len(woes) else 0.0
    return woes[-1] if woes else 0.0


def assign_risk_band(pd_score):
    if pd_score < 0.05:  return 'GREEN'
    if pd_score < 0.15:  return 'YELLOW'
    if pd_score < 0.25:  return 'HIGH'
    return 'RED'


def score_event(event, xgb, lgb, meta, xgb_features, lgb_features, woe_mappings, pop_baseline):
    """
    Score a raw transaction event using the full inference pipeline:
      raw_features -> PRE Z-scaling -> WoE -> ensemble -> calibrated PD
    """
    raw = event.get('features', {})

    # Step 1: Z-score normalise using training population stats
    z_dict = {}
    for feat, stats in pop_baseline.items():
        val = raw.get(feat, stats['mean'])
        z_col = feat + '_Z'
        z_dict[z_col] = float(np.clip(
            (val - stats['mean']) / (stats['std'] + 1e-9), -5, 5
        ))

    # Step 2: Apply WoE mappings
    woe_dict = {}
    for z_col, z_val in z_dict.items():
        woe_col = z_col[:-2] + '_WOE'          # strip _Z, add _WOE
        mapping = woe_mappings.get(z_col)
        woe_dict[woe_col] = apply_woe_single(z_val, mapping)

    # Step 3: Build feature arrays for each model
    X_xgb = np.array([[woe_dict.get(f, 0.0) for f in xgb_features]])
    X_lgb = pd.DataFrame(
        [[woe_dict.get(f, 0.0) for f in lgb_features]],
        columns=lgb_features
    )

    # Step 4: Score
    xgb_score = float(xgb.predict_proba(X_xgb)[0][1])
    lgb_score  = float(lgb.predict_proba(X_lgb)[0][1])

    # Step 5: Calibrated ensemble
    meta_X   = np.array([[xgb_score, lgb_score]])
    final_pd = float(meta.predict_proba(meta_X)[0][1])

    return {
        'customer_id': event['customer_id'],
        'timestamp':   datetime.now().isoformat(),
        'event_type':  event.get('event_type'),
        'xgb_score':   round(xgb_score, 4),
        'lgb_score':   round(lgb_score, 4),
        'final_pd':    round(final_pd, 4),
        'risk_band':   assign_risk_band(final_pd),
    }


def run_consumer(max_messages=50):
    xgb_features, lgb_features = load_selected_features()
    print(f"XGB features ({len(xgb_features)}): {xgb_features}")
    print(f"LGB features ({len(lgb_features)}): {lgb_features}")

    xgb, lgb, meta = load_models()
    woe_mappings   = load_woe_mappings()
    pop_baseline   = load_population_baseline()

    # Model warmup
    dummy_event = {'customer_id': 0, 'event_type': 'warmup', 'features': {}}
    score_event(dummy_event, xgb, lgb, meta, xgb_features, lgb_features, woe_mappings, pop_baseline)
    print("Models warmed up")

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        consumer_timeout_ms=15000,
        group_id='anvaya-scorer-v3'
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    print(f"Listening on {INPUT_TOPIC}...")
    count     = 0
    latencies = []

    for message in consumer:
        if count >= max_messages:
            break

        start  = time.time()
        event  = message.value
        result = score_event(
            event, xgb, lgb, meta,
            xgb_features, lgb_features,
            woe_mappings, pop_baseline
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        producer.send(OUTPUT_TOPIC, value=result)
        count += 1

        print(f"  [{count}] Customer {result['customer_id']} → "
              f"PD={result['final_pd']:.4f} | "
              f"Band={result['risk_band']} | "
              f"Latency={latency_ms:.1f}ms")

    consumer.close()
    producer.flush()
    producer.close()

    if latencies:
        print()
        print("=" * 50)
        print("LATENCY SUMMARY")
        print(f"  Messages processed : {count}")
        print(f"  Avg latency        : {np.mean(latencies):.1f}ms")
        print(f"  Max latency        : {np.max(latencies):.1f}ms")
        print(f"  Target             : under 800ms")
        print(f"  Status             : {'PASS' if np.max(latencies) < 800 else 'FAIL'}")
        print("=" * 50)
    else:
        print("No messages received — make sure producer runs after consumer starts")


if __name__ == "__main__":
    run_consumer()
