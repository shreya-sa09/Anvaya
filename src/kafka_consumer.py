import json
import time
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'anvaya-transactions'
OUTPUT_TOPIC = 'anvaya-risk-scores'

# XGBoost trained on 5 features
XGB_COLS = [
    'F3_income_irregularity_WOE',
    'F7_cash_hoarding_WOE',
    'F8_stress_velocity_WOE',
    'F2_savings_drawdown_WOE',
    'F9_payment_timing_entropy_WOE',
]

# LightGBM trained on 7 features
LGB_COLS = [
    'F3_income_irregularity_WOE',
    'F7_cash_hoarding_WOE',
    'F8_stress_velocity_WOE',
    'F2_savings_drawdown_WOE',
    'F9_payment_timing_entropy_WOE',
    'F11_overdraft_freq_WOE',
    'F10_cohort_stress_WOE',
]

def assign_risk_band(pd_score):
    if pd_score < 0.05:
        return 'GREEN'
    elif pd_score < 0.15:
        return 'YELLOW'
    else:
        return 'RED'

def load_models():
    xgb = joblib.load('../models/xgb_model.pkl')
    lgb = joblib.load('../models/lgb_model.pkl')
    print("Models loaded successfully")
    return xgb, lgb

def score_event(event, xgb, lgb):
    features = event.get('features', {})

    # Separate arrays for each model
    X_xgb = np.array([[features.get(col, 0.0) for col in XGB_COLS]])
    X_lgb = pd.DataFrame(
        [[features.get(col, 0.0) for col in LGB_COLS]],
        columns=LGB_COLS
    )

    xgb_score = float(xgb.predict_proba(X_xgb)[0][1])
    lgb_score = float(lgb.predict_proba(X_lgb)[0][1])

    # Gating layer
    if 0.15 <= xgb_score <= 0.25:
        final_pd = 0.4 * xgb_score + 0.6 * lgb_score
    else:
        final_pd = xgb_score

    return {
        'customer_id': event['customer_id'],
        'timestamp': datetime.now().isoformat(),
        'event_type': event.get('event_type'),
        'xgb_score': round(xgb_score, 4),
        'lgb_score': round(lgb_score, 4),
        'final_pd': round(final_pd, 4),
        'risk_band': assign_risk_band(final_pd),
    }

def run_consumer(max_messages=50):
    xgb, lgb = load_models()

    # Warmup — eliminates cold-start latency on first real message
    dummy_features = {col: 0.0 for col in LGB_COLS}
    dummy_event = {
        'customer_id': 0,
        'event_type': 'warmup',
        'features': dummy_features
    }
    score_event(dummy_event, xgb, lgb)
    print("Models warmed up")

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        consumer_timeout_ms=15000,
        group_id='anvaya-scorer-v2'
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    print(f"Listening on {INPUT_TOPIC}...")
    count = 0
    latencies = []

    for message in consumer:
        if count >= max_messages:
            break

        start = time.time()
        event = message.value
        result = score_event(event, xgb, lgb)
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
        print("="*50)
        print("LATENCY SUMMARY")
        print(f"  Messages processed : {count}")
        print(f"  Avg latency        : {np.mean(latencies):.1f}ms")
        print(f"  Max latency        : {np.max(latencies):.1f}ms")
        print(f"  Target             : under 800ms")
        print(f"  Status             : {'PASS' if np.max(latencies) < 800 else 'FAIL'}")
        print("="*50)
    else:
        print("No messages received — make sure producer runs after consumer starts")

if __name__ == "__main__":
    run_consumer()