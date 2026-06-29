import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

KAFKA_BROKER   = 'localhost:9092'
TOPIC          = 'anvaya-transactions'
PROCESSED_PATH = '../data/processed/'
SYNTH_PATH     = '../data/synthetic/'


def run_producer(n_events=50, delay=0.2):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load master data for raw feature values (not WoE — that happens in the consumer)
    master   = pd.read_csv(PROCESSED_PATH + 'master.csv')
    features = pd.read_csv(PROCESSED_PATH + 'features.csv')
    features['SK_ID_CURR'] = features['SK_ID_CURR'].astype(float).astype(int)
    master['SK_ID_CURR']   = master['SK_ID_CURR'].astype(float).astype(int)

    # Merge to get raw feature values
    merged = features.merge(
        master[['SK_ID_CURR', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE']],
        on='SK_ID_CURR', how='left'
    )

    print(f"Loaded {len(merged)} customers for event simulation")
    print(f"Sending {n_events} raw transaction events to topic: {TOPIC}")

    RAW_FEATURE_COLS = [
        'F1_emi_to_income', 'F2_savings_drawdown', 'F3_income_irregularity',
        'F4_spend_shift', 'F5_autodebit_failures', 'F6_lending_app_count',
        'F7_cash_hoarding', 'F8_stress_velocity', 'F9_payment_timing_entropy',
        'F10_cohort_stress', 'F11_overdraft_freq', 'F12_cross_loan_consistency',
        'F13_secondary_income',
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE'
    ]

    EVENT_TYPES = [
        'upi_transaction', 'atm_withdrawal', 'auto_debit',
        'salary_credit', 'balance_check'
    ]

    for i in range(n_events):
        row = merged.sample(1).iloc[0]

        # Add small noise to simulate real-time variation
        noisy_features = {}
        for col in RAW_FEATURE_COLS:
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            noise = random.gauss(0, abs(float(val)) * 0.05 + 0.001)
            noisy_features[col] = round(float(val) + noise, 6)

        event = {
            'customer_id': int(row['SK_ID_CURR']),
            'timestamp':   datetime.now().isoformat(),
            'event_type':  random.choice(EVENT_TYPES),
            'features':    noisy_features,         # raw features — consumer applies scaling + WoE
        }

        producer.send(TOPIC, value=event)

        if i % 10 == 0:
            print(f"  Sent {i+1}/{n_events} — "
                  f"Customer {event['customer_id']} — "
                  f"{event['event_type']}")
        time.sleep(delay)

    producer.flush()
    producer.close()
    print(f"\nDone. {n_events} raw events sent to topic: {TOPIC}")


if __name__ == "__main__":
    run_producer()
