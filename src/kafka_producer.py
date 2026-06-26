import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

KAFKA_BROKER = 'localhost:9092'
TOPIC = 'anvaya-transactions'

WOE_COLS = [
    'F3_income_irregularity_WOE',
    'F7_cash_hoarding_WOE',
    'F8_stress_velocity_WOE',
    'F2_savings_drawdown_WOE',
    'F9_payment_timing_entropy_WOE',
    'F11_overdraft_freq_WOE',
    'F10_cohort_stress_WOE',
]

def run_producer(n_events=50, delay=0.2):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    customers = pd.read_csv('../data/processed/features_final.csv')
    print(f"Loaded {len(customers)} customers")
    print(f"Sending {n_events} events...")

    for i in range(n_events):
        row = customers.sample(1).iloc[0]
        event = {
            'customer_id': int(row['SK_ID_CURR']),
            'timestamp': datetime.now().isoformat(),
            'event_type': random.choice([
                'upi_transaction',
                'atm_withdrawal',
                'auto_debit',
                'salary_credit',
                'balance_check'
            ]),
            'features': {
                col: float(row[col])
                for col in WOE_COLS
                if col in row.index
            }
        }
        producer.send(TOPIC, value=event)

        if i % 10 == 0:
            print(f"  Sent {i+1}/{n_events} — "
                  f"Customer {event['customer_id']} — "
                  f"{event['event_type']}")
        time.sleep(delay)

    producer.flush()
    producer.close()
    print(f"\nDone. {n_events} events sent to topic: {TOPIC}")

if __name__ == "__main__":
    run_producer()