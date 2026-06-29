import json
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime


KAFKA_BROKER = 'localhost:9092'
ALERT_TOPIC = 'anvaya-alerts'
CONTAGION_THRESHOLD = 5


ORG_TYPE_MAP = {
    0: 'Business Entity Type 3',
    1: 'School',
    2: 'Government',
    3: 'Religion',
    4: 'Other',
    5: 'Medicine',
    6: 'Business Entity Type 2',
    7: 'Self-employed',
    8: 'Transport Type 2',
    9: 'Construction',
    10: 'Housing',
    11: 'Kindergarten',
    12: 'Trade Type 6',
    13: 'Industry Type 11',
    14: 'Military',
    15: 'Services',
    16: 'Security Ministries',
    17: 'Transport Type 4',
    18: 'Trade Type 3',
    19: 'University',
    20: 'Transport Type 3',
    21: 'Police',
    22: 'Business Entity Type 1',
    23: 'Postal',
    24: 'Industry Type 4',
    25: 'Agriculture',
    26: 'Restaurant',
    27: 'Culture',
    28: 'Hotel',
    29: 'Electricity',
    30: 'Security',
    31: 'Realtor',
    32: 'Telecom',
    33: 'Industry Type 1',
    34: 'Emergency',
    35: 'Bank',
    36: 'Industry Type 9',
    37: 'Insurance',
    38: 'Trade Type 2',
    39: 'Trade Type 7',
    40: 'Industry Type 2',
    41: 'Trade Type 1',
    42: 'Industry Type 12',
    43: 'Industry Type 5',
    44: 'Industry Type 10',
    45: 'Legal Services',
    46: 'Advertising',
    47: 'Trade Type 5',
    48: 'Cleaning',
    49: 'Industry Type 13',
    50: 'Trade Type 4',
    51: 'Industry Type 3',
    52: 'Industry Type 7',
    53: 'Industry Type 8',
    54: 'Industry Type 6',
    55: 'XNA',
    56: 'Mobile',
    57: 'Industry Type 11',
}


def load_data():
    # Use combined_predictions so all 49,000 customers are covered, not just holdout
    predictions = pd.read_csv('../data/processed/combined_predictions.csv')
    master = pd.read_csv('../data/processed/master.csv')


    keep = ['SK_ID_CURR']
    if 'ORGANIZATION_TYPE' in master.columns:
        keep.append('ORGANIZATION_TYPE')


    merged = predictions.merge(
        master[keep], on='SK_ID_CURR', how='left'
    )


    if 'ORGANIZATION_TYPE' not in merged.columns:
        merged['ORGANIZATION_TYPE'] = 'Unknown'


    # Map encoded integers to readable names
    merged['ORGANIZATION_TYPE'] = (
        merged['ORGANIZATION_TYPE']
        .map(ORG_TYPE_MAP)
        .fillna('Unknown Sector')
    )


    return merged


def detect_contagion(df):
    stressed = df[df['risk_band'].isin(['YELLOW', 'RED'])].copy()


    alerts = []
    for employer, group in stressed.groupby('ORGANIZATION_TYPE'):
        if len(group) >= CONTAGION_THRESHOLD:
            total_at_employer = len(
                df[df['ORGANIZATION_TYPE'] == employer]
            )
            alert = {
                'alert_type': 'STRESS_CONTAGION',
                'employer': str(employer),
                'stressed_count': int(len(group)),
                'total_at_employer': int(total_at_employer),
                'stress_rate_pct': round(
                    len(group) / total_at_employer * 100, 1
                ),
                'red_count': int(
                    (group['risk_band'] == 'RED').sum()
                ),
                'yellow_count': int(
                    (group['risk_band'] == 'YELLOW').sum()
                ),
                'avg_pd': round(
                    float(group['final_pd'].mean()), 4
                ),
                'timestamp': datetime.now().isoformat(),
                'recommended_action': (
                    f"{len(group)} customers at {employer} "
                    f"showing stress signals. "
                    f"Consider sector-wide relief program."
                )
            }
            alerts.append(alert)


    alerts.sort(key=lambda x: x['stressed_count'], reverse=True)
    return alerts


def run_contagion_detector():
    print("="*55)
    print("STRESS CONTAGION DETECTOR")
    print("="*55)


    df = load_data()
    total = len(df)
    stressed_total = len(df[df['risk_band'].isin(['YELLOW', 'RED'])])


    print(f"Total customers    : {total}")
    print(f"Stressed customers : {stressed_total}")
    print(f"Contagion threshold: {CONTAGION_THRESHOLD} per employer")
    print()


    alerts = detect_contagion(df)


    if not alerts:
        print("No contagion detected")
        return


    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        kafka_available = True
    except Exception:
        kafka_available = False
        print("Kafka not running — printing alerts only")


    print(f"CONTAGION ALERTS FIRED: {len(alerts)}")
    print()


    for i, alert in enumerate(alerts[:5], 1):
        print(f"Alert {i}:")
        print(f"  Employer       : {alert['employer']}")
        print(f"  Stressed       : {alert['stressed_count']} "
              f"({alert['stress_rate_pct']}% of employer)")
        print(f"  RED band       : {alert['red_count']}")
        print(f"  YELLOW band    : {alert['yellow_count']}")
        print(f"  Avg PD         : {alert['avg_pd']:.4f}")
        print(f"  Action         : {alert['recommended_action']}")
        print()


        if kafka_available:
            producer.send(ALERT_TOPIC, value=alert)


    if kafka_available:
        producer.flush()
        producer.close()
        print(f"Alerts sent to Kafka topic: {ALERT_TOPIC}")


    print("="*55)


if __name__ == "__main__":
    run_contagion_detector()
