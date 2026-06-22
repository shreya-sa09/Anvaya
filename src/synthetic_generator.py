import csv
import pandas as pd
import numpy as np

SYNTHETIC_PATH = "../data/synthetic/"

def generate_daily_balance(customer_list):
    print("\nGenerating synthetic_daily_balance...")
    output_path = SYNTHETIC_PATH + "synthetic_daily_balance.csv"
    header = ['customer_id', 'date', 'closing_balance', 'is_negative']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for _, row in customer_list.iterrows():
            cid = int(row['SK_ID_CURR'])
            income = row['AMT_INCOME_TOTAL'] / 12
            annuity = row['AMT_ANNUITY'] / 12
            target = int(row.get('TARGET', 0))
            rng = np.random.default_rng(seed=cid % 100000)

            if target == 1:
                drift_rate = rng.normal(loc=-0.003, scale=0.006)
            else:
                drift_rate = rng.normal(loc=0.001, scale=0.006)

            daily_drift = drift_rate * income
            starting_balance = float(rng.normal(loc=income * 1.5, scale=income * 0.5))
            noise_std = income * 0.12

            for day in range(1, 181):
                trend = daily_drift * day
                noise = rng.normal(0, noise_std)
                closing = starting_balance + trend + noise
                writer.writerow([cid, day, round(float(closing), 2), int(closing < 0)])

    print("  Completed synthetic_daily_balance file")
    return None


def generate_income_arrival(customer_list):
    print("\nGenerating synthetic_income_arrival...")

    delay_map = {
        0: (1, 1),
        1: (1, 1),
        2: (2, 1),
        3: (4, 2),
        4: (7, 3),
    }

    rows = []
    for _, row in customer_list.iterrows():
        cid = int(row['SK_ID_CURR'])
        income = row['AMT_INCOME_TOTAL'] / 12
        income_type = int(row['NAME_INCOME_TYPE']) % 5
        # bias base delay by TARGET but add substantial overlap noise
        target = int(row.get('TARGET', 0))
        rng = np.random.default_rng(seed=(cid % 100000) + 1)

        if target == 1:
            lam = 2.5
        else:
            lam = 0.8

        for month in range(1, 7):
            base_delay = rng.poisson(lam=lam)
            delay = int(max(0, round(base_delay + rng.normal(0, 1.5))))

            if target == 1:
                amount = income * float(rng.uniform(0.75, 1.05))
            else:
                amount = income * float(rng.uniform(0.90, 1.05))

            rows.append({
                'customer_id': cid,
                'month_number': month,
                'expected_date': month * 30,
                'actual_date': month * 30 + delay,
                'delay_days': int(delay),
                'amount': round(float(amount), 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv(SYNTHETIC_PATH + "synthetic_income_arrival.csv", index=False)
    print(f"  Shape: {df.shape}")
    return df


def generate_transactions(customer_list):
    print("\nGenerating synthetic_transactions...")

    categories = ['groceries', 'rent_utilities', 'medical', 'entertainment',
                  'atm_cash', 'dining_travel', 'online_shopping']
    output_path = SYNTHETIC_PATH + "synthetic_transactions.csv"
    header = ['customer_id', 'date', 'category', 'amount', 'is_atm_withdrawal']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for _, row in customer_list.iterrows():
            cid = int(row['SK_ID_CURR'])
            income = row['AMT_INCOME_TOTAL'] / 12
            target = int(row.get('TARGET', 0))
            rng = np.random.default_rng(seed=(cid % 100000) + 2)

            if target == 1:
                atm_w = float(rng.uniform(0.12, 0.28))
                groceries_w = float(rng.uniform(0.10, 0.20))
                essentials_w = float(rng.uniform(0.16, 0.28))
            else:
                atm_w = float(rng.uniform(0.06, 0.18))
                groceries_w = float(rng.uniform(0.08, 0.18))
                essentials_w = float(rng.uniform(0.12, 0.22))

            base_weights = [0.18, 0.22, 0.05, 0.16, 0.10, 0.14, 0.15]
            weights = base_weights.copy()
            weights[0] = groceries_w
            weights[1] = essentials_w
            weights[4] = atm_w
            total = sum(weights)
            weights = [w / total for w in weights]

            for month in range(1, 7):
                monthly_transactions = int(rng.integers(18, 31))
                for _ in range(monthly_transactions):
                    cat = rng.choice(categories, p=weights)
                    amount = round(float(income * rng.uniform(0.01, 0.08)), 2)
                    day = int(rng.integers((month - 1) * 30 + 1, month * 30 + 1))
                    writer.writerow([cid, day, cat, amount, int(cat == 'atm_cash')])

    print("  Completed synthetic_transactions file")
    return None


def generate_synthetic_data(master):
    customer_list = master[['SK_ID_CURR', 'AMT_INCOME_TOTAL',
                             'AMT_ANNUITY', 'TARGET',
                             'NAME_INCOME_TYPE']].copy()

    balance_df = generate_daily_balance(customer_list)
    income_df  = generate_income_arrival(customer_list)
    txn_df     = generate_transactions(customer_list)

    print("\nAll synthetic tables saved.")
    return balance_df, income_df, txn_df


if __name__ == "__main__":
    from data_loader import load_raw_data, filter_customers
    from preprocessor import preprocess

    app, inst, bur, cc = load_raw_data()
    app, inst, bur, cc = filter_customers(app, inst, bur, cc)
    master = preprocess(app, inst, bur, cc)
    generate_synthetic_data(master)
    print("\nSynthetic data generation complete.")