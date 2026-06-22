import pandas as pd
import numpy as np

RAW_PATH = "../data/raw/"
PROCESSED_PATH = "../data/processed/"

def load_raw_data():
    print("Loading 4 CSV files...")

    app = pd.read_csv(RAW_PATH + "application_train.csv")
    inst = pd.read_csv(RAW_PATH + "installments_payments.csv")
    bur = pd.read_csv(RAW_PATH + "bureau.csv")
    cc = pd.read_csv(RAW_PATH + "credit_card_balance.csv")

    print(f"application_train:    {app.shape}")
    print(f"installments:         {inst.shape}")
    print(f"bureau:               {bur.shape}")
    print(f"credit_card_balance:  {cc.shape}")

    return app, inst, bur, cc


def filter_customers(app, inst, bur, cc, n=50000):
    print(f"\nFiltering to {n} customers...")

    np.random.seed(42)
    customer_ids = app['SK_ID_CURR'].sample(n, random_state=42).values

    app  = app[app['SK_ID_CURR'].isin(customer_ids)].reset_index(drop=True)
    inst = inst[inst['SK_ID_CURR'].isin(customer_ids)].reset_index(drop=True)
    bur  = bur[bur['SK_ID_CURR'].isin(customer_ids)].reset_index(drop=True)
    cc   = cc[cc['SK_ID_CURR'].isin(customer_ids)].reset_index(drop=True)

    print(f"application_train:    {app.shape}")
    print(f"installments:         {inst.shape}")
    print(f"bureau:               {bur.shape}")
    print(f"credit_card_balance:  {cc.shape}")

    print(f"\nTarget distribution:")
    print(app['TARGET'].value_counts(normalize=True).round(3))

    return app, inst, bur, cc


if __name__ == "__main__":
    app, inst, bur, cc = load_raw_data()
    app, inst, bur, cc = filter_customers(app, inst, bur, cc)
    print("\nData loading complete.")