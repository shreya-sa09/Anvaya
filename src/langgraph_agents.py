import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROCESSED_PATH = '../data/processed/'
MODEL = 'llama-3.3-70b-versatile'

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


# ── Agent 1 — The Analyser ───────────────────────────
def agent_1_analyser(customer_data: dict) -> str:
    prompt = f"""
You are a credit risk analyst at an Indian bank.
Write a 2-3 sentence plain English summary explaining
why this loan customer is at financial risk.

Customer details:
- Risk Band: {customer_data['risk_band']}
- Probability of Default: {customer_data['final_pd']*100:.0f}%
- Primary stress signal: {customer_data['top_driver_1_label']}
  Detail: {customer_data['top_driver_1_explanation']}
- Secondary signal: {customer_data['top_driver_2_label']}
  Detail: {customer_data['top_driver_2_explanation']}
- Third signal: {customer_data['top_driver_3_label']}
  Detail: {customer_data['top_driver_3_explanation']}

Rules:
- Write in simple English a bank officer can understand
- Do not use technical terms like SHAP, WOE, or Z-score
- Do not mention feature names like F3 or F7
- Focus on what is actually happening with this customer
- Keep it under 3 sentences
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# ── Agent 2 — The Action Agent ───────────────────────
def agent_2_action(case_summary: str, customer_data: dict) -> dict:

    # Calculate real rupee numbers for this customer
    current_emi = customer_data.get('AMT_ANNUITY', 26941)
    credit = customer_data.get('AMT_CREDIT', 594780)
    income = customer_data.get('AMT_INCOME_TOTAL', 165452)

    # New EMI if tenure extended to 48 months
    new_emi = round(credit / 48, 0)
    monthly_saving = round(current_emi - new_emi, 0)

    # Emergency credit limit = 50% of monthly income
    emergency_limit = round(income * 0.5, 0)

    prompt = f"""
You are a relationship manager at an Indian bank.
Based on this customer risk summary, write a warm
empathetic WhatsApp message to send to the customer.

Case Summary: {case_summary}

Recommended intervention: {customer_data['recommended_intervention']}

Customer financial details:
- Current monthly EMI: Rs {current_emi:,.0f}
- Total loan amount: Rs {credit:,.0f}
- Monthly income: Rs {income:,.0f}
- If tenure extended to 48 months, new EMI: Rs {new_emi:,.0f}
- Monthly saving from tenure extension: Rs {monthly_saving:,.0f}
- Emergency credit available: Rs {emergency_limit:,.0f}

Rules:
- Write in simple, warm, empathetic English
- Make the customer feel supported not threatened
- Include the EXACT rupee numbers from above
  that are relevant to the intervention
- Keep it under 4 sentences
- End with: "Reply STOP to opt out of these messages."
- Do not mention default or delinquency
- Sound like a helpful advisor not a collections agent
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4
    )
    return {
        'case_summary': case_summary,
        'recommended_intervention': customer_data[
            'recommended_intervention'
        ],
        'customer_message': response.choices[0].message.content.strip(),
        'risk_band': customer_data['risk_band'],
        'final_pd': customer_data['final_pd']
    }


# ── Full Workflow — Both Agents in Sequence ──────────
def run_anvaya_workflow(customer_data: dict) -> dict:
    print(f"\n  Processing customer {customer_data['SK_ID_CURR']}...")
    print(f"  Risk Band : {customer_data['risk_band']}")
    print(f"  PD Score  : {customer_data['final_pd']*100:.0f}%")

    print("  Agent 1 running — Analyser...")
    case_summary = agent_1_analyser(customer_data)
    print("  Agent 1 done.")

    print("  Agent 2 running — Action Agent...")
    result = agent_2_action(case_summary, customer_data)
    print("  Agent 2 done.")

    return result


# ── Test Single Customer ─────────────────────────────
def test_single_customer():
    print("Loading one RED band customer for testing...")

    shap_df = pd.read_csv(PROCESSED_PATH + 'shap_explanations.csv')
    master = pd.read_csv(PROCESSED_PATH + 'master.csv')
    shap_df = shap_df.merge(
        master[['SK_ID_CURR', 'AMT_ANNUITY',
                'AMT_CREDIT', 'AMT_INCOME_TOTAL']],
        on='SK_ID_CURR',
        how='left'
    )

    red_customers = shap_df[shap_df['risk_band'] == 'RED']
    if len(red_customers) == 0:
        print("No RED customers found — using YELLOW")
        red_customers = shap_df[shap_df['risk_band'] == 'YELLOW']

    customer = red_customers.iloc[0].to_dict()
    result = run_anvaya_workflow(customer)

    print("\n" + "="*60)
    print("AGENT OUTPUT — SINGLE CUSTOMER TEST")
    print("="*60)
    print(f"\nCustomer ID  : {customer['SK_ID_CURR']}")
    print(f"Risk Band    : {result['risk_band']}")
    print(f"PD Score     : {result['final_pd']*100:.0f}%")
    print(f"\nCASE SUMMARY (for RM):")
    print(result['case_summary'])
    print(f"\nRECOMMENDED INTERVENTION:")
    print(result['recommended_intervention'])
    print(f"\nCUSTOMER MESSAGE (ready to send):")
    print(result['customer_message'])
    print("="*60)


# ── Process Multiple Customers ───────────────────────
def run_all_customers(max_customers=10):
    print("="*55)
    print("LANGGRAPH 2-AGENT WORKFLOW")
    print("="*55)

    shap_df = pd.read_csv(PROCESSED_PATH + 'shap_explanations.csv')
    master = pd.read_csv(PROCESSED_PATH + 'master.csv')

    shap_df = shap_df.merge(
        master[['SK_ID_CURR', 'AMT_ANNUITY',
                'AMT_CREDIT', 'AMT_INCOME_TOTAL']],
        on='SK_ID_CURR',
        how='left'
    )
    print(f"Loan data merged. "
          f"Nulls in AMT_ANNUITY: "
          f"{shap_df['AMT_ANNUITY'].isnull().sum()}")

    # Sort RED first then YELLOW
    band_order = {'RED': 0, 'YELLOW': 1}
    shap_df['_sort'] = shap_df['risk_band'].map(band_order)
    shap_df = shap_df.dropna(subset=['_sort'])
    shap_df = shap_df.sort_values('_sort').drop(columns='_sort')

    customers = shap_df.head(max_customers)
    print(f"Processing {len(customers)} customers...")

    results = []
    latencies = []

    for _, row in customers.iterrows():
        customer_data = row.to_dict()
        start = time.time()

        try:
            result = run_anvaya_workflow(customer_data)
            latency = time.time() - start
            latencies.append(latency)

            results.append({
                'SK_ID_CURR': customer_data['SK_ID_CURR'],
                'risk_band': result['risk_band'],
                'final_pd': result['final_pd'],
                'intervention': result['recommended_intervention'],
                'case_summary': result['case_summary'],
                'customer_message': result['customer_message'],
                'latency_seconds': round(latency, 2)
            })

            print(f"  Done in {latency:.1f}s")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_PATH + 'agent_outputs.csv', index=False)

    print()
    print("="*55)
    print("WORKFLOW SUMMARY")
    print(f"  Customers processed : {len(results)}")
    if latencies:
        print(f"  Avg latency         : {sum(latencies)/len(latencies):.1f}s")
        print(f"  Max latency         : {max(latencies):.1f}s")
        print(f"  Target              : under 5 seconds")
        status = 'PASS' if max(latencies) < 5 else 'REVIEW'
        print(f"  Status              : {status}")
    print(f"  Saved               : agent_outputs.csv")
    print("="*55)


if __name__ == "__main__":
    # Step 1 — test one customer first
    test_single_customer()

    # Step 2 — ask before processing all
    answer = input("\nProcess 10 flagged customers? (y/n): ")
    if answer.lower() == 'y':
        run_all_customers(max_customers=10)