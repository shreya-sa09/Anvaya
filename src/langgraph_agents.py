"""
ANVAYA LangGraph 2-Agent Workflow
──────────────────────────────────
Uses a real LangGraph StateGraph with two nodes:
  • analyser   — reads SHAP signals, writes a plain-English RM summary
  • action     — reads the summary, drafts a WhatsApp message

State flows: START → analyser → action → END
"""
import os
import time
import pandas as pd
from typing import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

PROCESSED_PATH = '../data/processed/'
MODEL          = 'llama-3.3-70b-versatile'

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


# ── State schema ─────────────────────────────────────────────────────────────
class AnvayaState(TypedDict):
    customer_data: dict          # input: SHAP row + master fields
    case_summary:  str           # output of Agent 1
    customer_message: str        # output of Agent 2
    recommended_intervention: str
    risk_band: str
    final_pd: float


# ── Agent 1 Node — Risk Analyser ─────────────────────────────────────────────
def agent_analyser(state: AnvayaState) -> AnvayaState:
    cd = state['customer_data']
    prompt = f"""
You are a credit risk analyst at an Indian bank.
Write a 2-3 sentence plain English summary explaining
why this loan customer is at financial risk.

Customer details:
- Risk Band: {cd['risk_band']}
- Probability of Default: {cd['final_pd']*100:.0f}%
- Primary stress signal: {cd['top_driver_1_label']}
  Detail: {cd['top_driver_1_explanation']}
- Secondary signal: {cd['top_driver_2_label']}
  Detail: {cd['top_driver_2_explanation']}
- Third signal: {cd['top_driver_3_label']}
  Detail: {cd['top_driver_3_explanation']}

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
    state['case_summary'] = response.choices[0].message.content.strip()
    return state


# ── Agent 2 Node — Action Agent ───────────────────────────────────────────────
def agent_action(state: AnvayaState) -> AnvayaState:
    cd           = state['customer_data']
    case_summary = state['case_summary']

    current_emi   = cd.get('AMT_ANNUITY', 26941)
    credit        = cd.get('AMT_CREDIT', 594780)
    income        = cd.get('AMT_INCOME_TOTAL', 165452)
    new_emi       = round(credit / 48, 0)
    monthly_saving = round(current_emi - new_emi, 0)
    emergency_limit = round(income * 0.5, 0)
    intervention  = cd.get('recommended_intervention', 'Review account for tailored intervention')

    prompt = f"""
You are a relationship manager at an Indian bank.
Based on this customer risk summary, write a warm
empathetic WhatsApp message to send to the customer.

Case Summary: {case_summary}

Recommended intervention: {intervention}

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
- Include the EXACT rupee numbers relevant to the intervention
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
    state['customer_message']         = response.choices[0].message.content.strip()
    state['recommended_intervention'] = intervention
    state['risk_band']                = cd['risk_band']
    state['final_pd']                 = cd['final_pd']
    return state


# ── Build LangGraph ────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AnvayaState)
    g.add_node("analyser", agent_analyser)
    g.add_node("action",   agent_action)
    g.add_edge(START,       "analyser")
    g.add_edge("analyser",  "action")
    g.add_edge("action",    END)
    return g.compile()

GRAPH = build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────
def run_anvaya_workflow(customer_data: dict) -> dict:
    print(f"\n  Processing customer {customer_data['SK_ID_CURR']}...")
    print(f"  Risk Band : {customer_data['risk_band']}")
    print(f"  PD Score  : {customer_data['final_pd']*100:.0f}%")

    initial_state: AnvayaState = {
        "customer_data":          customer_data,
        "case_summary":           "",
        "customer_message":       "",
        "recommended_intervention": "",
        "risk_band":              customer_data['risk_band'],
        "final_pd":               customer_data['final_pd'],
    }

    final_state = GRAPH.invoke(initial_state)

    return {
        'case_summary':             final_state['case_summary'],
        'recommended_intervention': final_state['recommended_intervention'],
        'customer_message':         final_state['customer_message'],
        'risk_band':                final_state['risk_band'],
        'final_pd':                 final_state['final_pd'],
    }


# ── Test Single Customer ───────────────────────────────────────────────────────
def test_single_customer():
    print("Loading one RED band customer for testing...")

    shap_df = pd.read_csv(PROCESSED_PATH + 'shap_explanations.csv')
    master  = pd.read_csv(PROCESSED_PATH + 'master.csv')
    shap_df = shap_df.merge(
        master[['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']],
        on='SK_ID_CURR', how='left'
    )

    red = shap_df[shap_df['risk_band'] == 'RED']
    if red.empty:
        red = shap_df[shap_df['risk_band'] == 'YELLOW']

    customer = red.iloc[0].to_dict()
    result   = run_anvaya_workflow(customer)

    print("\n" + "=" * 60)
    print("AGENT OUTPUT — SINGLE CUSTOMER TEST")
    print("=" * 60)
    print(f"\nCustomer ID  : {customer['SK_ID_CURR']}")
    print(f"Risk Band    : {result['risk_band']}")
    print(f"PD Score     : {result['final_pd']*100:.0f}%")
    print(f"\nCASE SUMMARY (for RM):\n{result['case_summary']}")
    print(f"\nRECOMMENDED INTERVENTION:\n{result['recommended_intervention']}")
    print(f"\nCUSTOMER MESSAGE (ready to send):\n{result['customer_message']}")
    print("=" * 60)


# ── Process Multiple Customers ─────────────────────────────────────────────────
def run_all_customers(max_customers=10):
    print("=" * 55)
    print("ANVAYA LANGGRAPH 2-AGENT WORKFLOW")
    print("=" * 55)

    shap_df = pd.read_csv(PROCESSED_PATH + 'shap_explanations.csv')
    master  = pd.read_csv(PROCESSED_PATH + 'master.csv')
    shap_df = shap_df.merge(
        master[['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']],
        on='SK_ID_CURR', how='left'
    )

    band_order = {'RED': 0, 'HIGH': 1, 'YELLOW': 2}
    shap_df['_sort'] = shap_df['risk_band'].map(band_order)
    shap_df = shap_df.dropna(subset=['_sort']).sort_values('_sort').drop(columns='_sort')

    customers = shap_df.head(max_customers)
    print(f"Processing {len(customers)} customers...")

    results   = []
    latencies = []

    for _, row in customers.iterrows():
        customer_data = row.to_dict()
        start = time.time()
        try:
            result    = run_anvaya_workflow(customer_data)
            latency   = time.time() - start
            latencies.append(latency)
            results.append({
                'SK_ID_CURR':   customer_data['SK_ID_CURR'],
                'risk_band':    result['risk_band'],
                'final_pd':     result['final_pd'],
                'intervention': result['recommended_intervention'],
                'case_summary': result['case_summary'],
                'customer_message': result['customer_message'],
                'latency_seconds': round(latency, 2),
            })
            print(f"  Done in {latency:.1f}s")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    pd.DataFrame(results).to_csv(PROCESSED_PATH + 'agent_outputs.csv', index=False)

    print()
    print("=" * 55)
    print("WORKFLOW SUMMARY")
    print(f"  Customers processed : {len(results)}")
    if latencies:
        print(f"  Avg latency         : {sum(latencies)/len(latencies):.1f}s")
        print(f"  Max latency         : {max(latencies):.1f}s")
        status = 'PASS' if max(latencies) < 10 else 'REVIEW'
        print(f"  Status              : {status}")
    print(f"  Saved               : agent_outputs.csv")
    print("=" * 55)


if __name__ == "__main__":
    test_single_customer()
    answer = input("\nProcess 10 flagged customers? (y/n): ")
    if answer.lower() == 'y':
        run_all_customers(max_customers=10)
