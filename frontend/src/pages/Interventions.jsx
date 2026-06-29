import { useState, useEffect } from 'react'

const API = 'https://anvaya-z3zm.onrender.com'

export default function Interventions({ onViewCustomer, rmDecisions }) {
    const [data, setData] = useState([])
    const [loading, setLoading] = useState(true)
    const [expanded, setExpanded] = useState(null)

    useEffect(() => {
        fetch(`${API}/interventions`)
            .then(r => r.json())
            .then(d => { setData(d); setLoading(false) })
            .catch(() => setLoading(false))
    }, [])

    const total = data.length
    // P14: Agreed with AI count calculated dynamically from app-level state
    const agreedCount = data.filter(
        d => rmDecisions[d.SK_ID_CURR] === 'Agreed with AI'
    ).length

    // P13: Recovery Rate calculated dynamically based on GREEN (Healthy) risk band cases
    const recoveredCount = data.filter(
        d => d.risk_band === 'GREEN'
    ).length
    const recoveryRate = total > 0 ? Math.round((recoveredCount / total) * 100) : 0

    return (
        <div>
            <h1 style={{
                fontSize: '24px',
                fontWeight: 600,
                color: '#FFFFFF',
                marginBottom: '4px'
            }}>
                Interventions Tracker
            </h1>
            <p style={{
                fontSize: '13px',
                color: '#64748B',
                marginBottom: '24px'
            }}>
                Track and monitor RM action states and customer risk resolution
            </p>

            {/* P13: Metric Cards limited to exactly 3 specified items */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px',
                marginBottom: '24px'
            }}>
                {[
                    {
                        label: 'Total Interventions',
                        value: total,
                        color: '#06B6D4'
                    },
                    {
                        label: 'Agreed with AI',
                        value: agreedCount,
                        color: '#10B981'
                    },
                    {
                        label: 'Recovery Rate',
                        value: `${recoveryRate}%`,
                        color: '#F59E0B'
                    },
                ].map((m, i) => (
                    <div key={i} style={{
                        backgroundColor: '#111827',
                        borderRadius: '12px',
                        padding: '20px',
                        border: '1px solid #1F2937',
                        borderTop: `3px solid ${m.color}`
                    }}>
                        <div style={{
                            fontSize: '12px',
                            color: '#94A3B8',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                        }}>
                            {m.label}
                        </div>
                        <div style={{
                            fontSize: '32px',
                            fontWeight: 700,
                            color: m.color,
                            margin: '8px 0'
                        }}>
                            {m.value}
                        </div>
                    </div>
                ))}
            </div>

            {/* Table */}
            <div style={{
                backgroundColor: '#111827',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid #1F2937'
            }}>
                <div style={{
                    fontSize: '15px',
                    fontWeight: 600,
                    color: '#FFFFFF',
                    marginBottom: '16px'
                }}>
                    Interventions Action Log
                </div>

                {/* Table Header: Customer, AI Recommendation, RM Decision, Outcome, Action */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1.2fr 3.2fr 1.8fr 2fr 1fr',
                    padding: '8px 12px',
                    borderBottom: '1px solid #1F2937',
                    fontSize: '11px',
                    color: '#64748B',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                }}>
                    <span>Customer</span>
                    <span>AI Recommendation</span>
                    <span>RM Decision</span>
                    <span>Outcome</span>
                    <span>Action</span>
                </div>

                {loading && (
                    <div style={{
                        color: '#94A3B8',
                        padding: '20px'
                    }}>
                        Loading interventions data...
                    </div>
                )}

                {data.map(item => {
                    // P13: Determine outcome text and pill colors dynamically
                    let outcomeText = 'Monitoring'
                    let pillColor = '#F59E0B' // Amber
                    let pillBg = 'rgba(245,158,11,0.15)'

                    if (item.customer_message) {
                        outcomeText = 'Intervention Sent'
                        pillColor = '#06B6D4' // Cyan
                        pillBg = 'rgba(6,182,212,0.15)'
                    } else if (item.risk_band === 'GREEN') {
                        outcomeText = 'Recovered'
                        pillColor = '#10B981' // Emerald
                        pillBg = 'rgba(16,185,129,0.15)'
                    } else if (item.risk_band === 'RED') {
                        outcomeText = 'Escalated'
                        pillColor = '#EF4444' // Red
                        pillBg = 'rgba(239,68,68,0.15)'
                    }

                    // P14: RM decision loaded dynamically from lifted App state
                    const decision = rmDecisions[item.SK_ID_CURR] || 'Pending'
                    const decisionColor = decision === 'Pending' ? '#64748B' : decision === 'Agreed with AI' ? '#10B981' : decision === 'Escalated' ? '#F97316' : '#94A3B8'

                    return (
                        <div key={item.SK_ID_CURR}>
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: '1.2fr 3.2fr 1.8fr 2fr 1fr',
                                padding: '12px',
                                borderBottom: '1px solid #1F2937',
                                alignItems: 'center'
                            }}>
                                <span style={{
                                    color: '#FFFFFF',
                                    fontSize: '13px',
                                    fontWeight: 600
                                }}>
                                    USR-{item.SK_ID_CURR}
                                </span>

                                <span style={{
                                    color: '#94A3B8',
                                    fontSize: '13px'
                                }}>
                                    {item.recommended_intervention}
                                </span>

                                {/* P14: RM Decision display */}
                                <span style={{
                                    color: decisionColor,
                                    fontSize: '13px',
                                    fontWeight: 600
                                }}>
                                    {decision}
                                </span>

                                {/* P13: Outcome Pill */}
                                <div>
                                    <span style={{
                                        backgroundColor: pillBg,
                                        color: pillColor,
                                        border: `1px solid ${pillColor}`,
                                        borderRadius: '999px',
                                        padding: '3px 12px',
                                        fontSize: '11px',
                                        fontWeight: 600,
                                        display: 'inline-block'
                                    }}>
                                        {outcomeText}
                                    </span>
                                </div>

                                <button
                                    onClick={() => setExpanded(
                                        expanded === item.SK_ID_CURR
                                            ? null : item.SK_ID_CURR
                                    )}
                                    style={{
                                        backgroundColor: 'transparent',
                                        border: '1px solid #1F2937',
                                        borderRadius: '6px',
                                        padding: '4px 12px',
                                        color: '#94A3B8',
                                        fontSize: '12px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    {expanded === item.SK_ID_CURR
                                        ? '▲ Close' : '▼ Expand'}
                                </button>
                            </div>

                            {/* Expanded Row */}
                            {expanded === item.SK_ID_CURR && (
                                <div style={{
                                    backgroundColor: '#161E2E',
                                    padding: '16px',
                                    borderBottom: '1px solid #1F2937'
                                }}>
                                    <div style={{
                                        fontSize: '13px',
                                        color: '#94A3B8',
                                        lineHeight: 1.6,
                                        marginBottom: '12px'
                                    }}>
                                        <strong style={{ color: '#FFFFFF' }}>
                                            Case Summary:
                                        </strong> {item.case_summary}
                                    </div>
                                    {item.customer_message && (
                                        <div style={{
                                            backgroundColor: '#0B1120',
                                            borderRadius: '8px',
                                            padding: '12px',
                                            fontSize: '13px',
                                            color: '#94A3B8',
                                            borderLeft: '3px solid #10B981',
                                            marginBottom: '12px'
                                        }}>
                                            <strong style={{ color: '#10B981' }}>
                                                Customer Outreach Message:
                                            </strong>{' '}
                                            {item.customer_message}
                                        </div>
                                    )}
                                    <button
                                        onClick={() => onViewCustomer(item.SK_ID_CURR)}
                                        style={{
                                            backgroundColor: 'transparent',
                                            border: '1px solid #06B6D4',
                                            borderRadius: '6px',
                                            padding: '6px 14px',
                                            color: '#06B6D4',
                                            fontSize: '12px',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        View Full Profile
                                    </button>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
