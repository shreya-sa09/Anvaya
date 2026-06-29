import { useState } from 'react'

const API = 'https://anvaya-z3zm.onrender.com'

export default function LiveScoring({ onViewCustomer }) {
    const [form, setForm] = useState({
        income: '',
        emi: '',
        savings_drawdown: 50, // P17: default slider midpoint value
        failed_autodebits: '',
        lending_app: 0,
        days_since_income: ''
    })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [copied, setCopied] = useState(false) // P8: copy button active state
    const [pipeline, setPipeline] = useState([])

    const HEALTHY = {
        income: 55000, emi: 12000,
        savings_drawdown: 8, failed_autodebits: 0,
        lending_app: 0, days_since_income: 2
    }

    const STRESSED = {
        income: 22000, emi: 14000,
        savings_drawdown: 72, failed_autodebits: 3,
        lending_app: 1, days_since_income: 18
    }

    function loadPreset(preset) {
        setForm(preset)
        setResult(null)
        setPipeline([])
    }

    async function runScoring() {
        setLoading(true)
        setResult(null)
        setPipeline([])

        const steps = [
            'Kafka event received',
            'Rhythm Engine normalizing features',
            'WOE transformation complete',
            'XGBoost screening',
            'LightGBM deep analysis',
            'Meta model blending scores',
            'Calibration complete',
            'SHAP analysis done',
            'Risk band assigned',
        ]

        for (let i = 0; i < steps.length; i++) {
            await new Promise(r => setTimeout(r, 200))
            setPipeline(prev => [...prev, {
                text: steps[i],
                time: new Date().toLocaleTimeString()
            }])
        }

        const params = new URLSearchParams({
            income: form.income,
            emi: form.emi,
            savings_drawdown: form.savings_drawdown,
            failed_autodebits: form.failed_autodebits,
            lending_app: form.lending_app,
            days_since_income: form.days_since_income
        })

        const res = await fetch(`${API}/score?${params}`)
        const data = await res.json()
        setResult(data)
        setLoading(false)
    }

    const bandColor = band => {
        if (band === 'RED') return '#EF4444'
        if (band === 'HIGH') return '#F97316'
        if (band === 'YELLOW') return '#F59E0B'
        return '#10B981'
    }

    return (
        <div>
            <h1 style={{
                fontSize: '24px',
                fontWeight: 600,
                color: '#FFFFFF',
                marginBottom: '4px'
            }}>
                Live Scoring
            </h1>
            <p style={{
                fontSize: '13px',
                color: '#64748B',
                marginBottom: '24px'
            }}>
                Real-time risk assessment pipeline demo
            </p>

            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px'
            }}>
                {/* Left — Input Form */}
                <div style={{
                    backgroundColor: '#111827',
                    borderRadius: '12px',
                    padding: '20px',
                    border: '1px solid #1F2937'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        marginBottom: '20px'
                    }}>
                        <div style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: '#10B981'
                        }} />
                        <span style={{
                            fontSize: '15px',
                            fontWeight: 600,
                            color: '#FFFFFF'
                        }}>
                            Live Risk Assessment
                        </span>
                    </div>

                    {/* Preset Buttons */}
                    <div style={{
                        display: 'flex',
                        gap: '8px',
                        marginBottom: '20px'
                    }}>
                        <button
                            onClick={() => loadPreset(HEALTHY)}
                            style={{
                                flex: 1,
                                backgroundColor: 'transparent',
                                border: '1px solid #10B981',
                                borderRadius: '6px',
                                padding: '8px',
                                color: '#10B981',
                                fontSize: '12px',
                                fontWeight: 600,
                                cursor: 'pointer'
                            }}
                        >
                            Load Healthy Customer
                        </button>
                        <button
                            onClick={() => loadPreset(STRESSED)}
                            style={{
                                flex: 1,
                                backgroundColor: 'transparent',
                                border: '1px solid #EF4444',
                                borderRadius: '6px',
                                padding: '8px',
                                color: '#EF4444',
                                fontSize: '12px',
                                fontWeight: 600,
                                cursor: 'pointer'
                            }}
                        >
                            Load Stressed Customer
                        </button>
                    </div>

                    {/* Input Form Fields (Excluding savings_drawdown range slider) */}
                    {[
                        { key: 'income', label: 'Monthly Income (₹)' },
                        { key: 'emi', label: 'Monthly EMI (₹)' },
                        {
                            key: 'failed_autodebits',
                            label: 'Failed Auto-Debits'
                        },
                        {
                            key: 'days_since_income',
                            label: 'Days Since Last Income'
                        },
                    ].map(field => (
                        <div key={field.key} style={{
                            marginBottom: '12px'
                        }}>
                            <div style={{
                                fontSize: '12px',
                                color: '#64748B',
                                marginBottom: '4px'
                            }}>
                                {field.label}
                            </div>
                            <input
                                type="number"
                                value={form[field.key]}
                                onChange={e => setForm({
                                    ...form,
                                    [field.key]: e.target.value
                                })}
                                style={{
                                    width: '100%',
                                    backgroundColor: '#161E2E',
                                    border: '1px solid #1F2937',
                                    borderRadius: '6px',
                                    padding: '8px 12px',
                                    color: '#FFFFFF',
                                    fontSize: '13px',
                                    outline: 'none'
                                }}
                            />
                        </div>
                    ))}

                    {/* P17: Savings Drawdown Range Slider */}
                    <div style={{ marginBottom: '12px' }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '12px',
                            color: '#64748B',
                            marginBottom: '4px'
                        }}>
                            <span>Savings Drawdown %</span>
                            <span style={{ color: '#06B6D4', fontWeight: 600 }}>
                                {form.savings_drawdown}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            step="1"
                            value={form.savings_drawdown}
                            onChange={e => setForm({
                                ...form,
                                savings_drawdown: parseInt(e.target.value)
                            })}
                            style={{
                                width: '100%',
                                accentColor: '#06B6D4',
                                cursor: 'pointer'
                            }}
                        />
                    </div>

                    {/* Lending App Toggle */}
                    <div style={{ marginBottom: '20px' }}>
                        <div style={{
                            fontSize: '12px',
                            color: '#64748B',
                            marginBottom: '4px'
                        }}>
                            Lending App Usage
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            {[
                                { label: 'No', value: 0 },
                                { label: 'Yes', value: 1 }
                            ].map(opt => (
                                <button
                                    key={opt.value}
                                    onClick={() => setForm({
                                        ...form,
                                        lending_app: opt.value
                                    })}
                                    style={{
                                        flex: 1,
                                        backgroundColor:
                                            form.lending_app === opt.value
                                                ? '#06B6D4' : '#161E2E',
                                        border: '1px solid #1F2937',
                                        borderRadius: '6px',
                                        padding: '8px',
                                        color: form.lending_app === opt.value
                                            ? '#000000' : '#94A3B8',
                                        fontSize: '13px',
                                        fontWeight: 600,
                                        cursor: 'pointer'
                                    }}
                                >
                                    {opt.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <button
                        onClick={runScoring}
                        disabled={loading}
                        style={{
                            width: '100%',
                            backgroundColor: '#06B6D4',
                            border: 'none',
                            borderRadius: '8px',
                            padding: '12px',
                            color: '#000000',
                            fontSize: '14px',
                            fontWeight: 700,
                            cursor: loading ? 'not-allowed' : 'pointer',
                            opacity: loading ? 0.7 : 1
                        }}
                    >
                        {loading ? 'Assessing...' : 'Assess Risk Now'}
                    </button>
                </div>

                {/* Right — Pipeline Feed + Result */}
                <div style={{
                    backgroundColor: '#111827',
                    borderRadius: '12px',
                    padding: '20px',
                    border: '1px solid #1F2937'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        marginBottom: '20px'
                    }}>
                        <div style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: pipeline.length > 0
                                ? '#06B6D4' : '#64748B'
                        }} />
                        <span style={{
                            fontSize: '15px',
                            fontWeight: 600,
                            color: '#FFFFFF'
                        }}>
                            Pipeline Feed
                        </span>
                    </div>

                    {pipeline.length === 0 && !result && (
                        <div style={{
                            color: '#64748B',
                            fontSize: '13px',
                            textAlign: 'center',
                            padding: '40px 0'
                        }}>
                            Submit an assessment to see the pipeline in action
                        </div>
                    )}

                    {pipeline.map((step, i) => (
                        <div key={i} style={{
                            backgroundColor: '#161E2E',
                            borderRadius: '6px',
                            padding: '8px 12px',
                            marginBottom: '6px',
                            borderLeft: '3px solid #10B981',
                            fontSize: '12px',
                            color: '#94A3B8',
                            display: 'flex',
                            justifyContent: 'space-between'
                        }}>
                            <span>✓ {step.text}</span>
                            <span style={{ color: '#64748B' }}>
                                {step.time}
                            </span>
                        </div>
                    ))}

                    {/* P8: Complete result panel with driver, measure, outreach message and copy button */}
                    {result && (
                        <div style={{ marginTop: '20px' }}>
                            <div style={{
                                textAlign: 'center',
                                marginBottom: '16px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '16px',
                                backgroundColor: '#161E2E',
                                borderRadius: '10px',
                                padding: '16px',
                                border: `1px solid ${bandColor(result.risk_band)}`
                            }}>
                                <div style={{
                                    width: '70px',
                                    height: '70px',
                                    borderRadius: '50%',
                                    border: `5px solid ${bandColor(result.risk_band)}`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    backgroundColor: `${bandColor(result.risk_band)}10`
                                }}>
                                    <span style={{
                                        fontSize: '16px',
                                        fontWeight: 700,
                                        color: bandColor(result.risk_band)
                                    }}>
                                        {result.pd_score}%
                                    </span>
                                </div>
                                <div style={{ textAlign: 'left' }}>
                                    <div style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase', fontWeight: 600 }}>
                                        Assessed PD Range
                                    </div>
                                    <div style={{
                                        fontSize: '18px',
                                        fontWeight: 700,
                                        color: bandColor(result.risk_band)
                                    }}>
                                        {result.risk_band} RISK
                                    </div>
                                </div>
                            </div>

                            {/* Top Driver Card with deviation explanation */}
                            <div style={{
                                backgroundColor: '#161E2E',
                                borderRadius: '8px',
                                padding: '14px',
                                borderLeft: `3px solid ${bandColor(result.risk_band)}`,
                                marginBottom: '12px'
                            }}>
                                <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600, marginBottom: '4px' }}>
                                    TOP DELINQUENCY DRIVER
                                </div>
                                <div style={{ fontSize: '13px', color: '#FFFFFF', fontWeight: 600, marginBottom: '4px' }}>
                                    {result.top_driver.label}
                                </div>
                                <div style={{ fontSize: '12px', color: '#94A3B8', lineHeight: 1.4 }}>
                                    {result.top_driver.explanation}
                                </div>
                            </div>

                            {/* Suggested Measure Card with intervention */}
                            <div style={{
                                backgroundColor: '#161E2E',
                                borderRadius: '8px',
                                padding: '14px',
                                borderLeft: '3px solid #F59E0B',
                                marginBottom: '12px'
                            }}>
                                <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600, marginBottom: '4px' }}>
                                    SUGGESTED ACTION MEASURE
                                </div>
                                <div style={{ fontSize: '13px', color: '#FFFFFF', fontWeight: 600 }}>
                                    {result.suggested_measure}
                                </div>
                            </div>

                            {/* Drafted Customer Message with Copy button */}
                            <div style={{
                                backgroundColor: '#161E2E',
                                borderRadius: '8px',
                                padding: '14px',
                                borderLeft: '3px solid #10B981',
                                marginBottom: '16px'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                    <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>
                                        DRAFTED OUTREACH MESSAGE
                                    </span>
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(result.customer_message)
                                            setCopied(true)
                                            setTimeout(() => setCopied(false), 2000)
                                        }}
                                        style={{
                                            backgroundColor: copied ? '#10B981' : 'transparent',
                                            border: '1px solid #10B981',
                                            borderRadius: '4px',
                                            padding: '2px 8px',
                                            color: copied ? '#000000' : '#10B981',
                                            fontSize: '10px',
                                            fontWeight: 600,
                                            cursor: 'pointer',
                                            transition: 'all 0.2s'
                                        }}
                                    >
                                        {copied ? '✓ Copied' : 'Copy Message'}
                                    </button>
                                </div>
                                <div style={{ fontSize: '12px', color: '#94A3B8', lineHeight: 1.4, fontStyle: 'italic' }}>
                                    "{result.customer_message}"
                                </div>
                            </div>

                            {/* View Full Profile link (navigates to reference patient profile in Portfolio) */}
                            <div style={{ textAlign: 'center', marginTop: '10px' }}>
                                <button
                                    onClick={() => onViewCustomer(230821)} // loads high-stress benchmark profile USR-230821
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        color: '#06B6D4',
                                        textDecoration: 'underline',
                                        fontSize: '13px',
                                        cursor: 'pointer',
                                        fontWeight: 600
                                    }}
                                >
                                    View Reference Stressed Profile (USR-230821)
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
