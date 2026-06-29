import { useState, useEffect } from 'react'

const API = 'https://anvaya-z3zm.onrender.com'

export default function Alerts({ onViewCustomer, dismissedAlerts, onDismissAlert, readAlerts, onMarkRead }) {
    const [alerts, setAlerts] = useState([])
    const [filter, setFilter] = useState('All') // Maps to: 'All', 'Unread', 'Critical', 'High'
    const [timeRange, setTimeRange] = useState('24h')
    const [loading, setLoading] = useState(true)
    const [contagion, setContagion] = useState(null)

    useEffect(() => {
        // Fetch Alerts
        fetch(`${API}/alerts`)
            .then(r => r.json())
            .then(d => { setAlerts(d); setLoading(false) })
            .catch(() => setLoading(false))

        // Fetch Contagion Banner Info (P19)
        fetch(`${API}/contagion`)
            .then(r => r.json())
            .then(setContagion)
            .catch(() => { })
    }, [])

    // Filter out dismissed alerts locally (P12)
    const activeAlerts = alerts.filter(a => !dismissedAlerts.includes(a.SK_ID_CURR))

    // Apply Filter Tab selections (P11)
    const filtered = activeAlerts.filter(a => {
        if (filter === 'All') return true
        if (filter === 'Unread') return !readAlerts.includes(a.SK_ID_CURR)
        if (filter === 'Critical') return a.risk_band === 'RED'
        if (filter === 'High') return a.risk_band === 'HIGH'
        return true
    })

    const bandColor = band => {
        if (band === 'RED') return '#EF4444'
        if (band === 'HIGH') return '#F97316'
        return '#F59E0B'
    }

    return (
        <div>
            <h1 style={{
                fontSize: '24px',
                fontWeight: 600,
                color: '#FFFFFF',
                marginBottom: '4px'
            }}>
                Alerts Feed
            </h1>
            <p style={{
                fontSize: '13px',
                color: '#64748B',
                marginBottom: '24px'
            }}>
                Real-time warning signals requiring immediate client outreach and triage
            </p>

            {/* P19: Conditional Stress Contagion Banner */}
            {contagion && contagion.active && (
                <div style={{
                    backgroundColor: 'rgba(239,68,68,0.1)',
                    border: '1px solid #EF4444',
                    borderRadius: '10px',
                    padding: '16px',
                    marginBottom: '24px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                }}>
                    <div>
                        <div style={{
                            fontSize: '14px',
                            fontWeight: 600,
                            color: '#EF4444',
                            marginBottom: '4px'
                        }}>
                            🔥 Stress Contagion Detector Has Fired
                        </div>
                        <div style={{
                            fontSize: '13px',
                            color: '#94A3B8'
                        }}>
                            Employer level stress detected at <strong>{contagion.employer_name}</strong>.
                            {' '}{contagion.affected_count} Stressed Employees. Total Credit Exposure: ₹{Math.round(contagion.total_exposure).toLocaleString()}
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                            onClick={() => {
                                // Simulate filtering alerts to this group or alert notification
                                alert(`Displaying contagion details for ${contagion.employer_name}`);
                            }}
                            style={{
                                backgroundColor: '#EF4444',
                                border: 'none',
                                borderRadius: '6px',
                                padding: '8px 16px',
                                color: '#FFFFFF',
                                fontSize: '12px',
                                fontWeight: 600,
                                cursor: 'pointer'
                            }}
                        >
                            View Affected
                        </button>
                        <button
                            onClick={() => {
                                alert(`Employer group ${contagion.employer_name} escalated to senior credit committee.`);
                            }}
                            style={{
                                backgroundColor: 'transparent',
                                border: '1px solid #EF4444',
                                borderRadius: '6px',
                                padding: '8px 16px',
                                color: '#EF4444',
                                fontSize: '12px',
                                fontWeight: 600,
                                cursor: 'pointer'
                            }}
                        >
                            Escalate Group
                        </button>
                    </div>
                </div>
            )}

            {/* Filter Tabs & P18 Time Range Selector */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
                gap: '12px'
            }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                    {['All', 'Unread', 'Critical', 'High'].map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            style={{
                                padding: '6px 16px',
                                borderRadius: '6px',
                                border: '1px solid #1F2937',
                                backgroundColor: filter === f
                                    ? '#06B6D4' : '#161E2E',
                                color: filter === f ? '#000000' : '#94A3B8',
                                fontSize: '12px',
                                fontWeight: 600,
                                cursor: 'pointer'
                            }}
                        >
                            {f}
                        </button>
                    ))}
                </div>

                {/* P18: Time Range Dropdown Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '12px', color: '#64748B' }}>Time Range:</span>
                    <select
                        value={timeRange}
                        onChange={e => setTimeRange(e.target.value)}
                        style={{
                            backgroundColor: '#161E2E',
                            border: '1px solid #1F2937',
                            borderRadius: '6px',
                            padding: '6px 12px',
                            color: '#FFFFFF',
                            fontSize: '12px',
                            outline: 'none',
                            cursor: 'pointer'
                        }}
                    >
                        <option value="24h">Last 24 Hours</option>
                        <option value="48h">Last 48 Hours</option>
                        <option value="7d">Last 7 Days (Show All)</option>
                    </select>
                </div>
            </div>

            {/* Alert Cards */}
            {loading && (
                <div style={{ color: '#94A3B8' }}>Loading alerts...</div>
            )}

            {!loading && filtered.length === 0 && (
                <div style={{
                    color: '#64748B',
                    textAlign: 'center',
                    padding: '40px',
                    backgroundColor: '#111827',
                    borderRadius: '10px',
                    border: '1px solid #1F2937'
                }}>
                    No warnings found under this filter selection.
                </div>
            )}

            {filtered.map(alert => {
                const isUnread = !readAlerts.includes(alert.SK_ID_CURR)
                const color = bandColor(alert.risk_band)
                // P11: Unread alerts styled with a thicker border and dynamic text weights
                const borderStyle = isUnread ? `6px solid ${color}` : `3px solid ${color}`
                const cardBg = isUnread ? '#1C2538' : '#111827'

                // P11: dynamic time elapsed and RM fields derived from customer ID
                const timeElapsed = (alert.SK_ID_CURR % 12 + 1) + "h ago"
                const assignedRm = ['RM A. Sharma', 'RM R. Iyer', 'RM S. Patel', 'RM M. Sen'][alert.SK_ID_CURR % 4]

                return (
                    <div
                        key={alert.SK_ID_CURR}
                        style={{
                            backgroundColor: cardBg,
                            borderRadius: '10px',
                            padding: '16px',
                            border: '1px solid #1F2937',
                            borderLeft: borderStyle,
                            marginBottom: '12px',
                            transition: 'all 0.2s'
                        }}
                    >
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start'
                        }}>
                            <div style={{ flex: 1 }}>
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    marginBottom: '6px'
                                }}>
                                    {/* P11: Visual severity dot */}
                                    <span style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        backgroundColor: color,
                                        display: 'inline-block'
                                    }} />

                                    <span style={{
                                        backgroundColor: `${color}20`,
                                        color: color,
                                        border: `1px solid ${color}`,
                                        borderRadius: '999px',
                                        padding: '1px 8px',
                                        fontSize: '10px',
                                        fontWeight: 600
                                    }}>
                                        {alert.risk_band === 'RED' ? 'CRITICAL' : 'HIGH'}
                                    </span>

                                    {isUnread && (
                                        <span style={{
                                            backgroundColor: '#06B6D4',
                                            color: '#000000',
                                            fontSize: '9px',
                                            fontWeight: 800,
                                            padding: '1px 6px',
                                            borderRadius: '3px',
                                            letterSpacing: '0.05em'
                                        }}>
                                            UNREAD
                                        </span>
                                    )}

                                    {/* Customer Name and ID rendered cleanly */}
                                    <span style={{
                                        color: '#FFFFFF',
                                        fontSize: '14px',
                                        fontWeight: isUnread ? 700 : 500
                                    }}>
                                        Cust. {alert.SK_ID_CURR} (USR-{alert.SK_ID_CURR})
                                    </span>
                                </div>

                                <div style={{
                                    fontSize: '13px',
                                    color: isUnread ? '#E2E8F0' : '#94A3B8',
                                    marginBottom: '6px',
                                    lineHeight: 1.5
                                }}>
                                    <strong>Trigger reason:</strong> {alert.top_driver_1_label} — {alert.top_driver_1_explanation}
                                </div>

                                <div style={{
                                    fontSize: '12px',
                                    color: '#64748B',
                                    display: 'flex',
                                    gap: '12px'
                                }}>
                                    <span>PD Score: <strong>{(alert.final_pd * 100).toFixed(1)}%</strong></span>
                                    <span>•</span>
                                    <span>RM: <strong>{assignedRm}</strong></span>
                                    <span>•</span>
                                    <span>Time: <strong>{timeElapsed}</strong></span>
                                </div>
                            </div>

                            <div style={{
                                display: 'flex',
                                gap: '8px',
                                marginLeft: '16px'
                            }}>
                                <button
                                    onClick={() => {
                                        onMarkRead(alert.SK_ID_CURR)
                                        onViewCustomer(alert.SK_ID_CURR)
                                    }}
                                    style={{
                                        backgroundColor: 'transparent',
                                        border: '1px solid #06B6D4',
                                        borderRadius: '6px',
                                        padding: '6px 14px',
                                        color: '#06B6D4',
                                        fontSize: '12px',
                                        fontWeight: 600,
                                        cursor: 'pointer'
                                    }}
                                >
                                    View Profile
                                </button>
                                <button
                                    onClick={() => onDismissAlert(alert.SK_ID_CURR)}
                                    style={{
                                        backgroundColor: 'transparent',
                                        border: '1px solid #374151',
                                        borderRadius: '6px',
                                        padding: '6px 14px',
                                        color: '#94A3B8',
                                        fontSize: '12px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Dismiss
                                </button>
                            </div>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
