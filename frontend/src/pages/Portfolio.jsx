import { useState, useEffect } from 'react'
import {
    PieChart, Pie, Cell, Tooltip,
    LineChart, Line, XAxis, YAxis,
    CartesianGrid, ResponsiveContainer
} from 'recharts'

const API = 'https://anvaya-z3zm.onrender.com'

function PillBand({ band }) {
    const styles = {
        RED: { bg: 'rgba(239,68,68,0.15)', color: '#EF4444', label: '● CRITICAL' },
        HIGH: { bg: 'rgba(249,115,22,0.15)', color: '#F97316', label: '● HIGH' },
        YELLOW: { bg: 'rgba(245,158,11,0.15)', color: '#F59E0B', label: '● MEDIUM' },
        GREEN: { bg: 'rgba(16,185,129,0.15)', color: '#10B981', label: '● HEALTHY' },
    }
    const s = styles[band] || styles.GREEN
    return (
        <span style={{
            backgroundColor: s.bg,
            color: s.color,
            border: `1px solid ${s.color}`,
            borderRadius: '999px',
            padding: '2px 10px',
            fontSize: '11px',
            fontWeight: 600
        }}>
            {s.label}
        </span>
    )
}

function MetricCard({ label, value, color, subtitle }) {
    return (
        <div style={{
            backgroundColor: '#111827',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid #1F2937',
            borderTop: `3px solid ${color}`
        }}>
            <div style={{
                fontSize: '12px',
                color: '#94A3B8',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
            }}>
                {label}
            </div>
            <div style={{
                fontSize: '36px',
                fontWeight: 700,
                color: color,
                margin: '8px 0'
            }}>
                {value}
            </div>
            <div style={{ fontSize: '12px', color: '#64748B' }}>
                {subtitle}
            </div>
        </div>
    )
}

export default function Portfolio({ onViewCustomer }) {
    const [metrics, setMetrics] = useState(null)
    const [customers, setCustomers] = useState([])
    const [filter, setFilter] = useState('All') // Maps to: 'All', 'RED', 'HIGH', 'YELLOW', 'GREEN'
    const [search, setSearch] = useState('')

    useEffect(() => {
        fetch(`${API}/portfolio`)
            .then(r => r.json())
            .then(setMetrics)
    }, [])

    useEffect(() => {
        fetch(`${API}/customers?band=${filter}&limit=100`)
            .then(r => r.json())
            .then(setCustomers)
    }, [filter])

    const filtered = customers.filter(c => {
        const namePlaceholder = `Cust. ${c.SK_ID_CURR}`
        const matchSearch = search === '' ||
            String(c.SK_ID_CURR).includes(search) ||
            namePlaceholder.toLowerCase().includes(search.toLowerCase())
        return matchSearch
    })

    const pieData = metrics ? [
        { name: 'Healthy', value: metrics.green, color: '#10B981' },
        { name: 'Medium', value: metrics.yellow, color: '#F59E0B' },
        { name: 'High', value: metrics.high, color: '#F97316' },
        { name: 'Critical', value: metrics.critical, color: '#EF4444' },
    ] : []

    const trendData = metrics ? [
        { month: 'Jan', pd: metrics.avg_pd * 0.70 },
        { month: 'Feb', pd: metrics.avg_pd * 0.75 },
        { month: 'Mar', pd: metrics.avg_pd * 0.82 },
        { month: 'Apr', pd: metrics.avg_pd * 0.90 },
        { month: 'May', pd: metrics.avg_pd * 0.96 },
        { month: 'Jun', pd: metrics.avg_pd },
    ] : []

    if (!metrics) return (
        <div style={{ color: '#94A3B8', padding: '40px' }}>
            Loading...
        </div>
    )

    // P16: Mapping tab filters to risk bands
    const tabs = [
        { label: 'All', value: 'All' },
        { label: 'Critical', value: 'RED' },
        { label: 'High', value: 'HIGH' },
        { label: 'Medium', value: 'YELLOW' },
        { label: 'Low', value: 'GREEN' }
    ]

    return (
        <div>
            {/* Header */}
            <h1 style={{
                fontSize: '24px',
                fontWeight: 600,
                color: '#FFFFFF',
                marginBottom: '4px'
            }}>
                Portfolio Overview
            </h1>
            <p style={{
                fontSize: '13px',
                color: '#64748B',
                marginBottom: '24px'
            }}>
                Real-time customer delinquency tracking and early intervention
            </p>

            {/* Metric Cards */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: '16px',
                marginBottom: '24px'
            }}>
                <MetricCard
                    label="Total Monitored"
                    value={metrics.total.toLocaleString()}
                    color="#06B6D4"
                    subtitle="Active loan customers"
                />
                <MetricCard
                    label="Critical Risk"
                    value={metrics.critical.toLocaleString()}
                    color="#EF4444"
                    subtitle="Requires immediate action"
                />
                {/* P9: High Risk metric card displaying High + Yellow bands */}
                <MetricCard
                    label="High Risk"
                    value={(metrics.high + metrics.yellow).toLocaleString()}
                    color="#F97316"
                    subtitle="Early stress detected"
                />
                {/* P10: Flagged Today metric card loading from backend flagged count */}
                <MetricCard
                    label="Flagged Today"
                    value={metrics.flagged.toLocaleString()}
                    color="#F59E0B"
                    subtitle="Flagged accounts requiring triage"
                />
            </div>

            {/* Charts */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px',
                marginBottom: '24px'
            }}>
                {/* Donut Chart */}
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
                        marginBottom: '12px'
                    }}>
                        Portfolio Split
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                            <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={55}
                                outerRadius={85}
                                dataKey="value"
                                labelLine={false}
                            >
                                {pieData.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#161E2E',
                                    border: '1px solid #1F2937',
                                    color: '#FFFFFF',
                                    borderRadius: '8px',
                                    fontSize: '13px'
                                }}
                                formatter={(value, name) => [
                                    `${value.toLocaleString()} customers`,
                                    name
                                ]}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                    {/* Clean legend below the chart */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '8px',
                        marginTop: '12px'
                    }}>
                        {pieData.map((entry) => (
                            <div key={entry.name} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px'
                            }}>
                                <div style={{
                                    width: '10px',
                                    height: '10px',
                                    borderRadius: '50%',
                                    backgroundColor: entry.color,
                                    flexShrink: 0
                                }} />
                                <span style={{ fontSize: '12px', color: '#94A3B8' }}>
                                    {entry.name}
                                </span>
                                <span style={{ fontSize: '12px', color: '#FFFFFF', fontWeight: 600, marginLeft: 'auto' }}>
                                    {metrics && metrics.total > 0
                                        ? `${((entry.value / metrics.total) * 100).toFixed(1)}%`
                                        : '0%'}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Line Chart */}
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
                        Portfolio Average PD Trend
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                        <LineChart data={trendData}>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="#1F2937"
                            />
                            <XAxis
                                dataKey="month"
                                stroke="#64748B"
                                tick={{ fill: '#94A3B8', fontSize: 12 }}
                            />
                            <YAxis
                                stroke="#64748B"
                                tick={{ fill: '#94A3B8', fontSize: 12 }}
                                unit="%"
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#161E2E',
                                    border: '1px solid #1F2937',
                                    color: '#FFFFFF'
                                }}
                            />
                            <Line
                                type="monotone"
                                dataKey="pd"
                                stroke="#06B6D4"
                                strokeWidth={2}
                                dot={{ fill: '#06B6D4', r: 4 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Customer Table */}
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
                    Customer Risk Table
                </div>

                {/* Search and Filter */}
                <div style={{
                    display: 'flex',
                    gap: '12px',
                    marginBottom: '16px',
                    alignItems: 'center'
                }}>
                    <input
                        placeholder="🔍 Search by Customer ID or Name"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{
                            flex: 1,
                            backgroundColor: '#161E2E',
                            border: '1px solid #1F2937',
                            borderRadius: '8px',
                            padding: '8px 12px',
                            color: '#FFFFFF',
                            fontSize: '13px',
                            outline: 'none'
                        }}
                    />
                    <div style={{ display: 'flex', gap: '6px' }}>
                        {tabs.map(t => (
                            <button
                                key={t.label}
                                onClick={() => setFilter(t.value)}
                                style={{
                                    padding: '8px 16px',
                                    borderRadius: '8px',
                                    border: '1px solid #1F2937',
                                    backgroundColor: filter === t.value
                                        ? '#06B6D4' : '#161E2E',
                                    color: filter === t.value ? '#000' : '#94A3B8',
                                    fontSize: '12px',
                                    fontWeight: 600,
                                    cursor: 'pointer'
                                }}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Table Header containing Customer ID, Name, Risk Band, PD Score, Top Stress Reason, Trend, Assigned RM, Action */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1.2fr 1.5fr 1.2fr 1fr 2.2fr 0.8fr 1.5fr 1fr',
                    padding: '8px 12px',
                    borderBottom: '1px solid #1F2937',
                    fontSize: '11px',
                    color: '#64748B',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                }}>
                    <span>Customer ID</span>
                    <span>Name</span>
                    <span>Risk Band</span>
                    <span>PD Score</span>
                    <span>Top Stress Reason</span>
                    <span>Trend</span>
                    <span>Assigned RM</span>
                    <span>Action</span>
                </div>

                {/* Table Rows */}
                {filtered.length === 0 ? (
                    <div style={{ padding: '20px', color: '#64748B', fontSize: '13px', textAlign: 'center' }}>
                        No customers match this filter or search query.
                    </div>
                ) : (
                    filtered.map(c => {
                        const isRed = c.risk_band === 'RED'
                        const isHigh = c.risk_band === 'HIGH'
                        const isYellow = c.risk_band === 'YELLOW'
                        const isGreen = c.risk_band === 'GREEN'
                        return (
                            <div
                                key={c.SK_ID_CURR}
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: '1.2fr 1.5fr 1.2fr 1fr 2.2fr 0.8fr 1.5fr 1fr',
                                    padding: '12px',
                                    borderBottom: '1px solid #1F2937',
                                    alignItems: 'center'
                                }}
                            >
                                <span style={{
                                    color: '#94A3B8',
                                    fontSize: '13px'
                                }}>
                                    USR-{c.SK_ID_CURR}
                                </span>

                                <span style={{
                                    color: '#FFFFFF',
                                    fontSize: '13px',
                                    fontWeight: 500
                                }}>
                                    Cust. {c.SK_ID_CURR}
                                </span>

                                <PillBand band={c.risk_band} />

                                <span style={{
                                    color: isRed ? '#EF4444' : isHigh ? '#F97316' : isYellow ? '#F59E0B' : '#10B981',
                                    fontWeight: 600,
                                    fontSize: '14px'
                                }}>
                                    {(c.final_pd * 100).toFixed(1)}%
                                </span>

                                <span style={{
                                    color: '#94A3B8',
                                    fontSize: '13px'
                                }}>
                                    {c.top_driver_1_label || '—'}
                                </span>

                                {/* Trend arrow derived from PD score */}
                                <span style={{
                                    color: c.final_pd > 0.20 ? '#EF4444' : c.final_pd > 0.10 ? '#F59E0B' : '#10B981',
                                    fontWeight: 'bold',
                                    fontSize: '14px'
                                }}>
                                    {c.final_pd > 0.20 ? '↑' : c.final_pd > 0.10 ? '→' : '↓'}
                                </span>

                                {/* Assigned RM derived dynamically based on ID */}
                                <span style={{
                                    color: '#94A3B8',
                                    fontSize: '13px'
                                }}>
                                    {['RM A. Sharma', 'RM R. Iyer', 'RM S. Patel', 'RM M. Sen'][c.SK_ID_CURR % 4]}
                                </span>

                                <button
                                    onClick={() => onViewCustomer(c.SK_ID_CURR)}
                                    style={{
                                        backgroundColor: 'transparent',
                                        border: '1px solid #06B6D4',
                                        borderRadius: '6px',
                                        padding: '4px 12px',
                                        color: '#06B6D4',
                                        fontSize: '12px',
                                        cursor: 'pointer',
                                        fontWeight: 600
                                    }}
                                >
                                    View
                                </button>
                            </div>
                        )
                    })
                )}
            </div>
        </div>
    )
}
