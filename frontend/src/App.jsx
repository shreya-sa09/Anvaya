import { useState } from 'react'
import {
    LayoutDashboard,
    User,
    Bell,
    Zap,
    Radio
} from 'lucide-react'
import Portfolio from './pages/Portfolio'
import CustomerProfile from './pages/CustomerProfile'
import Alerts from './pages/Alerts'
import Interventions from './pages/Interventions'
import LiveScoring from './pages/LiveScoring'

const navSections = [
    {
        title: 'Monitoring',
        items: [
            { key: 'portfolio', label: 'Portfolio Overview', icon: LayoutDashboard },
            { key: 'customer', label: 'Customer Profile', icon: User },
            { key: 'alerts', label: 'Alerts', icon: Bell }
        ]
    },
    {
        title: 'Decisions & Actions',
        items: [
            { key: 'interventions', label: 'Interventions Tracker', icon: Zap },
            { key: 'live', label: 'Live Scoring', icon: Radio }
        ]
    }
]

export default function App() {
    const [page, setPage] = useState('portfolio')
    const [selectedCustomer, setSelectedCustomer] = useState(null)

    // P11, P12, P14: Lifted states to be shared across pages
    const [dismissedAlerts, setDismissedAlerts] = useState([])
    const [readAlerts, setReadAlerts] = useState([])
    const [rmDecisions, setRmDecisions] = useState({}) // Format: { customerId: decision }

    function goToCustomer(id) {
        setSelectedCustomer(id)
        // Auto mark as read when viewed
        if (id && !readAlerts.includes(id)) {
            setReadAlerts(prev => [...prev, id])
        }
        setPage('customer')
    }

    return (
        <div className="flex h-screen overflow-hidden"
            style={{ backgroundColor: '#0B1120' }}>

            {/* Sidebar */}
            <aside className="w-64 flex-shrink-0 flex flex-col"
                style={{
                    backgroundColor: '#0D1424',
                    borderRight: '1px solid #1F2937'
                }}>

                {/* Logo */}
                <div className="px-6 py-8">
                    <div className="text-2xl font-black tracking-widest"
                        style={{ color: '#06B6D4' }}>
                        ANVAYA
                    </div>
                    <div className="text-[10px] font-bold tracking-wider mt-1.5 uppercase"
                        style={{ color: '#64748B' }}>
                        Pre-Delinquency Intelligence
                    </div>
                </div>

                {/* Nav */}
                <nav className="flex-1 px-4 space-y-7">
                    {navSections.map(section => (
                        <div key={section.title} className="space-y-2">
                            {/* Section Label */}
                            <div className="px-3 text-[10px] font-bold tracking-widest text-[#475569] uppercase">
                                {section.title}
                            </div>
                            
                            {/* Section Links */}
                            <div className="space-y-1">
                                {section.items.map(item => {
                                    const Icon = item.icon
                                    const active = page === item.key
                                    return (
                                        <button
                                            key={item.key}
                                            onClick={() => {
                                                // P6/Sidebar reset: clear selected customer when clicking Customer Profile in sidebar directly
                                                if (item.key === 'customer') {
                                                    setSelectedCustomer(null)
                                                }
                                                setPage(item.key)
                                            }}
                                            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl
                                               text-sm font-medium transition-all duration-150 cursor-pointer
                                               hover:bg-[#161E2E] hover:text-[#FFFFFF]"
                                            style={{
                                                backgroundColor: active
                                                    ? 'rgba(6, 182, 212, 0.08)' : 'transparent',
                                                color: active ? '#06B6D4' : '#94A3B8'
                                            }}
                                        >
                                            <Icon size={18} style={{ color: active ? '#06B6D4' : '#475569' }} />
                                            <span>{item.label}</span>
                                        </button>
                                    )
                                })}
                            </div>
                        </div>
                    ))}
                </nav>

                {/* Footer */}
                <div className="px-6 py-4 text-xs"
                    style={{ color: '#64748B' }}>
                    © 2026 Risk Analytics
                </div>
            </aside>

            {/* Main content */}
            <main className="flex-1 overflow-y-auto p-8">
                {page === 'portfolio' && (
                    <Portfolio onViewCustomer={goToCustomer} />
                )}
                {page === 'customer' && (
                    <CustomerProfile 
                        customerId={selectedCustomer} 
                        rmDecisions={rmDecisions}
                        onSaveDecision={(id, dec) => setRmDecisions(prev => ({ ...prev, [id]: dec }))}
                    />
                )}
                {page === 'alerts' && (
                    <Alerts 
                        onViewCustomer={goToCustomer} 
                        dismissedAlerts={dismissedAlerts}
                        onDismissAlert={(id) => setDismissedAlerts(prev => [...prev, id])}
                        readAlerts={readAlerts}
                        onMarkRead={(id) => {
                            if (!readAlerts.includes(id)) {
                                setReadAlerts(prev => [...prev, id])
                            }
                        }}
                    />
                )}
                {page === 'interventions' && (
                    <Interventions 
                        onViewCustomer={goToCustomer} 
                        rmDecisions={rmDecisions}
                    />
                )}
                {page === 'live' && (
                    <LiveScoring onViewCustomer={goToCustomer} />
                )}
            </main>
        </div>
    )
}
