import React, { useEffect, useState } from 'react'
import axios from 'axios'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer,
} from 'recharts'
import './App.css'

interface TimeSeriesPoint {
    datetime: string
    open: number
    high: number
    low: number
    close: number
    volume: number
}

function App() {
    const [timeSeries, setTimeSeries] = useState<TimeSeriesPoint[]>([])
    const [portfolio, setPortfolio] = useState<any>({})
    const [wsMessages, setWsMessages] = useState<string[]>([])

    // Fetch REST endpoints on mount.
    useEffect(() => {
        axios.get('http://localhost:8080/time-series/AAPL')
            .then((res) => {
                if (res.data?.data) {
                    setTimeSeries(res.data.data)
                }
            })
            .catch(console.error)

        axios.get('http://localhost:8080/portfolio/live')
            .then((res) => {
                if (res.data?.portfolio) {
                    setPortfolio(res.data.portfolio)
                }
            })
            .catch(console.error)
    }, [])

    // Establish WebSocket connection.
    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8080/ws')

        ws.onopen = () => {
            console.log('[WebSocket] Connected')
        }

        ws.onmessage = (evt) => {
            console.log('[WebSocket] Message:', evt.data)
            setWsMessages((prev) => [...prev, evt.data])
        }

        ws.onclose = () => {
            console.log('[WebSocket] Disconnected')
        }

        return () => {
            ws.close()
        }
    }, [])

    return (
        <div className="App">
            <h1>My Trading Dashboard</h1>
            <div style={{ marginBottom: '2rem' }}>
                <h2>Time Series (AAPL)</h2>
                <div style={{ width: '90%', height: 400, margin: '0 auto' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={timeSeries}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="datetime" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="open" stroke="#8884d8" name="Open" />
                            <Line type="monotone" dataKey="close" stroke="#82ca9d" name="Close" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
            <div style={{ marginBottom: '2rem' }}>
                <h2>Portfolio Metrics</h2>
                <p>Timestamp: {portfolio.timestamp}</p>
                <p>Balance: {portfolio.balance}</p>
                <p>Equity: {portfolio.equity}</p>
                <p>Open Positions: {portfolio.openPositions}</p>
                <p>Profit/Loss: {portfolio.profitLoss}</p>
            </div>
            <div style={{ marginBottom: '2rem' }}>
                <h2>WebSocket Messages</h2>
                <ul>
                    {wsMessages.map((msg, idx) => (
                        <li key={idx}>{msg}</li>
                    ))}
                </ul>
            </div>
        </div>
    )
}

export default App
