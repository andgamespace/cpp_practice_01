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
    const [tickers, setTickers] = useState<string[]>([])
    const [selectedTicker, setSelectedTicker] = useState<string>('AAPL')
    const [transactions, setTransactions] = useState<any[]>([])
    const [isLoading, setIsLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    // Function to fetch ticker data
    const fetchTickerData = (ticker: string) => {
        setIsLoading(true);
        setError(null);
        
        axios.get(`http://localhost:8080/time-series/${ticker}`)
            .then((res) => {
                if (res.data?.data) {
                    setTimeSeries(res.data.data);
                }
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to fetch data for ${ticker}: ${err.message}`);
            })
            .finally(() => {
                setIsLoading(false);
            });
    };
    
    // Function to fetch portfolio data
    const fetchPortfolioData = () => {
        axios.get('http://localhost:8080/portfolio/live')
            .then((res) => {
                if (res.data?.portfolio) {
                    setPortfolio(res.data.portfolio);
                }
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to fetch portfolio data: ${err.message}`);
            });
    };
    
    // Function to fetch transactions
    const fetchTransactions = () => {
        axios.get('http://localhost:8080/transactions')
            .then((res) => {
                if (res.data?.transactions) {
                    setTransactions(res.data.transactions);
                }
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to fetch transactions: ${err.message}`);
            });
    };
    
    // Function to start backtest
    const startBacktest = () => {
        axios.post('http://localhost:8080/backtest/start')
            .then((res) => {
                console.log('Backtest started:', res.data);
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to start backtest: ${err.message}`);
            });
    };
    
    // Function to reset backtest
    const resetBacktest = () => {
        axios.post('http://localhost:8080/backtest/reset')
            .then((res) => {
                console.log('Backtest reset:', res.data);
                // Refresh data after reset
                fetchPortfolioData();
                fetchTransactions();
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to reset backtest: ${err.message}`);
            });
    };

    // Fetch data on mount and when selected ticker changes
    useEffect(() => {
        // Fetch available tickers
        axios.get('http://localhost:8080/tickers')
            .then((res) => {
                if (res.data?.tickers) {
                    setTickers(res.data.tickers);
                }
            })
            .catch((err) => {
                console.error(err);
                setError(`Failed to fetch tickers: ${err.message}`);
            });
        
        // Fetch initial data
        fetchTickerData(selectedTicker);
        fetchPortfolioData();
        fetchTransactions();
        
        // Set up periodic refresh for portfolio data
        const refreshInterval = setInterval(() => {
            fetchPortfolioData();
        }, 5000); // Refresh every 5 seconds
        
        return () => {
            clearInterval(refreshInterval);
        };
    }, [selectedTicker]);

    // Establish WebSocket connection with reconnection logic
    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        const reconnectInterval = 2000; // 2 seconds
        let reconnectTimeout: number | null = null;

        // Function to create and setup WebSocket
        const setupWebSocket = () => {
            // Try different ports if configured
            const ports = [8080, 9000, 10000];
            const port = ports[reconnectAttempts % ports.length];
            
            ws = new WebSocket(`ws://localhost:${port}/ws`);

            ws.onopen = () => {
                console.log(`[WebSocket] Connected on port ${port}`);
                reconnectAttempts = 0;
                
                // Subscribe to portfolio updates
                const subscribeMsg = {
                    command: "subscribe",
                    stream: "portfolio"
                };
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(subscribeMsg));
                }
            };

            ws.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    console.log('[WebSocket] Message:', data);
                    
                    // Handle different message types
                    if (data.type === 'portfolio_update') {
                        setPortfolio(data.data || {});
                    }
                    
                    // Add to message log
                    setWsMessages((prev) => {
                        // Keep only the last 20 messages
                        const newMessages = [...prev, evt.data];
                        if (newMessages.length > 20) {
                            return newMessages.slice(newMessages.length - 20);
                        }
                        return newMessages;
                    });
                } catch (e) {
                    console.error('[WebSocket] Error parsing message:', e);
                    setWsMessages((prev) => [...prev, `Error: ${evt.data}`]);
                }
            };

            ws.onclose = () => {
                console.log('[WebSocket] Disconnected');
                
                // Try to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    console.log(`[WebSocket] Attempting to reconnect (${reconnectAttempts + 1}/${maxReconnectAttempts})...`);
                    reconnectTimeout = window.setTimeout(setupWebSocket, reconnectInterval);
                    reconnectAttempts++;
                } else {
                    console.error('[WebSocket] Max reconnect attempts reached');
                }
            };

            ws.onerror = (error) => {
                console.error('[WebSocket] Error:', error);
            };
        };

        // Initial setup
        setupWebSocket();

        // Cleanup function
        return () => {
            if (ws) {
                ws.close();
            }
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }
        };
    }, []);

    return (
        <div className="App">
            <header>
                <h1>Trading Environment Dashboard</h1>
                {error && <div className="error-message">{error}</div>}
            </header>
            
            <div className="controls">
                <div className="control-group">
                    <label>Select Ticker:</label>
                    <select
                        value={selectedTicker}
                        onChange={(e) => setSelectedTicker(e.target.value)}
                        disabled={isLoading}
                    >
                        {tickers.map(ticker => (
                            <option key={ticker} value={ticker}>{ticker}</option>
                        ))}
                    </select>
                </div>
                
                <div className="control-group">
                    <button onClick={startBacktest} disabled={isLoading}>
                        Start Backtest
                    </button>
                    <button onClick={resetBacktest} disabled={isLoading}>
                        Reset Backtest
                    </button>
                </div>
            </div>
            
            <div className="dashboard-grid">
                <div className="chart-container">
                    <h2>Price Chart: {selectedTicker}</h2>
                    {isLoading ? (
                        <div className="loading">Loading...</div>
                    ) : (
                        <div style={{ width: '100%', height: 400 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={timeSeries}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis
                                        dataKey="datetime"
                                        tick={{ fontSize: 12 }}
                                        angle={-45}
                                        textAnchor="end"
                                        height={70}
                                    />
                                    <YAxis />
                                    <Tooltip />
                                    <Line type="monotone" dataKey="open" stroke="#8884d8" name="Open" dot={false} />
                                    <Line type="monotone" dataKey="high" stroke="#82ca9d" name="High" dot={false} />
                                    <Line type="monotone" dataKey="low" stroke="#ff8042" name="Low" dot={false} />
                                    <Line type="monotone" dataKey="close" stroke="#ff4569" name="Close" dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
                
                <div className="metrics-container">
                    <h2>Portfolio Metrics</h2>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <h3>Balance</h3>
                            <p className="metric-value">${portfolio.cash?.toFixed(2) || '0.00'}</p>
                        </div>
                        <div className="metric-card">
                            <h3>Initial Balance</h3>
                            <p className="metric-value">${portfolio.initialBalance?.toFixed(2) || '0.00'}</p>
                        </div>
                        <div className="metric-card">
                            <h3>Return</h3>
                            <p className="metric-value">{portfolio.percentChange?.toFixed(2) || '0.00'}%</p>
                        </div>
                        <div className="metric-card">
                            <h3>Transactions</h3>
                            <p className="metric-value">{portfolio.transactions || '0'}</p>
                        </div>
                        <div className="metric-card">
                            <h3>Wins</h3>
                            <p className="metric-value">{portfolio.wins || '0'}</p>
                        </div>
                        <div className="metric-card">
                            <h3>Losses</h3>
                            <p className="metric-value">{portfolio.losses || '0'}</p>
                        </div>
                    </div>
                    
                    <h3>Holdings</h3>
                    <div className="holdings-list">
                        {portfolio.holdings?.length > 0 ? (
                            <ul>
                                {portfolio.holdings.map((holding: any, idx: number) => (
                                    <li key={idx}>{holding.ticker}: {holding.quantity} shares</li>
                                ))}
                            </ul>
                        ) : (
                            <p>No current holdings</p>
                        )}
                    </div>
                </div>
                
                <div className="transactions-container">
                    <h2>Recent Transactions</h2>
                    {transactions.length > 0 ? (
                        <table className="transactions-table">
                            <thead>
                                <tr>
                                    <th>Date/Time</th>
                                    <th>Ticker</th>
                                    <th>Action</th>
                                    <th>Quantity</th>
                                    <th>Price</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {transactions.slice(0, 10).map((tx, idx) => (
                                    <tr key={idx}>
                                        <td>{tx.datetime}</td>
                                        <td>{tx.ticker}</td>
                                        <td className={tx.action === 'BUY' ? 'buy' : tx.action === 'SELL' ? 'sell' : ''}>
                                            {tx.action}
                                        </td>
                                        <td>{tx.quantity}</td>
                                        <td>${tx.price?.toFixed(2) || '0.00'}</td>
                                        <td>${tx.value?.toFixed(2) || '0.00'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <p>No transactions yet</p>
                    )}
                </div>
                
                <div className="websocket-container">
                    <h2>Live Updates</h2>
                    <div className="websocket-messages">
                        {wsMessages.length > 0 ? (
                            <ul>
                                {wsMessages.map((msg, idx) => {
                                    try {
                                        const parsedMsg = JSON.parse(msg);
                                        return (
                                            <li key={idx} className="message">
                                                <span className="message-type">{parsedMsg.type || 'unknown'}</span>
                                                <span className="message-time">
                                                    {new Date().toLocaleTimeString()}
                                                </span>
                                            </li>
                                        );
                                    } catch {
                                        return (
                                            <li key={idx} className="message">
                                                <span className="message-raw">{msg}</span>
                                            </li>
                                        );
                                    }
                                })}
                            </ul>
                        ) : (
                            <p>No messages received yet</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default App
