#ifndef BACKTESTENGINE_H
#define BACKTESTENGINE_H

#include <string>
#include <vector>
#include <deque>         // Use deque for efficient front removal
#include <map>
#include <optional>
#include <memory>
#include <mutex>
#include <functional>
#include <arrow/api.h>
#include <spdlog/spdlog.h>

/**
 * @brief BacktestEngine manages tick-by-tick simulation of trading strategies.
 */
class BacktestEngine {
public:
    // Transaction action type.
    enum class Action { Buy, Sell };

    // A single trade transaction.
    struct Transaction {
        Action action;          // BUY or SELL
        std::string ticker;     // Ticker symbol
        int quantity;           // Number of shares
        double price;           // Price per share
        std::string datetime;   // Timestamp from the Arrow row
    };

    // Abstract base for trading strategies.
    class Strategy {
    public:
        virtual ~Strategy() = default;
        // Called at each tick. Return a Transaction if an action should occur.
        virtual std::optional<Transaction> onTick(const std::string &ticker,
                                                    const std::shared_ptr<arrow::Table> &table,
                                                    size_t currentIndex,
                                                    int currentHolding) = 0;
    };

    BacktestEngine();
    ~BacktestEngine() = default;

    // Overloaded assignment operator to set a custom starting cash balance.
    BacktestEngine &operator=(double new_balance);

    // Add ticker data for one ticker.
    void addTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table);

    // Bulk-set ticker data from a map.
    void setTickerData(const std::map<std::string, std::shared_ptr<arrow::Table>> &data);

    // Register a strategy for a given ticker.
    void registerStrategy(const std::string &ticker, std::unique_ptr<Strategy> strategy);

    // Set a callback to broadcast portfolio metrics (e.g. via WebSocket).
    void setBroadcastCallback(std::function<void(const std::string &)> callback);

    // Runs the backtest concurrently using Taskflow.
    void runBacktest();

    // Returns portfolio metrics as a summary string.
    std::string getPortfolioMetrics() const;

    // Optional getter for cash balance.
    double getCashBalance() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return cash_balance_;
    }

    // === New RL interface ===

    // Structure for a single step result.
    struct StepResult {
        std::map<std::string, double> observations; // e.g. current close prices per ticker
        double reward;      // For example, relative change in cash balance (or risk-adjusted)
        bool done;          // True if no more data is available for any ticker
    };

    // Process one tick (bar) for each ticker and return observations/reward.
    StepResult step();

    // Process one tick using external actions from DRL agents.
    // externalActions: key = ticker, value = number of shares to trade (positive: buy, negative: sell).
    StepResult step(const std::map<std::string, double> &externalActions);

    // Reset the simulation to the beginning (for a new RL episode).
    void reset();

private:
    // Ticker data and current tick index.
    std::map<std::string, std::shared_ptr<arrow::Table>> tickerData_;
    std::map<std::string, size_t> tickerIndices_;

    // Strategy per ticker.
    std::map<std::string, std::unique_ptr<Strategy>> strategies_;

    // Holdings and transactions.
    std::map<std::string, int> holdings_;
    std::vector<Transaction> transactions_;

    // Mutex for shared state.
    mutable std::mutex mtx_;

    // Financial metrics.
    double initial_balance_;  // default: $100,000
    double cash_balance_;
    int wins_;
    int losses_;

    // For computing wins/losses, store buy prices by ticker.
    std::map<std::string, std::deque<double>> buy_prices_;

    // Callback for broadcasting metrics.
    std::function<void(const std::string &)> broadcast_callback_;

    // Helper: logs a transaction to spdlog.
    void logTransaction(const Transaction &tx);
};

#endif // BACKTESTENGINE_H
