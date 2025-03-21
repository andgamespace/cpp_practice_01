#ifndef BACKTESTENGINE_H
#define BACKTESTENGINE_H

#include <string>
#include <vector>
#include <deque>         // Use deque for efficient front removal
#include <map>
#include <unordered_map>
#include <optional>
#include <memory>
#include <mutex>
#include <functional>
#include <arrow/api.h>
#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

// Optional PyTorch support
#if PYTORCH_AVAILABLE
// Forward declaration for PyTorch integration
namespace torch {
    class Tensor;
}
#endif

/**
 * @brief BacktestEngine manages tick-by-tick simulation of trading strategies.
 * Supports both traditional backtesting and reinforcement learning interfaces.
 */
class BacktestEngine {
public:
    // Transaction action type.
    enum class Action { Buy, Sell, Hold };

    // A single trade transaction.
    struct Transaction {
        Action action;          // BUY, SELL, or HOLD
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

    // Performance metrics for a backtest
    struct PerformanceMetrics {
        double initialBalance;
        double finalBalance;
        double totalReturn;
        double annualizedReturn;
        double sharpeRatio;
        double maxDrawdown;
        int totalTrades;
        int winningTrades;
        int losingTrades;
        double winRate;
        double profitFactor;
        double averageWin;
        double averageLoss;
        double expectancy;
    };

    // Structure for a single step result.
    struct StepResult {
        std::map<std::string, double> observations; // e.g. current close prices per ticker
        double reward;      // For example, relative change in cash balance (or risk-adjusted)
        bool done;          // True if no more data is available for any ticker
        std::map<std::string, std::vector<double>> features; // Additional features for ML models
    };

    BacktestEngine();
    ~BacktestEngine() = default;

    // Overloaded assignment operator to set a custom starting cash balance.
    BacktestEngine &operator=(double new_balance);

    // Add ticker data for one ticker.
    void addTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table);

    // Bulk-set ticker data from a map.
    void setTickerData(const std::map<std::string, std::shared_ptr<arrow::Table>> &data);
    void setTickerData(const std::unordered_map<std::string, std::shared_ptr<arrow::Table>> &data);

    // Register a strategy for a given ticker.
    void registerStrategy(const std::string &ticker, std::unique_ptr<Strategy> strategy);

    // Set a callback to broadcast portfolio metrics (e.g. via WebSocket).
    void setBroadcastCallback(std::function<void(const std::string &)> callback);
    
    // Set a callback to broadcast JSON metrics for the frontend
    void setJsonBroadcastCallback(std::function<void(const std::string &)> callback);

    // Runs the backtest concurrently using Taskflow.
    void runBacktest();

    // Returns portfolio metrics as a summary string.
    std::string getPortfolioMetrics() const;
    
    // Returns detailed performance metrics
    PerformanceMetrics getPerformanceMetrics() const;
    
    // Returns portfolio metrics as a JSON string for the frontend
    std::string getPortfolioMetricsJson() const;

    // Get current holdings
    std::map<std::string, int> getHoldings() const;
    
    // Get transaction history
    const std::vector<Transaction>& getTransactions() const;

    // Optional getter for cash balance.
    double getCashBalance() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return cash_balance_;
    }

    // === RL interface ===

    // Process one tick (bar) for each ticker and return observations/reward.
    StepResult step();

    // Process one tick using external actions from DRL agents.
    // externalActions: key = ticker, value = number of shares to trade (positive: buy, negative: sell).
    StepResult step(const std::map<std::string, double> &externalActions);
    
#if PYTORCH_AVAILABLE
    // Process one tick using PyTorch tensor actions
    StepResult stepWithTensor(const torch::Tensor &actions);
#endif

    // Reset the simulation to the beginning (for a new RL episode).
    void reset();
    
    // Get the observation space dimension (number of features per ticker)
    size_t getObservationDimension() const;
    
    // Get the action space dimension (number of tickers)
    size_t getActionDimension() const;
    
    // Get available tickers
    std::vector<std::string> getAvailableTickers() const;
    
    // Enable/disable debug logging
    void setDebugMode(bool enable) { debug_ = enable; }

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
    bool debug_;

    // For computing wins/losses, store buy prices by ticker.
    std::map<std::string, std::deque<double>> buy_prices_;

    // Callback for broadcasting metrics.
    std::function<void(const std::string &)> broadcast_callback_;
    std::function<void(const std::string &)> json_broadcast_callback_;

    // Helper: logs a transaction to spdlog.
    void logTransaction(const Transaction &tx);
    
    // Helper: process a transaction
    void processTransaction(const Transaction &tx);
    
    // Helper: calculate features for ML models
    std::map<std::string, std::vector<double>> calculateFeatures();
    
    // Helper: calculate technical indicators
    std::vector<double> calculateIndicators(const std::string &ticker, size_t currentIndex);
    
#if PYTORCH_AVAILABLE
    // Helper: convert map to tensor
    torch::Tensor mapToTensor(const std::map<std::string, double> &map);
    
    // Helper: convert tensor to map
    std::map<std::string, double> tensorToMap(const torch::Tensor &tensor);
#endif
};

#endif // BACKTESTENGINE_H
