#ifndef BACKTESTENGINE_H
#define BACKTESTENGINE_H

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>
#include <mutex>
#include <arrow/api.h>
#include <spdlog/spdlog.h>

class BacktestEngine {
public:
    // Transaction action type.
    enum class Action { Buy, Sell };

    // Struct to store a single transaction.
    struct Transaction {
        Action action;          // Buy or Sell (only one per transaction)
        std::string ticker;     // Ticker symbol
        int quantity;           // Number of shares
        double price;           // Price per share at the time of transaction
        std::string datetime;   // Timestamp from the Arrow data row
    };

    // Abstract base class for strategies.
    // The user can derive from this class to implement custom tick-by-tick trading logic.
    class Strategy {
    public:
        virtual ~Strategy() = default;
        // Called on every tick for a given ticker.
        // currentIndex refers to the row index within the Arrow table.
        // currentHolding is the current number of shares held.
        // If the strategy wants to trigger a transaction, it returns a Transaction.
        // Otherwise, it returns std::nullopt.
        virtual std::optional<Transaction> onTick(const std::string& ticker,
                                                    const std::shared_ptr<arrow::Table>& table,
                                                    size_t currentIndex,
                                                    int currentHolding) = 0;
    };

    BacktestEngine();
    ~BacktestEngine() = default;

    // Adds ticker data as loaded by the DataLoader to the engine.
    void addTickerData(const std::string& ticker, const std::shared_ptr<arrow::Table>& table);

    // Registers a strategy for a particular ticker.
    void registerStrategy(const std::string& ticker, std::unique_ptr<Strategy> strategy);

    // Runs the backtesting simulation with concurrency.
    void runBacktest();

    // Returns a summary string with portfolio holdings and transaction count.
    std::string getPortfolioMetrics() const;

private:
    // Mapping from ticker symbol to its Arrow table data.
    std::map<std::string, std::shared_ptr<arrow::Table>> tickerData_;
    // Current row index for each ticker.
    std::map<std::string, size_t> tickerIndices_;
    // Registered strategy per ticker.
    std::map<std::string, std::unique_ptr<Strategy>> strategies_;
    // Current holdings: ticker -> number of shares held.
    std::map<std::string, int> holdings_;
    // List of transactions executed during the backtest.
    std::vector<Transaction> transactions_;
    // Mutex to protect shared updates in concurrent tasks.
    std::mutex mtx_;

    // Helper: logs a transaction via spdlog.
    void logTransaction(const Transaction& tx);
};

#endif // BACKTESTENGINE_H
