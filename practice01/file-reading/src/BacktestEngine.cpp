#include "BacktestEngine.h"
#include <sstream>
#include <chrono>
#include <thread>

BacktestEngine::BacktestEngine()
    : tickerData_{}, tickerIndices_{}, strategies_{}, holdings_{}, transactions_{} {
    spdlog::info("BacktestEngine initialized");
}

void BacktestEngine::addTickerData(const std::string& ticker, const std::shared_ptr<arrow::Table>& table) {
    tickerData_[ticker] = table;
    tickerIndices_[ticker] = 0;
    holdings_[ticker] = 0;
    spdlog::info("Added ticker data for {}", ticker);
}

void BacktestEngine::registerStrategy(const std::string& ticker, std::unique_ptr<Strategy> strategy) {
    strategies_[ticker] = std::move(strategy);
    spdlog::info("Registered strategy for ticker {}", ticker);
}

void BacktestEngine::logTransaction(const Transaction& tx) {
    spdlog::info("Transaction: {} {} shares of {} at ${} on {}",
                 (tx.action == Action::Buy ? "BUY" : "SELL"),
                 tx.quantity,
                 tx.ticker,
                 tx.price,
                 tx.datetime);
}

void BacktestEngine::runBacktest() {
    spdlog::info("Starting backtest simulation");

    // Run until all tickers have been fully processed.
    bool dataRemaining = true;
    size_t iteration = 0;
    constexpr size_t updateFrequency = 16; // Update portfolio metrics every 16 iterations

    while (dataRemaining) {
        dataRemaining = false;
        // Iterate over each ticker.
        for (auto& [ticker, table] : tickerData_) {
            size_t& index = tickerIndices_[ticker];
            if (index < table->num_rows()) {
                dataRemaining = true;
                spdlog::debug("Processing {} at index {}", ticker, index);

                // If a strategy has been registered for this ticker, invoke its onTick method.
                if (strategies_.contains(ticker)) {
                    auto& strategy = strategies_[ticker];
                    auto txOpt = strategy->onTick(ticker, table, index, holdings_[ticker]);
                    if (txOpt.has_value()) {
                        const auto& tx = txOpt.value();
                        transactions_.push_back(tx);
                        logTransaction(tx);
                        // Update holdings based on the transaction action.
                        if (tx.action == Action::Buy) {
                            holdings_[ticker] += tx.quantity;
                        } else if (tx.action == Action::Sell) {
                            holdings_[ticker] -= tx.quantity;
                        }
                    }
                }
                // Move to the next data row for this ticker.
                index++;
            }
        }
        iteration++;

        // Periodically log portfolio metrics (and later, these metrics can be sent via a websocket).
        if (iteration % updateFrequency == 0) {
            spdlog::info("Iteration {}: Portfolio Metrics: {}", iteration, getPortfolioMetrics());
            // Simulate a processing delay (if needed).
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    spdlog::info("Backtest simulation completed after {} iterations", iteration);
}

std::string BacktestEngine::getPortfolioMetrics() const {
    std::ostringstream oss;
    oss << "Holdings: ";
    for (const auto& [ticker, quantity] : holdings_) {
        oss << ticker << "=" << quantity << " ";
    }
    oss << "| Total Transactions: " << transactions_.size();
    return oss.str();
}

#ifdef BACKTEST_ENGINE_TEST
// -------------------------
// Testing the BacktestEngine
// -------------------------

#include "DataLoader.h"
#include <iostream>

// A dummy strategy for testing that uses simple threshold logic on the "close" price.
// If the close price is below a buy threshold, it triggers a BUY transaction.
// If above a sell threshold and there is a current holding, it triggers a SELL transaction.
class DummyStrategy : public BacktestEngine::Strategy {
public:
    DummyStrategy(double buyThreshold, double sellThreshold)
        : buyThreshold_(buyThreshold), sellThreshold_(sellThreshold) {}

    std::optional<BacktestEngine::Transaction> onTick(const std::string& ticker,
                                                        const std::shared_ptr<arrow::Table>& table,
                                                        size_t currentIndex,
                                                        int currentHolding) override {
        // Retrieve the "close" price from column index 4.
        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
        if (currentIndex >= table->num_rows()) {
            return std::nullopt;
        }
        double price = closeArray->Value(currentIndex);

        BacktestEngine::Transaction tx;
        tx.ticker = ticker;
        // For the datetime, use the value from the first column.
        auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
        tx.datetime = datetimeArray->GetString(currentIndex);
        tx.price = price;

        // Buy one share if price is below the buy threshold.
        if (price < buyThreshold_) {
            tx.action = BacktestEngine::Action::Buy;
            tx.quantity = 1;
            return tx;
        }
        // Sell one share if price is above the sell threshold and there is an existing holding.
        else if (price > sellThreshold_ && currentHolding > 0) {
            tx.action = BacktestEngine::Action::Sell;
            tx.quantity = 1;
            return tx;
        }

        return std::nullopt;
    }

private:
    double buyThreshold_;
    double sellThreshold_;
};

int main() {
    spdlog::info("Testing BacktestEngine");

    // Create an instance of the DataLoader.
    DataLoader loader;
    std::string baseDir = "../src/stock_data/";
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};

    BacktestEngine engine;

    // Load data for each ticker and register a dummy strategy.
    for (const auto& ticker : tickers) {
        std::vector<std::string> filePaths = {
            baseDir + "time-series-" + ticker + "-5min.csv",
            baseDir + "time-series-" + ticker + "-5min(1).csv",
            baseDir + "time-series-" + ticker + "-5min(2).csv"
        };

        if (loader.loadTickerData(ticker, filePaths)) {
            auto table = loader.getTickerData(ticker);
            if (table) {
                engine.addTickerData(ticker, table);
                // For testing, register the DummyStrategy with arbitrary thresholds.
                engine.registerStrategy(ticker, std::make_unique<DummyStrategy>(100.0, 200.0));
            }
        } else {
            spdlog::error("Failed to load data for ticker: {}", ticker);
        }
    }

    // Run the backtest simulation.
    engine.runBacktest();

    std::cout << "Final Portfolio Metrics: " << engine.getPortfolioMetrics() << std::endl;

    return 0;
}
#endif // BACKTEST_ENGINE_TEST
