#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <numeric>
#include <arrow/api.h>
#include <spdlog/spdlog.h>
#include <iomanip>
#include <drogon/drogon.h>
#include <thread>
#include "DataLoader.h"
#include "BacktestEngine.h"

// Simple function to compute average of a container
template <typename T>
double average(const T &container) {
    if (container.empty()) return 0.0;
    double sum = std::accumulate(container.begin(), container.end(), 0.0);
    return sum / static_cast<double>(container.size());
}

/**
 * @brief MovingAverageCrossoverStrategy
 *        - shortPeriod = 5
 *        - longPeriod  = 20
 *        - Buys when shortSMA crosses above longSMA, sells when shortSMA < longSMA.
 *        - Holds at most 1 share.
 */
class MovingAverageCrossoverStrategy : public BacktestEngine::Strategy {
public:
    MovingAverageCrossoverStrategy(size_t shortPeriod = 5, size_t longPeriod = 20)
        : shortPeriod_(shortPeriod), longPeriod_(longPeriod),
          prevShortSMA_(0.0), prevLongSMA_(0.0)
    {}

    std::optional<BacktestEngine::Transaction> onTick(const std::string &ticker,
                                                      const std::shared_ptr<arrow::Table> &table,
                                                      size_t currentIndex,
                                                      int currentHolding) override
    {
        if (currentIndex >= table->num_rows()) {
            return std::nullopt;
        }
        // Grab the close price for the current row
        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
        double closePrice = closeArray->Value(currentIndex);

        // Update our ring buffers
        shortWindow_.push_back(closePrice);
        if (shortWindow_.size() > shortPeriod_) {
            shortWindow_.pop_front();
        }

        longWindow_.push_back(closePrice);
        if (longWindow_.size() > longPeriod_) {
            longWindow_.pop_front();
        }

        // If we don't yet have enough data for the long window, do nothing
        if (longWindow_.size() < longPeriod_) {
            return std::nullopt;
        }

        double shortSMA = average(shortWindow_);
        double longSMA  = average(longWindow_);

        // We look for crossing:
        // - if shortSMA > longSMA and we have 0 shares => buy 1 share
        // - if shortSMA < longSMA and we have 1 share => sell

        BacktestEngine::Transaction tx;
        tx.ticker = ticker;
        auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
        tx.datetime = datetimeArray->GetString(currentIndex);
        tx.price = closePrice;

        // Check for a cross from below to above (bullish)
        if (shortSMA > longSMA && currentHolding == 0) {
            tx.action = BacktestEngine::Action::Buy;
            tx.quantity = 1;
            return tx;
        }
        // Check for cross from above to below (bearish)
        else if (shortSMA < longSMA && currentHolding > 0) {
            tx.action = BacktestEngine::Action::Sell;
            tx.quantity = currentHolding;  // sell all holdings
            return tx;
        }
        // Otherwise, do nothing
        return std::nullopt;
    }

private:
    size_t shortPeriod_;
    size_t longPeriod_;
    double prevShortSMA_;
    double prevLongSMA_;

    // We store the last N closes in deques
    std::deque<double> shortWindow_;
    std::deque<double> longWindow_;
};

// Helper function to print the first few rows of an Arrow table.
void printHead(const std::shared_ptr<arrow::Table> &table, int numRows = 5) {
    auto dateArray   = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
    auto openArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    auto highArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
    auto lowArray    = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
    auto closeArray  = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
    auto volumeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));

    int rows = std::min(static_cast<int>(table->num_rows()), numRows);
    std::cout << std::left << std::setw(20) << "Datetime"
              << std::right << std::setw(10) << "Open"
              << std::setw(10) << "High"
              << std::setw(10) << "Low"
              << std::setw(10) << "Close"
              << std::setw(12) << "Volume" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << std::left << std::setw(20) << dateArray->GetString(i)
                  << std::right << std::setw(10) << openArray->Value(i)
                  << std::setw(10) << highArray->Value(i)
                  << std::setw(10) << lowArray->Value(i)
                  << std::setw(10) << closeArray->Value(i)
                  << std::setw(12) << volumeArray->Value(i) << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    // If run with "--server", we'll start Drogon + run backtest in a thread
    bool runServer = (argc > 1 && std::string(argv[1]) == "--server");

    spdlog::info("Starting DataLoader test");
    std::string baseDir = "../src/stock_data/";
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};

    // Load ticker data
    DataLoader loader;
    std::map<std::string, std::shared_ptr<arrow::Table>> tickerDataMap;
    for (const auto &ticker : tickers) {
        std::vector<std::string> filePaths = {
            baseDir + "time-series-" + ticker + "-5min.csv",
            baseDir + "time-series-" + ticker + "-5min(1).csv",
            baseDir + "time-series-" + ticker + "-5min(2).csv"
        };
        if (loader.loadTickerData(ticker, filePaths)) {
            auto table = loader.getTickerData(ticker);
            if (table) {
                tickerDataMap[ticker] = table;
                std::cout << "Ticker: " << ticker << "\nSchema:\n"
                          << table->schema()->ToString() << "\nFirst few rows:\n";
                printHead(table, 5);
            }
        } else {
            spdlog::error("Failed to load data for ticker: {}", ticker);
        }
    }

    // Create the BacktestEngine and set data
    spdlog::info("Testing BacktestEngine with a moving-average crossover strategy");
    BacktestEngine engine;
    engine = 150000.0; // Optionally set starting balance if you like
    engine.setTickerData(tickerDataMap);

    // Register a “historically decent” strategy: 5/20 SMA crossover
    for (const auto &ticker : tickers) {
        engine.registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
    }

    // Optionally broadcast metrics to WebSocket or logs
    engine.setBroadcastCallback([](const std::string &metrics) {
        spdlog::info("Broadcasting portfolio update: {}", metrics);
        // In production, you might do:
        //   MyWebSocketControllerInstance.broadcastJson(...)
    });

    if (runServer) {
        // Start Drogon on a fallback port
        std::vector<int> ports = {9000, 8080, 10000};
        int chosenPort = ports[0];
        drogon::app().addListener("0.0.0.0", chosenPort);
        spdlog::info("Starting server on port {}", chosenPort);

        // Run backtest in a separate thread
        std::thread engineThread([&engine]() {
            engine.runBacktest();
        });
        drogon::app().run();  // blocking
        engineThread.join();
    } else {
        // Just run the backtest in console mode
        engine.runBacktest();
        std::cout << "Final Portfolio Metrics: " << engine.getPortfolioMetrics() << std::endl;
    }

    return 0;
}
