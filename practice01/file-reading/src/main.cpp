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
#include <memory>
#include <chrono>

// Optional PyTorch support - comment out if not available
// #include <torch/torch.h>
#define PYTORCH_AVAILABLE 0

#include "DataLoader.h"
#include "BacktestEngine.h"
#include "FrontendController.h"
#include "MyWebSocketController.h"

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
        // Otherwise, hold
        tx.action = BacktestEngine::Action::Hold;
        tx.quantity = 0;
        return tx;
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

#if PYTORCH_AVAILABLE
/**
 * @brief PyTorchStrategy - A strategy that uses a PyTorch model for predictions
 * Note: This strategy is only available if PyTorch is installed
 */
class PyTorchStrategy : public BacktestEngine::Strategy {
public:
    PyTorchStrategy(const std::string &modelPath) {
        try {
            // Load the PyTorch model
            model_ = torch::jit::load(modelPath);
            model_.eval();  // Set to evaluation mode
            spdlog::info("PyTorch model loaded successfully from {}", modelPath);
            modelLoaded_ = true;
        } catch (const std::exception& e) {
            spdlog::error("Error loading PyTorch model: {}", e.what());
            modelLoaded_ = false;
        }
    }

    std::optional<BacktestEngine::Transaction> onTick(const std::string &ticker,
                                                     const std::shared_ptr<arrow::Table> &table,
                                                     size_t currentIndex,
                                                     int currentHolding) override {
        if (!modelLoaded_ || currentIndex >= table->num_rows() || currentIndex < 10) {
            return std::nullopt;
        }

        // Extract features for the model
        std::vector<float> features = extractFeatures(table, currentIndex);
        
        // Convert to PyTorch tensor
        auto inputTensor = torch::tensor(features).reshape({1, -1});
        
        // Run inference
        torch::Tensor output;
        try {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);
            output = model_.forward(inputs).toTensor();
        } catch (const std::exception& e) {
            spdlog::error("Error running PyTorch model: {}", e.what());
            return std::nullopt;
        }
        
        // Get prediction (assuming output is a single value between -1 and 1)
        float prediction = output.item<float>();
        
        // Create transaction based on prediction
        BacktestEngine::Transaction tx;
        tx.ticker = ticker;
        auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
        tx.datetime = datetimeArray->GetString(currentIndex);
        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
        tx.price = closeArray->Value(currentIndex);
        
        // Simple logic: buy if prediction > 0.5, sell if < -0.5, hold otherwise
        if (prediction > 0.5 && currentHolding == 0) {
            tx.action = BacktestEngine::Action::Buy;
            tx.quantity = 1;
            return tx;
        } else if (prediction < -0.5 && currentHolding > 0) {
            tx.action = BacktestEngine::Action::Sell;
            tx.quantity = currentHolding;
            return tx;
        } else {
            tx.action = BacktestEngine::Action::Hold;
            tx.quantity = 0;
            return tx;
        }
    }

private:
    torch::jit::script::Module model_;
    bool modelLoaded_ = false;
    
    // Extract features from the table for the model
    std::vector<float> extractFeatures(const std::shared_ptr<arrow::Table> &table, size_t currentIndex) {
        std::vector<float> features;
        
        // Get price data
        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
        auto openArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
        auto highArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
        auto lowArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
        auto volumeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));
        
        // Add current OHLCV
        features.push_back(static_cast<float>(openArray->Value(currentIndex)));
        features.push_back(static_cast<float>(highArray->Value(currentIndex)));
        features.push_back(static_cast<float>(lowArray->Value(currentIndex)));
        features.push_back(static_cast<float>(closeArray->Value(currentIndex)));
        features.push_back(static_cast<float>(volumeArray->Value(currentIndex)));
        
        // Add previous closes (last 10 bars)
        for (int i = 1; i <= 10; i++) {
            if (currentIndex >= i) {
                features.push_back(static_cast<float>(closeArray->Value(currentIndex - i)));
            } else {
                features.push_back(0.0f);
            }
        }
        
        // Calculate some technical indicators
        // SMA-5
        float sma5 = 0.0f;
        for (int i = 0; i < 5; i++) {
            if (currentIndex >= i) {
                sma5 += static_cast<float>(closeArray->Value(currentIndex - i));
            }
        }
        sma5 /= 5.0f;
        features.push_back(sma5);
        
        // SMA-10
        float sma10 = 0.0f;
        for (int i = 0; i < 10; i++) {
            if (currentIndex >= i) {
                sma10 += static_cast<float>(closeArray->Value(currentIndex - i));
            }
        }
        sma10 /= 10.0f;
        features.push_back(sma10);
        
        return features;
    }
};
#endif // PYTORCH_AVAILABLE

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

// Function to run comprehensive tests
bool runTests(bool debug = true) {
    spdlog::info("Running comprehensive tests...");
    
    // Test DataLoader
    spdlog::info("Testing DataLoader...");
    DataLoader loader;
    loader.setDebugMode(debug);
    
    std::string baseDir = "./src/stock_data/";
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};
    
    // Test loading a single ticker
    std::string testTicker = "AAPL";
    std::vector<std::string> filePaths = {
        baseDir + "time-series-" + testTicker + "-5min.csv",
        baseDir + "time-series-" + testTicker + "-5min(1).csv",
        baseDir + "time-series-" + testTicker + "-5min(2).csv"
    };
    
    bool loadSuccess = loader.loadTickerData(testTicker, filePaths);
    if (!loadSuccess) {
        spdlog::error("Failed to load ticker data for {}", testTicker);
        return false;
    }
    
    auto table = loader.getTickerData(testTicker);
    if (!table) {
        spdlog::error("Failed to get ticker data for {}", testTicker);
        return false;
    }
    
    spdlog::info("Successfully loaded {} rows for {}", table->num_rows(), testTicker);
    
    // Test loading multiple tickers concurrently
    std::unordered_map<std::string, std::vector<std::string>> tickerFilePaths;
    for (const auto &ticker : tickers) {
        tickerFilePaths[ticker] = {
            baseDir + "time-series-" + ticker + "-5min.csv",
            baseDir + "time-series-" + ticker + "-5min(1).csv",
            baseDir + "time-series-" + ticker + "-5min(2).csv"
        };
    }
    
    int loadedTickers = loader.loadMultipleTickers(tickerFilePaths);
    if (loadedTickers != tickers.size()) {
        spdlog::error("Failed to load all tickers: {} out of {}", loadedTickers, tickers.size());
        return false;
    }
    
    spdlog::info("Successfully loaded all {} tickers", loadedTickers);
    
    // Test BacktestEngine
    spdlog::info("Testing BacktestEngine...");
    auto engine = std::make_shared<BacktestEngine>();
    engine->setDebugMode(debug);
    *engine = 100000.0; // Set starting balance
    
    // Set ticker data
    engine->setTickerData(loader.getAllTickerData());
    
    // Register strategies
    for (const auto &ticker : tickers) {
        engine->registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
    }
    
    // Test step function
    spdlog::info("Testing step function...");
    auto result = engine->step();
    if (result.observations.empty()) {
        spdlog::error("Step function returned empty observations");
        return false;
    }
    
    // Test reset function
    spdlog::info("Testing reset function...");
    engine->reset();
    
    // Test running a few steps
    spdlog::info("Testing multiple steps...");
    for (int i = 0; i < 10; i++) {
        result = engine->step();
    }
    
    // Test WebSocket controller
    spdlog::info("Testing WebSocket controller...");
    auto wsController = std::make_shared<MyWebSocketController>();
    wsController->setBacktestEngine(engine);
    
    // Test Frontend controller
    spdlog::info("Testing Frontend controller...");
    auto frontendController = std::make_shared<FrontendController>();
    frontendController->setBacktestEngine(engine);
    
    // Test running a short backtest
    spdlog::info("Testing short backtest...");
    engine->reset();
    engine->runBacktest();
    
    auto metrics = engine->getPerformanceMetrics();
    spdlog::info("Backtest results: Initial=${}, Final=${}, Return={}%",
                metrics.initialBalance, metrics.finalBalance,
                metrics.totalReturn * 100.0);
    
    spdlog::info("All tests passed successfully!");
    return true;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool runServer = false;
    bool debug = false;
    bool runTestMode = false;
    int port = 8080;
    std::string strategy = "sma";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--server") {
            runServer = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--test") {
            runTestMode = true;
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy = argv[++i];
        }
    }
    
    // Run in test mode if requested
    if (runTestMode) {
        spdlog::set_level(spdlog::level::info);
        bool testsPassed = runTests(true);
        return testsPassed ? 0 : 1;
    }
    
    // Configure logging
    if (debug) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }
    
    spdlog::info("Starting Trading Environment");
    std::string baseDir = "./src/stock_data/";
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};

    // Load ticker data using concurrent loading
    spdlog::info("Loading ticker data concurrently");
    DataLoader loader;
    loader.setDebugMode(debug);
    
    std::unordered_map<std::string, std::vector<std::string>> tickerFilePaths;
    for (const auto &ticker : tickers) {
        tickerFilePaths[ticker] = {
            baseDir + "time-series-" + ticker + "-5min.csv",
            baseDir + "time-series-" + ticker + "-5min(1).csv",
            baseDir + "time-series-" + ticker + "-5min(2).csv"
        };
    }
    
    int loadedTickers = loader.loadMultipleTickers(tickerFilePaths);
    spdlog::info("Loaded {} tickers", loadedTickers);
    
    if (loadedTickers == 0) {
        spdlog::error("Failed to load any ticker data. Exiting.");
        return 1;
    }
    
    // Print sample data
    if (debug) {
        for (const auto &ticker : tickers) {
            auto table = loader.getTickerData(ticker);
            if (table) {
                std::cout << "Ticker: " << ticker << "\nSchema:\n"
                          << table->schema()->ToString() << "\nFirst few rows:\n";
                printHead(table, 5);
            }
        }
    }

    // Create the BacktestEngine and set data
    spdlog::info("Initializing BacktestEngine");
    auto engine = std::make_shared<BacktestEngine>();
    engine->setDebugMode(debug);
    *engine = 150000.0; // Set starting balance
    engine->setTickerData(loader.getAllTickerData());

    // Register strategies based on command line argument
    if (strategy == "sma") {
        spdlog::info("Using Moving Average Crossover strategy");
        for (const auto &ticker : tickers) {
            engine->registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
        }
    } else if (strategy == "pytorch") {
#if PYTORCH_AVAILABLE
        spdlog::info("Using PyTorch model strategy");
        // Note: This is a placeholder. In a real implementation, you would have a trained model file.
        // for (const auto &ticker : tickers) {
        //     engine->registerStrategy(ticker, std::make_unique<PyTorchStrategy>("path/to/model.pt"));
        // }
        spdlog::warn("PyTorch model strategy selected but no model file provided. Using SMA strategy instead.");
        for (const auto &ticker : tickers) {
            engine->registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
        }
#else
        spdlog::warn("PyTorch strategy selected but PyTorch is not available. Using SMA strategy instead.");
        for (const auto &ticker : tickers) {
            engine->registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
        }
#endif
    } else {
        spdlog::warn("Unknown strategy: {}. Using SMA strategy instead.", strategy);
        for (const auto &ticker : tickers) {
            engine->registerStrategy(ticker, std::make_unique<MovingAverageCrossoverStrategy>(5, 20));
        }
    }

    if (runServer) {
        spdlog::info("Starting server on port {}", port);
        
        // Create WebSocket controller
        auto wsController = std::make_shared<MyWebSocketController>();
        wsController->setBacktestEngine(engine);
        
        // Create Frontend controller
        auto frontendController = std::make_shared<FrontendController>();
        frontendController->setBacktestEngine(engine);
        
        // Set up the engine to broadcast to WebSocket
        engine->setJsonBroadcastCallback([wsController](const std::string& json) {
            Json::Value root;
            Json::Reader reader;
            if (reader.parse(json, root)) {
                root["type"] = "portfolio_update";
                wsController->broadcastJson(root);
            }
        });
        
        // Configure Drogon
        drogon::app().addListener("0.0.0.0", port);
        
        // Enable CORS for development
        drogon::app().registerPostHandlingAdvice(
            [](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp) {
                resp->addHeader("Access-Control-Allow-Origin", "*");
                resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
            }
        );
        
        // Start WebSocket periodic updates
        wsController->startPeriodicUpdates(1000); // Update every second
        
        // Run backtest in a separate thread
        std::thread engineThread([engine]() {
            std::this_thread::sleep_for(std::chrono::seconds(2)); // Give server time to start
            engine->runBacktest();
        });
        
        // Start the server (blocking call)
        drogon::app().run();
        
        // Clean up
        engineThread.join();
        wsController->stopPeriodicUpdates();
    } else {
        // Just run the backtest in console mode
        spdlog::info("Running backtest in console mode");
        engine->runBacktest();
        
        // Print final metrics
        auto metrics = engine->getPerformanceMetrics();
        std::cout << "=== Final Portfolio Metrics ===" << std::endl;
        std::cout << "Initial Balance: $" << metrics.initialBalance << std::endl;
        std::cout << "Final Balance: $" << metrics.finalBalance << std::endl;
        std::cout << "Total Return: " << (metrics.totalReturn * 100.0) << "%" << std::endl;
        std::cout << "Annualized Return: " << (metrics.annualizedReturn * 100.0) << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << metrics.sharpeRatio << std::endl;
        std::cout << "Max Drawdown: " << (metrics.maxDrawdown * 100.0) << "%" << std::endl;
        std::cout << "Win Rate: " << (metrics.winRate * 100.0) << "%" << std::endl;
        std::cout << "Total Trades: " << metrics.totalTrades << std::endl;
        std::cout << "Winning Trades: " << metrics.winningTrades << std::endl;
        std::cout << "Losing Trades: " << metrics.losingTrades << std::endl;
        std::cout << "Profit Factor: " << metrics.profitFactor << std::endl;
        std::cout << "Average Win: $" << metrics.averageWin << std::endl;
        std::cout << "Average Loss: $" << metrics.averageLoss << std::endl;
        std::cout << "Expectancy: $" << metrics.expectancy << std::endl;
    }

    return 0;
}
