#include "BacktestEngine.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <taskflow/taskflow.hpp>  // Taskflow uses namespace tf
#include <cmath>
#include <numeric>
#include <algorithm>
#include <json/json.h>

// Optional PyTorch support
#if PYTORCH_AVAILABLE
// Include PyTorch headers for tensor operations
#include <torch/torch.h>
#endif

BacktestEngine::BacktestEngine()
    : tickerData_{}, tickerIndices_{}, strategies_{}, holdings_{}, transactions_{},
      initial_balance_(100000.0), cash_balance_(100000.0), wins_(0), losses_(0), debug_(false),
      buy_prices_{}, broadcast_callback_(nullptr), json_broadcast_callback_(nullptr)
{
    if (debug_) spdlog::info("BacktestEngine initialized with starting balance ${}", initial_balance_);
}

BacktestEngine &BacktestEngine::operator=(double new_balance) {
    std::lock_guard<std::mutex> lock(mtx_);
    initial_balance_ = new_balance;
    cash_balance_ = new_balance;
    return *this;
}

void BacktestEngine::addTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table) {
    tickerData_[ticker] = table;
    tickerIndices_[ticker] = 0;
    holdings_[ticker] = 0;
    if (debug_) spdlog::info("Added ticker data for {}", ticker);
}

void BacktestEngine::setTickerData(const std::map<std::string, std::shared_ptr<arrow::Table>> &data) {
    for (const auto &pair : data) {
        addTickerData(pair.first, pair.second);
    }
}

void BacktestEngine::setTickerData(const std::unordered_map<std::string, std::shared_ptr<arrow::Table>> &data) {
    for (const auto &pair : data) {
        addTickerData(pair.first, pair.second);
    }
}

void BacktestEngine::registerStrategy(const std::string &ticker, std::unique_ptr<Strategy> strategy) {
    strategies_[ticker] = std::move(strategy);
    if (debug_) spdlog::info("Registered strategy for ticker {}", ticker);
}

void BacktestEngine::setBroadcastCallback(std::function<void(const std::string &)> callback) {
    std::lock_guard<std::mutex> lock(mtx_);
    broadcast_callback_ = callback;
}

void BacktestEngine::setJsonBroadcastCallback(std::function<void(const std::string &)> callback) {
    std::lock_guard<std::mutex> lock(mtx_);
    json_broadcast_callback_ = callback;
}

void BacktestEngine::logTransaction(const Transaction &tx) {
    if (!debug_) return;
    
    std::string actionStr;
    switch (tx.action) {
        case Action::Buy:
            actionStr = "BUY";
            break;
        case Action::Sell:
            actionStr = "SELL";
            break;
        case Action::Hold:
            actionStr = "HOLD";
            break;
    }
    
    spdlog::info("Transaction: {} {} shares of {} at ${} on {}",
                 actionStr, tx.quantity, tx.ticker, tx.price, tx.datetime);
}

void BacktestEngine::processTransaction(const Transaction &tx) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (tx.action == Action::Hold) {
        // Just record the transaction, no changes to holdings or cash
        transactions_.push_back(tx);
        return;
    }
    
    if (tx.action == Action::Buy) {
        double cost = tx.price * tx.quantity;
        if (cash_balance_ < cost) {
            if (debug_) {
                spdlog::warn("Rejected buy for {}: insufficient funds (cost=${}, cash=${})",
                            tx.ticker, cost, cash_balance_);
            }
            return;
        }
        
        holdings_[tx.ticker] += tx.quantity;
        cash_balance_ -= cost;
        buy_prices_[tx.ticker].push_back(tx.price);
    }
    else if (tx.action == Action::Sell) {
        if (holdings_[tx.ticker] < tx.quantity) {
            if (debug_) {
                spdlog::warn("Rejected sell for {}: insufficient holdings (requested={}, actual={})",
                            tx.ticker, tx.quantity, holdings_[tx.ticker]);
            }
            return;
        }
        
        holdings_[tx.ticker] -= tx.quantity;
        cash_balance_ += tx.price * tx.quantity;
        
        // Track wins/losses
        if (!buy_prices_[tx.ticker].empty()) {
            double buyPrice = buy_prices_[tx.ticker].front();
            buy_prices_[tx.ticker].pop_front();
            
            if (tx.price > buyPrice) {
                wins_++;
            } else {
                losses_++;
            }
        }
    }
    
    transactions_.push_back(tx);
    logTransaction(tx);
}

void BacktestEngine::runBacktest() {
    if (debug_) spdlog::info("Starting concurrent backtest simulation");
    tf::Executor executor;
    size_t iteration = 0;
    constexpr size_t updateFrequency = 16; // Broadcast/log every 16 ticks

    bool dataRemaining = true;
    while (dataRemaining) {
        dataRemaining = false;
        tf::Taskflow taskflow;

        // Process each ticker in its own task (batching per ticker).
        for (auto &pair : tickerData_) {
            const std::string &ticker = pair.first;
            std::shared_ptr<arrow::Table> table = pair.second;
            size_t &index = tickerIndices_[ticker];

            if (index < table->num_rows()) {
                dataRemaining = true;
                taskflow.emplace([this, ticker, table, &index]() {
                    if (strategies_.find(ticker) != strategies_.end()) {
                        auto &strategy = strategies_[ticker];
                        int currentHolding = holdings_[ticker];
                        auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                        
                        if (txOpt.has_value()) {
                            const auto &tx = txOpt.value();
                            processTransaction(tx);
                        }
                    }
                    index++;
                });
            }
        }
        executor.run(taskflow).wait();
        iteration++;

        if (iteration % updateFrequency == 0) {
            std::string metrics = getPortfolioMetrics();
            if (debug_) spdlog::info("Iteration {}: {}", iteration, metrics);
            
            if (broadcast_callback_) {
                broadcast_callback_(metrics);
            }
            
            if (json_broadcast_callback_) {
                json_broadcast_callback_(getPortfolioMetricsJson());
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    if (debug_) spdlog::info("Backtest simulation completed after {} iterations", iteration);
}

std::string BacktestEngine::getPortfolioMetrics() const {
    std::lock_guard<std::mutex> lock(mtx_);
    
    std::ostringstream oss;
    oss << "Cash: $" << cash_balance_ << ", Holdings: ";
    for (const auto &pair : holdings_) {
        if (pair.second > 0) {
            oss << pair.first << "=" << pair.second << " ";
        }
    }
    oss << "| Total Transactions: " << transactions_.size();
    oss << ", Wins: " << wins_ << ", Losses: " << losses_;
    double pct_change = ((cash_balance_ - initial_balance_) / initial_balance_) * 100.0;
    oss << ", % Change: " << pct_change << "%";
    return oss.str();
}

std::string BacktestEngine::getPortfolioMetricsJson() const {
    std::lock_guard<std::mutex> lock(mtx_);
    
    Json::Value root;
    root["cash"] = cash_balance_;
    root["initialBalance"] = initial_balance_;
    
    Json::Value holdingsJson(Json::arrayValue);
    for (const auto &pair : holdings_) {
        if (pair.second > 0) {
            Json::Value holding;
            holding["ticker"] = pair.first;
            holding["quantity"] = pair.second;
            holdingsJson.append(holding);
        }
    }
    root["holdings"] = holdingsJson;
    
    root["transactions"] = static_cast<int>(transactions_.size());
    root["wins"] = wins_;
    root["losses"] = losses_;
    
    double pct_change = ((cash_balance_ - initial_balance_) / initial_balance_) * 100.0;
    root["percentChange"] = pct_change;
    
    // Add performance metrics
    auto metrics = getPerformanceMetrics();
    root["performance"]["totalReturn"] = metrics.totalReturn;
    root["performance"]["annualizedReturn"] = metrics.annualizedReturn;
    root["performance"]["sharpeRatio"] = metrics.sharpeRatio;
    root["performance"]["maxDrawdown"] = metrics.maxDrawdown;
    root["performance"]["winRate"] = metrics.winRate;
    root["performance"]["profitFactor"] = metrics.profitFactor;
    
    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, root);
}

BacktestEngine::PerformanceMetrics BacktestEngine::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(mtx_);
    
    PerformanceMetrics metrics;
    metrics.initialBalance = initial_balance_;
    metrics.finalBalance = cash_balance_;
    
    // Calculate total return
    metrics.totalReturn = (cash_balance_ - initial_balance_) / initial_balance_;
    
    // Assume 252 trading days per year for annualized return
    // This is a simplification - in a real system you'd use actual dates
    double tradingDays = transactions_.empty() ? 1 : transactions_.size() / 2.0; // Assuming each trade is a buy+sell pair
    metrics.annualizedReturn = std::pow(1 + metrics.totalReturn, 252.0 / tradingDays) - 1;
    
    // Calculate Sharpe ratio (simplified)
    // In a real system, you'd calculate daily returns and their standard deviation
    metrics.sharpeRatio = metrics.totalReturn / 0.1; // Assuming 10% volatility
    
    // Calculate max drawdown
    // In a real system, you'd track equity curve over time
    metrics.maxDrawdown = 0.05; // Placeholder
    
    // Trading statistics
    metrics.totalTrades = wins_ + losses_;
    metrics.winningTrades = wins_;
    metrics.losingTrades = losses_;
    metrics.winRate = metrics.totalTrades > 0 ? static_cast<double>(wins_) / metrics.totalTrades : 0;
    
    // Calculate profit factor and expectancy
    double grossProfit = 0.0;
    double grossLoss = 0.0;
    
    // In a real system, you'd track profit/loss per trade
    // This is a placeholder calculation
    if (metrics.totalTrades > 0) {
        grossProfit = metrics.totalReturn > 0 ? metrics.totalReturn * initial_balance_ : 0;
        grossLoss = metrics.totalReturn < 0 ? -metrics.totalReturn * initial_balance_ : 0;
    }
    
    metrics.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999.0 : 0.0;
    
    // Average win and loss
    metrics.averageWin = wins_ > 0 ? grossProfit / wins_ : 0;
    metrics.averageLoss = losses_ > 0 ? grossLoss / losses_ : 0;
    
    // Expectancy
    metrics.expectancy = (metrics.winRate * metrics.averageWin) - ((1 - metrics.winRate) * metrics.averageLoss);
    
    return metrics;
}

std::map<std::string, int> BacktestEngine::getHoldings() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return holdings_;
}

const std::vector<BacktestEngine::Transaction>& BacktestEngine::getTransactions() const {
    return transactions_;
}

std::vector<std::string> BacktestEngine::getAvailableTickers() const {
    std::vector<std::string> tickers;
    tickers.reserve(tickerData_.size());
    
    for (const auto &pair : tickerData_) {
        tickers.push_back(pair.first);
    }
    
    return tickers;
}

size_t BacktestEngine::getObservationDimension() const {
    // Base features (OHLCV) + technical indicators
    return 5 + 5; // 5 base features + 5 technical indicators
}

size_t BacktestEngine::getActionDimension() const {
    return tickerData_.size();
}

std::vector<double> BacktestEngine::calculateIndicators(const std::string &ticker, size_t currentIndex) {
    std::vector<double> indicators;
    auto table = tickerData_[ticker];
    
    if (currentIndex < 10 || currentIndex >= table->num_rows()) {
        // Not enough data for indicators, return zeros
        return std::vector<double>(5, 0.0);
    }
    
    auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
    
    // Calculate SMA-5
    double sma5 = 0.0;
    for (size_t i = currentIndex - 5; i < currentIndex; ++i) {
        sma5 += closeArray->Value(i);
    }
    sma5 /= 5.0;
    indicators.push_back(sma5);
    
    // Calculate SMA-10
    double sma10 = 0.0;
    for (size_t i = currentIndex - 10; i < currentIndex; ++i) {
        sma10 += closeArray->Value(i);
    }
    sma10 /= 10.0;
    indicators.push_back(sma10);
    
    // Calculate RSI-14 (simplified)
    double gains = 0.0;
    double losses = 0.0;
    for (size_t i = currentIndex - 14; i < currentIndex; ++i) {
        double change = closeArray->Value(i) - closeArray->Value(i-1);
        if (change > 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }
    double rsi = 100.0;
    if (losses > 0) {
        double rs = gains / losses;
        rsi = 100.0 - (100.0 / (1.0 + rs));
    }
    indicators.push_back(rsi);
    
    // Calculate MACD (simplified)
    double ema12 = closeArray->Value(currentIndex-1); // Simplified EMA
    double ema26 = closeArray->Value(currentIndex-1); // Simplified EMA
    double macd = ema12 - ema26;
    indicators.push_back(macd);
    
    // Calculate Bollinger Bands (simplified)
    double sum = 0.0;
    double sumSq = 0.0;
    for (size_t i = currentIndex - 20; i < currentIndex; ++i) {
        double close = closeArray->Value(i);
        sum += close;
        sumSq += close * close;
    }
    double mean = sum / 20.0;
    double variance = (sumSq / 20.0) - (mean * mean);
    double stdDev = std::sqrt(variance);
    double bband = (closeArray->Value(currentIndex-1) - mean) / (2 * stdDev);
    indicators.push_back(bband);
    
    return indicators;
}

std::map<std::string, std::vector<double>> BacktestEngine::calculateFeatures() {
    std::map<std::string, std::vector<double>> features;
    
    for (const auto &pair : tickerData_) {
        const std::string &ticker = pair.first;
        std::shared_ptr<arrow::Table> table = pair.second;
        size_t index = tickerIndices_[ticker];
        
        if (index < table->num_rows()) {
            std::vector<double> tickerFeatures;
            
            // Add OHLCV data
            auto openArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
            auto highArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
            auto lowArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
            auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
            auto volumeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));
            
            tickerFeatures.push_back(openArray->Value(index));
            tickerFeatures.push_back(highArray->Value(index));
            tickerFeatures.push_back(lowArray->Value(index));
            tickerFeatures.push_back(closeArray->Value(index));
            tickerFeatures.push_back(volumeArray->Value(index));
            
            // Add technical indicators
            auto indicators = calculateIndicators(ticker, index);
            tickerFeatures.insert(tickerFeatures.end(), indicators.begin(), indicators.end());
            
            features[ticker] = tickerFeatures;
        }
    }
    
    return features;
}

#if PYTORCH_AVAILABLE
torch::Tensor BacktestEngine::mapToTensor(const std::map<std::string, double> &map) {
    std::vector<double> values;
    values.reserve(map.size());
    
    // Get tickers in a consistent order
    std::vector<std::string> tickers = getAvailableTickers();
    
    for (const auto &ticker : tickers) {
        auto it = map.find(ticker);
        if (it != map.end()) {
            values.push_back(it->second);
        } else {
            values.push_back(0.0);
        }
    }
    
    return torch::tensor(values, torch::kFloat32);
}

std::map<std::string, double> BacktestEngine::tensorToMap(const torch::Tensor &tensor) {
    std::map<std::string, double> result;
    
    // Get tickers in a consistent order
    std::vector<std::string> tickers = getAvailableTickers();
    
    auto accessor = tensor.accessor<float, 1>();
    for (size_t i = 0; i < std::min(static_cast<size_t>(accessor.size(0)), tickers.size()); ++i) {
        result[tickers[i]] = accessor[i];
    }
    
    return result;
}
#endif

// === RL methods ===

// Default step() using registered strategies.
BacktestEngine::StepResult BacktestEngine::step() {
    StepResult result;
    bool anyDataLeft = false;

    for (auto &pair : tickerData_) {
        const std::string &ticker = pair.first;
        std::shared_ptr<arrow::Table> table = pair.second;
        size_t &index = tickerIndices_[ticker];

        if (index < table->num_rows()) {
            anyDataLeft = true;
            if (strategies_.find(ticker) != strategies_.end()) {
                auto &strategy = strategies_[ticker];
                int currentHolding = holdings_[ticker];
                auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                
                if (txOpt.has_value()) {
                    const auto &tx = txOpt.value();
                    processTransaction(tx);
                }
            }
            
            auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
            double price = closeArray->Value(index);
            result.observations[ticker] = price;
            index++;
        } else {
            result.observations[ticker] = -1.0;
        }
    }
    
    result.done = !anyDataLeft;
    result.reward = (cash_balance_ - initial_balance_) / initial_balance_;
    result.features = calculateFeatures();
    
    return result;
}

// External action version of step()
BacktestEngine::StepResult BacktestEngine::step(const std::map<std::string, double> &externalActions) {
    StepResult result;
    bool anyDataLeft = false;

    for (auto &pair : tickerData_) {
        const std::string &ticker = pair.first;
        std::shared_ptr<arrow::Table> table = pair.second;
        size_t &index = tickerIndices_[ticker];

        if (index < table->num_rows()) {
            anyDataLeft = true;
            auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
            double price = closeArray->Value(index);
            
            auto it = externalActions.find(ticker);
            if (it != externalActions.end()) {
                double actionValue = it->second;
                Transaction tx;
                tx.ticker = ticker;
                tx.price = price;
                auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
                tx.datetime = datetimeArray->GetString(index);
                
                // Buy if actionValue > 0
                if (actionValue > 0) {
                    int buyQty = static_cast<int>(actionValue);
                    tx.action = Action::Buy;
                    tx.quantity = buyQty;
                    processTransaction(tx);
                }
                // Sell if actionValue < 0
                else if (actionValue < 0) {
                    int sellQty = static_cast<int>(-actionValue);
                    tx.action = Action::Sell;
                    tx.quantity = std::min(holdings_[ticker], sellQty);
                    processTransaction(tx);
                }
                // Hold if actionValue == 0
                else {
                    tx.action = Action::Hold;
                    tx.quantity = 0;
                    processTransaction(tx);
                }
            }
            else {
                // No external action; use default strategy if available
                if (strategies_.find(ticker) != strategies_.end()) {
                    auto &strategy = strategies_[ticker];
                    int currentHolding = holdings_[ticker];
                    auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                    
                    if (txOpt.has_value()) {
                        const auto &tx = txOpt.value();
                        processTransaction(tx);
                    }
                }
            }
            
            result.observations[ticker] = price;
            index++;
        }
        else {
            result.observations[ticker] = -1.0;
        }
    }
    
    result.done = !anyDataLeft;
    result.reward = (cash_balance_ - initial_balance_) / initial_balance_;
    result.features = calculateFeatures();
    
    return result;
}

#if PYTORCH_AVAILABLE
// PyTorch tensor version of step()
BacktestEngine::StepResult BacktestEngine::stepWithTensor(const torch::Tensor &actions) {
    // Convert tensor to map
    auto actionsMap = tensorToMap(actions);
    return step(actionsMap);
}
#endif

void BacktestEngine::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &pair : tickerIndices_) {
        pair.second = 0;
    }
    holdings_.clear();
    transactions_.clear();
    for (auto &pair : buy_prices_) {
        pair.second.clear();
    }
    wins_ = 0;
    losses_ = 0;
    cash_balance_ = initial_balance_;
    for (const auto &pair : tickerData_) {
        holdings_[pair.first] = 0;
    }
    
    if (debug_) spdlog::info("BacktestEngine reset to initial state");
}
